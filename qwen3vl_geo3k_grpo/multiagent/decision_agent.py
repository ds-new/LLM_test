"""决策智能体：编排其余四个智能体，审核报告，按明确策略决策并把理由写入 events.jsonl（type=decision）。

阶段：env_check → data_acquire → data_clean → baseline_eval → difficulty_filter
      → [plan_round → train_round → review_round] × N → final_eval → done

决策策略：
  1. 准入：GPU / 依赖 / 磁盘 / 奖励函数自检任一不通过即停止，不浪费计费时间；
  2. 数据：清洗报告中训练集损失超过 15% 或 val/test 存在不可判分题时记录警告；难度过滤后不足下限则放弃过滤；
  3. 每轮训练中（每个验证点）：
       - 新最佳 → 让训练智能体钉住该检查点；
       - 连续 patience 次验证无提升（> min_delta）且已过 min_steps → 提前结束本轮（省钱）；
       - 连续两次比最佳低 diverge_drop 个百分点 → 判定退化，结束本轮；
       - 到达本轮截止时间 → 结束本轮；
  4. 每轮结束：导出最佳检查点，用独立评估流程在 val 上复核并与基线逐题对比；
     由架构智能体依据诊断（熵塌缩、截断率、提升幅度）提出下一轮算法，预算不足则直接进入最终评估；
  5. 最终：只用 val 选出的最佳模型在 test 上评估一次，与基线模型的 test 结果对比，写报告。
"""

import re
import shutil
import subprocess
import traceback
from pathlib import Path

from .architect_agent import PRESETS, ArchitectAgent, describe_presets
from .common import (DRY_RUN, RUN_DIR, Blackboard, Bus, RestartRequested, StopRequested, control_signal, now,
                     read_json, stamp, write_text)
from .data_agent import DataAgent
from .eval_agent import EvalAgent
from .trainer_agent import TrainerAgent, val_accuracy

NAME = "decision"


class DecisionAgent:
    def __init__(self, config, bus=None, board=None):
        self.config = config
        self.bus = bus or Bus()
        self.board = board or Blackboard()
        if not self.board.get("started"):
            start = now()
            end = start + config["budget_hours"] * 3600
            self.board.update(started=start, started_text=stamp(start), deadline=end, deadline_text=stamp(end),
                              phase="env_check", rounds=[], decisions=0)
        self.deadline = self.board.get("deadline")
        self.data = DataAgent(config, self.bus)
        self.architect = ArchitectAgent(config, self.bus)
        self.trainer = TrainerAgent(config, self.bus, self.board)
        self.evaluator = EvalAgent(config, self.bus)

    # ------------------------------------------------------------------ 工具
    def decide(self, recipient, what, why, **detail):
        self.board.set("decisions", self.board.get("decisions", 0) + 1)
        return self.bus.post(NAME, recipient, "decision", {"decision": what, "why": why, **detail})

    def check_control(self):
        sig = control_signal()
        if sig == "stop":
            raise StopRequested("STOP file present")
        if sig == "restart":
            raise RestartRequested("RESTART file present")

    def remaining(self):
        return self.deadline - now()

    def hours(self, h):
        return h * 3600

    # ------------------------------------------------------------------ 主循环
    def run(self):
        while True:
            self.check_control()
            phase = self.board.get("phase")
            if phase == "done":
                self.write_report()
                return
            if self.remaining() <= self.hours(self.config["final_reserve_hours"]) and phase not in (
                    "final_eval", "review_round", "env_check", "data_acquire", "data_clean", "baseline_eval"):
                if phase in ("plan_round", "difficulty_filter", "train_round"):
                    self.decide("all", "budget_reserve", "剩余预算只够最终评估，停止新的训练")
                    self.board.set("phase", "final_eval" if phase != "train_round" else "review_round")
                    continue
            self.bus.post(NAME, "all", "phase", {"phase": phase, "remaining_hours": round(self.remaining() / 3600, 2)})
            nxt = getattr(self, "phase_" + phase)()
            self.board.update(phase=nxt, phase_failures=0)
            self.write_report()

    # ------------------------------------------------------------------ 准入
    def phase_env_check(self):
        self.bus.post(NAME, "trainer", "directive", {"task": "env_check"})
        env = self.trainer.env_check()
        self.board.update(env=env)
        self.bus.post(NAME, "architect", "directive", {"task": "reward_selftest"})
        reward = self.architect.reward_selftest()
        problems = list(env["problems"])
        if not reward["passed"]:
            problems.append(f"奖励函数自检失败: {reward['failures']}")
        if problems:  # 先查环境再下载模型：磁盘/依赖不满足时不浪费时间下载 9GB 权重
            self.decide("all", "abort", "准入检查未通过，避免浪费 GPU 计费时间", problems=problems)
            raise StopRequested("; ".join(problems))
        self.bus.post(NAME, "architect", "directive", {"task": "ensure_model + probe_verl"})
        model = self.architect.ensure_model()
        caps = self.architect.probe_verl()
        self.board.update(model_info=model, verl={k: v for k, v in caps.items() if k != "keys"},
                          reward_selftest=reward)
        # 准入通过才开始计预算（之前因环境问题中止的时间不计入）
        start = now()
        self.deadline = start + self.config["budget_hours"] * 3600
        self.board.update(started=start, started_text=stamp(start), deadline=self.deadline,
                          deadline_text=stamp(self.deadline))
        if not caps.get("ok"):
            err = caps.get("error") or ""
            if any(k in err for k in ("Error:", "Traceback")) and not self.config["train"].get("allow_legacy_overrides"):
                # verl 连配置都打印不出来时训练必然失败，此时继续只会浪费计费时间
                self.decide("all", "abort", "verl 无法运行（--cfg job 报错），训练必然失败；修复环境后重新启动",
                            error=err[-600:])
                raise StopRequested("verl --cfg job failed")
            self.decide("trainer", "legacy_overrides", "无法读取 verl 配置树，退回 ++key=value 方式（拼错的键不会报错，请留意日志）",
                        error=err)
        self.decide("data", "start_data", f"环境就绪：{len(env['gpus'])}×{env['gpus'][0]['name']}，"
                    f"数据盘剩余 {env['disk_free_gb']}GB，模型 {model['path']}（约 {model['params_b']}B 参数），"
                    f"rollout 引擎 {env.get('rollout_engine')}，"
                    f"可用算法预设 {caps.get('available_presets', ['grpo'])}")
        return "data_acquire"

    # ------------------------------------------------------------------ 数据
    def phase_data_acquire(self):
        self.bus.post(NAME, "data", "directive", {"task": "acquire", "source": self.config["data"]["source"]})
        out = self.data.acquire()
        self.board.set("acquire", out)
        return "data_clean"

    def phase_data_clean(self):
        self.bus.post(NAME, "data", "directive", {"task": "clean"})
        rep = self.data.clean()
        tr = rep["splits"]["train"]
        loss = 1 - tr["kept"] / max(1, tr["input"])
        if loss > 0.15:
            self.decide("data", "warn_data_loss", f"清洗剔除了 {loss:.1%} 的训练题，请检查 clean_report.json",
                        issues=tr["issues"])
        for s in ("val", "test"):
            bad = rep["splits"][s]["issues"].get("unverifiable_gt_kept", 0)
            if bad:
                self.decide("evaluator", "note_eval_noise", f"{s} 中有 {bad} 题标准答案无法被判分器识别（保留以便与公开结果可比，准确率上限相应降低）")
        tok = rep.get("prompt_tokens", {})
        max_prompt = self.config["train"]["max_prompt_length"]
        if tok.get("measured"):
            max_prompt = tok["recommended_max_prompt_length"]
            self.decide("trainer", "set_max_prompt_length", f"实测 prompt 最长 {tok['max']} token（p99 {tok['p99']}），"
                        f"max_prompt_length 设为 {max_prompt}")
        self.board.update(clean=rep, max_prompt_length=max_prompt)
        return "baseline_eval"

    def gen_len(self):
        """训练 max_response_length 与评估 max_new_tokens 始终相同，保证基线与训练后模型可比。"""
        return int(self.board.get("gen_len") or self.config["train"]["max_response_length"])

    def phase_baseline_eval(self):
        base = self.config["model"]["path"]
        cap = int(self.config["train"].get("max_response_length_cap", 4096))
        # 先在 val 上确定生成长度：截断比例过高说明长度不够，截断的回答奖励为 0，会把基线和训练信号都压低
        while True:
            self.bus.post(NAME, "evaluator", "directive", {"task": "evaluate_base", "split": "val", "max_new_tokens": self.gen_len()})
            r = self.evaluator.evaluate(base, self.data.path("clean", "val"), "base_val", max_new_tokens=self.gen_len())
            trunc = r["summary"].get("truncated_rate", 0)
            if trunc > 10 and self.gen_len() < cap:
                new = min(cap, self.gen_len() * 2)
                self.decide("all", "raise_generation_length",
                            f"基线在 val 上 {trunc}% 的回答在 {self.gen_len()} token 处被截断（平均 {r['summary'].get('mean_chars')} 字符），"
                            f"训练与评估的生成长度统一提高到 {new}，并重新评估基线", old=self.gen_len(), new=new)
                self.board.set("gen_len", new)
                continue
            if trunc > 10:
                self.decide("all", "note_truncation", f"生成长度已到上限 {cap}，仍有 {trunc}% 截断；训练会自然压缩回答长度")
            break
        res = {"val": r["summary"]["accuracy"]}
        self.bus.post(NAME, "evaluator", "directive", {"task": "evaluate_base", "split": "test", "max_new_tokens": self.gen_len()})
        r = self.evaluator.evaluate(base, self.data.path("clean", "test"), "base_test", max_new_tokens=self.gen_len())
        res["test"] = r["summary"]["accuracy"]
        res["gen_len"] = self.gen_len()
        self.board.set("baseline", res)
        self.decide("all", "baseline_recorded", f"基线模型 val {res['val']}%，test {res['test']}%（生成长度 {self.gen_len()}；"
                    f"test 仅用于最终对照）")
        return "difficulty_filter"

    def phase_difficulty_filter(self):
        dc = self.config["data"]
        backend = self.config["eval"]["backend"]
        if not dc["difficulty_filter"]:
            self.decide("data", "skip_difficulty_filter", "配置关闭了难度过滤")
            return "plan_round"
        pk = (self.board.get("env") or {}).get("packages", {})
        no_engine = backend == "auto" and not (pk.get("sglang") or pk.get("vllm"))
        if (backend == "hf" or no_engine) and not DRY_RUN:
            self.decide("data", "skip_difficulty_filter", "评估后端为 HF 逐批生成，训练集 k 次采样耗时过长，跳过难度过滤")
            return "plan_round"
        k = dc["difficulty_samples"]
        self.bus.post(NAME, "evaluator", "directive", {"task": "train_pass_rate", "k": k})
        r = self.evaluator.evaluate(self.config["model"]["path"], self.data.path("clean", "train"), f"base_train_k{k}",
                                    n_samples=k, temperature=1.0, max_new_tokens=self.gen_len())
        self.bus.post(NAME, "data", "directive", {"task": "difficulty_filter"})
        out = self.data.difficulty_filter(self.evaluator.dir / f"base_train_k{k}.json")
        self.board.set("difficulty", out)
        if out["applied"]:
            self.decide("trainer", "use_filtered_train",
                        f"基线对 {out['dropped_all_correct']} 道题 {k} 次全对（组内优势恒为 0），剔除后训练集 {out['kept']} 题；"
                        f"全错题 {out['all_wrong_kept']} 道保留（可能随训练变得可学）",
                        hist=r["summary"].get("pass_rate_hist"))
        else:
            self.decide("trainer", "keep_full_train", out.get("reason", ""))
        return "plan_round"

    # ------------------------------------------------------------------ 轮次
    def phase_plan_round(self):
        rounds = self.board.get("rounds", [])
        acfg = self.config["algorithm"]
        if len(rounds) >= acfg["max_rounds"]:
            self.decide("evaluator", "final", f"已完成 {len(rounds)} 轮（上限 {acfg['max_rounds']}），进入最终评估")
            return "final_eval"
        reserve = self.config["final_reserve_hours"] + (0.5 if not DRY_RUN else 0.01)
        avail = self.remaining() - self.hours(reserve)
        min_round = 1.5 if not DRY_RUN else 0.002
        if avail < self.hours(min_round):
            self.decide("evaluator", "final", f"剩余可训练时间 {avail / 3600:.1f}h 不足一轮，进入最终评估")
            return "final_eval"
        self.bus.post(NAME, "architect", "directive", {"task": "propose_algorithm", "presets": describe_presets()})
        forced = acfg.get("next_preset")
        prop = self.architect.propose(rounds)
        if forced and forced in PRESETS:
            prop = {"preset": forced, **PRESETS[forced], "max_response_length": self.gen_len(),
                    "rationale": f"按用户要求继续用 {forced}（{PRESETS[forced]['desc']}）"}
        if prop is None:
            self.decide("evaluator", "final", "所有可用预设都已尝试，进入最终评估")
            return "final_eval"
        if prop.get("max_response_length", 0) < self.gen_len():
            prop["max_response_length"] = self.gen_len()
        self.board.set("gen_len", max(self.gen_len(), prop["max_response_length"]))
        self.bus.post("architect", NAME, "proposal", prop)
        data = {"train_file": self.data.train_file(), "val_file": self.data.path("clean", "val"),
                "max_prompt_length": self.board.get("max_prompt_length")}
        if acfg.get("continue_from_best"):
            prev = max([r for r in rounds if r.get("model") and Path(r["model"]).exists()],
                       key=lambda r: r.get("val_acc") or 0, default=None)
            if prev:
                data["init_model"] = prev["model"]
                prop["rationale"] += f"；从第 {prev['round']} 轮最佳模型（val {prev['val_acc']}%）继续训练"
        level = self.board.get("memory_level", 0)  # 同一台机器上已经验证过的显存级别，避免每轮重新 OOM
        plan = self.trainer.make_plan(len(rounds) + 1, prop, data, self.board.get("model_info", {}), level=level)
        round_end = min(now() + self.hours(self.config["train"]["round_hours_max"]), now() + avail)
        self.board.set("current", {"plan": plan, "val_history": [], "best": None, "notes": [], "train_stats": {},
                                   "started": stamp(), "deadline": round_end, "deadline_text": stamp(round_end)})
        if plan.get("attn_implementation") == "sdpa":
            self.decide("trainer", "use_sdpa", "flash-attn 的 CUDA 扩展不可用，训练改用 PyTorch SDPA 注意力并关闭 remove_padding"
                        "（速度约慢 10%～30%；装好匹配的 flash-attn 后会自动切回）")
        self.decide("trainer", "approve_plan", prop["rationale"] + (f"；{plan['warning']}" if plan.get("warning") else ""),
                    plan={k: plan[k] for k in ("round", "preset", "n_gpus", "train_batch_size", "n_samples", "lr",
                                                "max_prompt_length", "max_response_length", "total_epochs",
                                                "keep_ckpts", "save_optimizer")},
                    memory=plan["memory"], round_deadline=stamp(round_end))
        return "train_round"

    def phase_train_round(self):
        cur = self.board.get("current")
        plan = cur["plan"]
        self.bus.post(NAME, "trainer", "directive", {"task": "train_round", "round": plan["round"],
                                                     "until": cur["deadline_text"]})
        res = self.trainer.run_round(plan, cur["deadline"], self.review_step, cur.get("attempt_state"))
        cur = self.board.get("current")
        cur["plan"] = plan
        cur["result"] = {k: v for k, v in res.items() if k != "attempt_state"}
        cur["attempt_state"] = res.get("attempt_state")
        self.board.set("current", cur)
        if res.get("attempt_state"):
            self.board.set("memory_level", max(self.board.get("memory_level", 0), res["attempt_state"]["level"]))
        if res["status"] == "interrupted":
            self.check_control()
        if res["status"] == "failed" and res.get("crash_class") in ("env_error", "config_error"):
            # 环境/配置类错误：不进入下一轮（下一轮会以同样方式失败并耗尽预算），保留本轮计划后停止
            tail = res.get("log_tail") or ""
            key = [l for l in tail.splitlines() if re.search(r"Error|error:", l)][-3:]
            cur["attempt_state"] = None
            cur.pop("result", None)
            self.board.set("current", cur)
            self.decide("all", "abort", f"训练因{'环境' if res['crash_class'] == 'env_error' else '配置'}错误失败，"
                        f"重试无意义；修复后重新启动 run_loop.sh 会从本轮开头继续", error_lines=key, log=res.get("log"))
            raise StopRequested(f"training {res['crash_class']}: {' | '.join(key)[-300:]}")
        return "review_round"

    def review_step(self, step, metrics):
        """训练智能体每解析到一步就调用；返回 "stop:<原因>" 让其结束本轮。"""
        cur = self.board.get("current")
        ts = cur["train_stats"]
        for key, name in (("actor/entropy", "entropy"), ("response_length/clip_ratio", "clip_ratio"),
                          ("critic/score/mean", "score"), ("timing_s/step", "step_seconds")):
            if key in metrics:
                ts.setdefault(name, []).append([step, round(metrics[key], 4)])
                ts[name] = ts[name][-400:]
        acc, kind, key = val_accuracy(metrics)
        verdict = None
        if acc is not None:
            tcfg = self.config["train"]
            hist = cur["val_history"]
            hist.append({"step": step, "acc": round(acc, 2), "kind": kind, "key": key, "time": stamp()})
            best = cur["best"]
            improved = best is None or acc > best["acc"] + tcfg["min_delta"]
            if improved or best is None:
                cur["best"] = {"step": step, "acc": round(acc, 2)}
                if step > 0:
                    pinned = self.trainer.pin_checkpoint(cur["plan"], step)
                    if pinned:
                        cur["best_pinned"] = {"step": step, "acc": round(acc, 2), "pinned": pinned}
                        self.decide("trainer", "pin_best", f"第 {step} 步验证准确率 {acc:.2f}%（{kind}）创新高，钉住该检查点",
                                    pinned=pinned)
                    else:
                        self.decide("trainer", "pin_missed", f"第 {step} 步创新高（{acc:.2f}%）但检查点已被轮转删除，"
                                    f"导出仍使用上一个钉住的检查点；可调大 max_actor_ckpt_to_keep 或 save_freq")
                cur["since_best"] = 0
            else:
                cur["since_best"] = cur.get("since_best", 0) + 1
                drop = cur["best"]["acc"] - acc
                cur["bad_streak"] = cur.get("bad_streak", 0) + 1 if drop >= tcfg["diverge_drop"] else 0
                if cur["bad_streak"] >= 2:
                    verdict = f"stop:diverged（连续两次比最佳 {cur['best']['acc']}% 低 {tcfg['diverge_drop']} 个点以上）"
                elif cur["since_best"] >= tcfg["patience"] and step >= tcfg["min_steps"]:
                    verdict = f"stop:plateau（连续 {cur['since_best']} 次验证未超过最佳 {cur['best']['acc']}%）"
            if verdict:
                self.decide("trainer", "early_stop", verdict[5:], step=step)
        self.board.set("current", cur)
        return verdict

    def diagnostics(self, ts):
        def avg(rows):
            return round(sum(v for _, v in rows) / len(rows), 4) if rows else None
        ent = ts.get("entropy", [])
        clip = ts.get("clip_ratio", [])
        e0, e1 = avg(ent[:3]), avg(ent[-3:])
        return {"entropy_first": e0, "entropy_last": e1,
                "entropy_collapse": bool(e0 and e1 is not None and e1 < 0.3 * e0),
                "response_clip_ratio": avg(clip[-5:]), "score_first": avg(ts.get("score", [])[:3]),
                "score_last": avg(ts.get("score", [])[-3:]), "step_seconds": avg(ts.get("step_seconds", [])[-10:])}

    def phase_review_round(self):
        cur = self.board.get("current")
        plan = cur["plan"]
        res = cur.get("result") or {"status": "interrupted_before_result"}
        diag = self.diagnostics(cur.get("train_stats", {}))
        record = {"round": plan["round"], "preset": plan["preset"], "plan": plan, "status": res.get("status"),
                  "stop_reason": res.get("reason"), "val_history": cur["val_history"], "diagnostics": diag,
                  "verl_best": cur.get("best"), "exported_step": (cur.get("best_pinned") or {}).get("step"),
                  "finished": stamp()}
        best = cur.get("best_pinned")
        if best and Path(best["pinned"]).exists():
            target = Path(plan["round_dir"]) / "best_hf"
            ck = Path(plan["save_dir"])  # 先删轮转检查点腾空间（最佳检查点已用硬链接钉住）
            if ck.exists() and ck.resolve().is_relative_to(RUN_DIR.resolve()):
                shutil.rmtree(ck, ignore_errors=True)
            self.bus.post(NAME, "evaluator", "directive", {"task": "export_and_verify", "step": best["step"]})
            model = self.evaluator.export_hf(best["pinned"], target)
            r = self.evaluator.evaluate(model, self.data.path("clean", "val"), f"round{plan['round']}_val",
                                        max_new_tokens=self.gen_len())
            base = read_json(self.evaluator.dir / "base_val.json")
            cmp = self.evaluator.compare(base, r)
            record.update(model=model, val_acc=r["summary"]["accuracy"], val_vs_base=cmp,
                          val_gain=round(r["summary"]["accuracy"] - base["summary"]["accuracy"], 2),
                          eval_summary=self.evaluator._brief(r["summary"]))
            self.decide("all", "round_result",
                        f"第 {plan['round']} 轮（{plan['preset']}）最佳 step {best['step']}：val {r['summary']['accuracy']}%，"
                        f"较基线 {record['val_gain']:+} 个点（修复 {cmp['fixed']} / 弄坏 {cmp['broken']}，"
                        f"{'显著' if cmp['significant_95'] else '不显著'}）", diagnostics=diag)
            self._free_round_disk(plan)
        else:
            record.update(model=None, val_acc=None, val_gain=None)
            self.decide("all", "round_no_gain", f"第 {plan['round']} 轮没有超过训练前验证分数的检查点（状态 {res.get('status')}）",
                        diagnostics=diag, log_tail=(res.get("log_tail") or "")[-600:])
        rounds = self.board.get("rounds", [])
        rounds.append(record)
        self.board.update(rounds=rounds, current=None, train_attempts=None)
        if res.get("status") == "failed" and res.get("crash_class") in ("disk_full", "host_oom"):
            self.decide("evaluator", "final", f"训练失败（{res.get('crash_class')}），无法通过降级恢复，直接进入最终评估")
            return "final_eval"
        return "plan_round"

    def _free_round_disk(self, plan):
        """本轮最佳已导出为 HF 目录，删除轮转检查点与钉住分片，为下一轮腾出数据盘。"""
        rdir = Path(plan["round_dir"])
        freed = []
        for sub in ("ckpt", "pinned"):
            p = rdir / sub
            if p.exists() and p.resolve().is_relative_to(RUN_DIR.resolve()):
                shutil.rmtree(p, ignore_errors=True)
                freed.append(str(p))
        self.bus.post(NAME, "trainer", "status", {"action": "free_disk", "removed": freed})

    # ------------------------------------------------------------------ 最终评估
    def phase_final_eval(self):
        if self.board.get("current"):
            return "review_round"
        rounds = [r for r in self.board.get("rounds", []) if r.get("val_acc") is not None]
        baseline = self.board.get("baseline") or {}
        best = max(rounds, key=lambda r: r["val_acc"], default=None)
        if not best or best["val_acc"] <= baseline.get("val", 0):
            self.decide("all", "complete_no_improvement",
                        f"没有任何一轮在 val 上超过基线（{baseline.get('val')}%），不在 test 上评估训练模型")
            self.board.set("final", {"best": None, "baseline": baseline})
            return self._finish()
        self.bus.post(NAME, "evaluator", "directive", {"task": "final_test", "round": best["round"]})
        r = self.evaluator.evaluate(best["model"], self.data.path("clean", "test"), "final_test",
                                    max_new_tokens=self.gen_len())
        base = read_json(self.evaluator.dir / "base_test.json")
        cmp = self.evaluator.compare(base, r)
        final = {"best_round": best["round"], "preset": best["preset"], "model": best["model"],
                 "val_acc": best["val_acc"], "test_acc": r["summary"]["accuracy"], "test_vs_base": cmp,
                 "baseline": baseline, "test_summary": self.evaluator._brief(r["summary"])}
        self.board.set("final", final)
        self.decide("all", "complete", f"最佳为第 {best['round']} 轮（{best['preset']}）：test {final['test_acc']}%，"
                    f"基线 {baseline.get('test')}%（修复 {cmp['fixed']} / 弄坏 {cmp['broken']}，z={cmp['sign_test_z']}）；"
                    f"模型在 {best['model']}")
        return self._finish()

    def _finish(self):
        self.write_report()
        if self.config.get("shutdown_when_done") and not DRY_RUN:
            self.decide("all", "shutdown", "按配置在完成后关机以停止 AutoDL 计费")
            subprocess.Popen(["bash", "-c", "sleep 60 && shutdown -h now"])
        return "done"

    # ------------------------------------------------------------------ 报告
    def write_report(self):
        try:
            write_text(RUN_DIR / "REPORT.md", render_report(self.board.data, self.bus))
        except Exception:
            traceback.print_exc()


def render_report(state, bus):
    L = ["# Qwen3-VL-4B geometry3k 多智能体 GRPO 报告", "",
         f"- 更新：{state.get('updated', stamp())}　开始：{state.get('started_text')}　截止：{state.get('deadline_text')}",
         f"- 当前阶段：**{state.get('phase')}**", ""]
    env = state.get("env") or {}
    if env.get("gpus"):
        g = env["gpus"]
        L += [f"- 硬件：{len(g)} × {g[0]['name']}（{g[0]['mem_gb']}GB），数据盘剩余 {env.get('disk_free_gb')}GB",
              f"- verl：{(state.get('verl') or {}).get('verl_version')}（配置模式 {(state.get('verl') or {}).get('mode')}）", ""]
    base = state.get("baseline")
    final = state.get("final")
    if base:
        L += ["## 结果", "", "| 模型 | val 准确率 | test 准确率 |", "|---|---|---|",
              f"| 基线（未训练） | {base.get('val')}% | {base.get('test')}% |"]
        for r in state.get("rounds", []):
            test_cell = f"{final['test_acc']}%" if final and final.get("best_round") == r["round"] else "—"
            L.append(f"| 第 {r['round']} 轮 {r['preset']} | {r.get('val_acc', '—')}% | {test_cell} |")
        L.append("")
        if final and final.get("test_vs_base"):
            c = final["test_vs_base"]
            L += [f"test 逐题对比：修复 {c['fixed']} 题、弄坏 {c['broken']} 题，符号检验 z = {c['sign_test_z']}"
                  f"（{'95% 显著' if c['significant_95'] else '不显著'}）。最佳模型：`{final.get('model')}`", ""]
    clean = state.get("clean")
    if clean:
        L += ["## 数据清洗", "", "| 划分 | 输入 | 保留 | 问题 |", "|---|---|---|---|"]
        for s, v in clean["splits"].items():
            L.append(f"| {s} | {v['input']} | {v['kept']} | {v['issues']} |")
        diff = state.get("difficulty")
        if diff:
            L.append(f"\n难度过滤：{diff}")
        L.append("")
    cur = state.get("current")
    if cur:
        p = cur["plan"]
        L += [f"## 进行中：第 {p['round']} 轮（{p['preset']}）", "",
              f"- {p['n_gpus']} 卡，batch {p['train_batch_size']} × n {p['n_samples']}，lr {p['lr']}，"
              f"prompt/response {p['max_prompt_length']}/{p['max_response_length']}，显存级别 L{p['memory']['level']}",
              f"- 本轮截止：{cur.get('deadline_text')}；最佳：{cur.get('best')}",
              "- 验证曲线：" + "，".join(f"{h['step']}:{h['acc']}" for h in cur["val_history"][-20:]), ""]
    for r in state.get("rounds", []):
        L += [f"## 第 {r['round']} 轮：{r['preset']}（{r.get('status')}）", "",
              f"- 方案理由：{r['plan'].get('rationale')}",
              f"- 结束原因：{r.get('stop_reason') or r.get('status')}；诊断：{r.get('diagnostics')}",
              "- 验证曲线：" + "，".join(f"{h['step']}:{h['acc']}" for h in r.get("val_history", [])), ""]
    L += ["## 决策记录（最近 30 条）", ""]
    for e in bus.tail(30, kinds=("decision",)):
        c = e["content"]
        L.append(f"- {e['time']} **{c.get('decision')}** → {e['to']}：{c.get('why')}")
    L.append("")
    return "\n".join(L)
