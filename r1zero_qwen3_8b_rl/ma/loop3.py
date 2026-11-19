"""第三轮多智能体 loop（决策智能体主导），第二轮结束后手动启动：

  R1Z_STAGE=16k（默认）：从第二轮的最佳 checkpoint 出发，上下文 16K + 动态采样，再训 R1Z_EXTRA_STEPS 步（默认 100）。
      评选分数按 16K 生成长度重新计算：起点步先评一次，作为本阶段的基线；之后只有超过它才替换系统盘上的最佳。
      开训前删掉数据盘上第二轮的最新 checkpoint（只留最新与最佳的规则：新阶段的最新在 16K 目录里）。
  R1Z_STAGE=dyn8k：仍是 8K 上下文，只加动态采样，从第二轮最新 checkpoint 接着训 R1Z_EXTRA_STEPS 步；
      评选设置和第二轮相同，最佳 checkpoint 直接沿用。

启动：  R1Z_STAGE=16k setsid nohup bash run_loop3.sh &   然后   setsid nohup bash watchdog3.sh &
完成或无法恢复时的处理与第二轮相同（完成 → 最终评估 → 报告 → 关机；失败 → 10 分钟宽限后关机）。
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from .analysis_agent import AnalysisAgent
from .common import RUN, Board, now, read_json
from .data_agent import DataAgent
from .env_agent import EnvAgent
from .eval_agent import BEST_ROOT, EvalAgent, ckpt_complete, ckpt_steps, dir_gb, latest_saved, save_space
from .framework_agent import CKPT_DIR, CKPT_DIR_16K, EXP, EXP_16K, FrameworkAgent, levels_for, plan
from .loop2 import finish, pause_for_claude
from .trainer_agent import TrainerAgent

NAME = "decision"
# 第三轮内的各阶段（2026-11-17）：阶段名 -> (plan 种类, 设置说明)
R3_STAGES = {
    "upd8": ("formal3a", "8K 上下文；每批数据更新 8 次（原 2 次）"),
    "upd8dyn": ("formal3b", "8K 上下文；每批数据更新 8 次（原 2 次）＋DAPO 动态采样"),
    "lr2": ("formal3c", "8K 上下文；每批数据更新 8 次；学习率 2e-6（原 1e-6）；记录生成端概率（只监控）"),
    "lr2dyn": ("formal3d", "8K 上下文；每批数据更新 8 次＋DAPO 动态采样；学习率 2e-6（原 1e-6）；记录生成端概率（只监控）"),
    "frontier": ("formal3e", "8K 上下文；只训边界题（先离线测通过率，每题 8 次）；每题 n=32、每步 32 题；每批更新 8 次；"
                             "学习率和是否动态采样按 B、C 的结论选；记录生成端概率（只监控）"),
}


def _verdict(name):
    try:
        return json.loads((RUN / "audit" / name).read_text()).get("verdict")
    except Exception:  # noqa: BLE001
        return None


def prepare_frontier(board, fw, trainer, lvl):
    """第三轮 D 的准备（2026-11-17 定的流程，看结果前定）：
    1. 用当前最新 checkpoint 离线测 2,048 道训练题的通过率（每题 8 次，温度 1.0）；
    2. 选边界题（做对 1–5 次）+ 最多占 30% 的全错题 -> train_frontier.parquet；
    3. 学习率：C 显著提升则用 2e-6，否则 1e-6；动态采样：B 显著提升才保留；
    4. 把最新 checkpoint 的 data.pt（旧数据集的读取进度）改名，让续训从新数据集开头读。
    返回非 None 表示已暂停等 Claude。"""
    from . import build_frontier as BF
    from .framework_agent import VALGEN_DIR
    from .rescore import newest_file
    last = latest_saved(CKPT_DIR)
    lp = CKPT_DIR / f"global_step_{last}"
    if not BF.PROFILE.exists():
        BF.make_profile()
    exp = f"profile_s{last}"
    vdir = Path(VALGEN_DIR) / exp
    f = newest_file(vdir) if vdir.is_dir() else None
    if f is None:
        board.post(NAME, "decision", {"why": f"D：离线测训练题通过率（step {last}，2,048 题 × 8 次，温度 1.0）"})
        pp = plan("eval_profile", lvl, exp=exp, resume_path=lp)
        ok, err = fw.validate(pp)
        if not ok:
            return pause_for_claude(board, f"D：通过率测试配置校验失败：{err[-600:]}")
        r = trainer.train(pp, hang_min=90)
        f = newest_file(vdir) if vdir.is_dir() else None
        if r["status"] != "done" or f is None:
            return pause_for_claude(board, "D：通过率测试没有正常结束或没有生成结果文件", r)
    stats = BF.select(f)
    board.post(NAME, "frontier_stats", {k: v for k, v in stats.items() if k != "dist"} | {"dist": stats.get("dist")})
    if stats["n_train"] < 200:
        return pause_for_claude(board, f"D：边界题太少（{stats['n_train']} 道），等 Claude 判断", frontier=stats)
    prev = (board.get("prev_stages") or [{}])[-1].get("stage")
    c_v = _verdict(f"stage_{prev}_result.json") if prev in ("lr2", "lr2dyn") else None
    b_v = _verdict("stage_b_result.json")
    if c_v == "显著下降":
        # 2026-11-18 补充：C 让效果显著变差时，不在变差的模型上自动接着做 D，停下来由 Claude 决定起点
        return pause_for_claude(board, "C（学习率 2e-6）让效果显著变差：D 不自动开始，起点由 Claude 判断", c_verdict=c_v)
    extra = {"lr": 2e-6 if c_v == "显著提升" else 1e-6, "dyn": b_v == "显著提升"}
    dp = lp / "data.pt"
    if dp.exists():
        dp.rename(lp / "data.pt.pre_frontier")
    board.set(frontier_built=True, frontier_stats=stats, plan_extra=extra,
              settings=board.get("settings", "") + "；实际：学习率 %g（C：%s），动态采样 %s（B：%s），边界题 %d 道"
              % (extra["lr"], c_v, "保留" if extra["dyn"] else "不用", b_v, stats["n_train"]))
    board.post(NAME, "decision", {"why": "D：边界题 %d 道（边界 %d + 全错 %d），学习率 %g，动态采样 %s"
                                         % (stats["n_train"], stats["frontier"], stats["hard_used"], extra["lr"], extra["dyn"])})
    return None
MAX_RESUMES = 8


def init_round3(board):
    old = read_json(RUN / "state.json", {}) or {}
    stage_env = os.environ.get("R1Z_STAGE", "16k")
    if board.get("round") == 3:
        # 2026-11-17：第三轮内换阶段（A=upd8 → B=upd8dyn）。只有上一阶段已完成（stage_done）且指定了不同的阶段才重新初始化
        w = old.get("waiting") or {}
        if not (old.get("phase") == "waiting_claude" and w.get("stage_done") and stage_env != old.get("stage")):
            return
        shutil.copy2(RUN / "state.json", RUN / f"state_round3_{old.get('stage')}.json")
        prev = list(old.get("prev_stages") or []) + [{"stage": old.get("stage"), "start_step": old.get("start_step"),
                                                      "target_steps": old.get("target_steps"), "final_eval": old.get("final_eval")}]
    else:
        prev = []
    if old.get("round") == 2 and old.get("phase") not in ("done", "failed"):
        raise RuntimeError(f"第二轮还没结束（phase={old.get('phase')}）")
    if not (RUN / "state_round2.json").exists():
        shutil.copy2(RUN / "state.json", RUN / "state_round2.json")
    stage = stage_env
    extra = int(os.environ.get("R1Z_EXTRA_STEPS", "100"))
    best = old.get("best")
    if stage == "16k":
        if not best or not Path(best["path"]).is_dir():
            raise RuntimeError(f"找不到第二轮的最佳 checkpoint：{best}")
        s = dict(kind="stage16k", ckpt_dir=str(CKPT_DIR_16K), mfile=str(RUN / f"metrics_{EXP_16K}.jsonl"),
                 start_from=best["path"], target_steps=int(best["step"]) + extra, level=0, best=None,
                 settings="16K 上下文 + DAPO 动态采样；每 2 步存盘、每 4 步验证；评选集 400 条（16K 生成）；超长软惩罚（>12288 token 扣分）；只存模型权重")
    elif stage == "dyn8k":
        start = latest_saved(CKPT_DIR)
        s = dict(kind="formal2_dyn", ckpt_dir=str(CKPT_DIR), mfile=str(RUN / f"metrics_{EXP}.jsonl"), start_from=None,
                 target_steps=start + extra, level=old.get("level", 1), best=best,
                 settings="8K 上下文 + DAPO 动态采样；每 2 步存盘、每 4 步验证；评选集 400 条；超长软惩罚（>6144 token 扣分）；只存模型权重")
    elif stage in R3_STAGES:
        # 2026-11-17：第二轮在 step 76 提前结束（平台期）。A=upd8：只把每批更新次数提到 8；B=upd8dyn：再加动态采样。
        start = latest_saved(CKPT_DIR)
        s = dict(kind=R3_STAGES[stage][0], ckpt_dir=str(CKPT_DIR), mfile=str(RUN / f"metrics_{EXP}.jsonl"),
                 start_from=None, start_step=start, target_steps=start + extra, level=old.get("level", 1), best=best,
                 eval_seen=old.get("eval_seen", []),
                 settings=(R3_STAGES[stage][1]
                           + "；修正版判分 reward_v2；训练集去掉 6 道泄露题；每 2 步存盘、每 4 步验证；评选集 400 条（v2 判分）"))
    else:
        raise ValueError(f"R1Z_STAGE={stage}")
    board.state = {"round": 3, "stage": stage, "phase": "preflight", "env": old.get("env"), "data": old.get("data"),
                   "select_set": old.get("select_set"),
                   "round2_final_eval": old.get("round2_final_eval") if old.get("round") == 3 else old.get("final_eval"),
                   "r2_best_eval_v2": old.get("r2_best_eval_v2"), "r2_last_eval_v2": old.get("r2_last_eval_v2"),
                   "prev_stages": prev,
                   "round3_start": now(), "resumes": 0, "no_progress": 0,
                   "after": os.environ.get("R1Z_AFTER", "shutdown"), **s}
    board.set()
    board.post(NAME, "round3_init", {k: v for k, v in s.items() if k != "best"} | {"stage": stage})


def main():
    board = Board()
    init_round3(board)
    kind, ckpt, mfile = board.get("kind"), Path(board.get("ckpt_dir")), Path(board.get("mfile"))
    target, start_from = board.get("target_steps"), board.get("start_from")
    env, data, fw, trainer = EnvAgent(board), DataAgent(board), FrameworkAgent(board), TrainerAgent(board)
    ev = EvalAgent(board, ckpt_dir=ckpt, start_path=start_from, since_key="round3_start")
    an = AnalysisAgent(board, since_key="round3_start", mfile=mfile, out="REPORT_round3",
                       title=f"r1zero 第三轮（{board.get('stage')}）报告")
    phase = board.get("phase")
    if phase in ("done", "failed", "waiting_claude"):
        return 0
    board.post(NAME, "start", {"phase": phase, "kind": kind, "target_steps": target})
    try:
        if phase == "preflight":
            if kind == "stage16k":
                # 只留「最新 + 最佳」：16K 阶段的最新写在新目录，第二轮数据盘上的最新可以删了（最佳在系统盘）
                for k in ckpt_steps(CKPT_DIR):
                    shutil.rmtree(CKPT_DIR / f"global_step_{k}", ignore_errors=True)
                    board.post(NAME, "pruned_round2", {"step": k})
            reclaim = dir_gb(start_from) if kind == "stage16k" else (dir_gb(BEST_ROOT) if BEST_ROOT.exists() else 0.0)
            ok, msg = env.preflight(sys_reclaim_gb=reclaim)
            if not ok:
                return pause_for_claude(board, f"预检：{msg}")
            if not (RUN / "data" / "val_select.parquet").exists():
                ok, stats = data.build_select()
                if not ok:
                    return pause_for_claude(board, f"评选集：{stats}")
            lvl = board.get("level", 1)
            if not board.get("r2_best_eval_v2"):
                # 审计后补做（2026-11-16）：第二轮的最终评测按原判分、没有落盘生成，也没有 AIME 2026；
                # 这里用 v2 判分对第二轮最佳做一次完整评测（含 AIME 2026 留出集、去掉泄露题的口径），作为第三轮的对照起点。
                bj = BEST_ROOT / "best.json"
                bpath = json.loads(bj.read_text())["path"] if bj.exists() else None
                res = None
                if bpath and Path(bpath).is_dir():
                    board.post(NAME, "decision", {"why": f"第三轮前：v2 判分完整评测第二轮最佳 {bpath}"})
                    res = ev.final_eval(trainer, "r2_best_v2", bpath, lvl, kind="eval")
                if not res:
                    return pause_for_claude(board, f"第二轮最佳的完整评测失败：{bpath}")
                board.set(r2_best_eval_v2=res)
            if kind in ("formal3a", "formal3b") and not board.get("r2_last_eval_v2"):
                lp = str(CKPT_DIR / f"global_step_{latest_saved(CKPT_DIR)}")
                if lp != (board.get("r2_best_eval_v2") or {}).get("path"):
                    board.post(NAME, "decision", {"why": f"第三轮前：v2 判分完整评测第二轮最新 {lp}"})
                    res = ev.final_eval(trainer, "r2_last_v2", lp, lvl, kind="eval")
                    if not res:
                        return pause_for_claude(board, f"第二轮最新的完整评测失败：{lp}")
                    board.set(r2_last_eval_v2=res)
            if kind == "formal3e" and not board.get("frontier_built"):
                r = prepare_frontier(board, fw, trainer, lvl)
                if r is not None:
                    return r
            board.set(phase="train")
            phase = "train"

        if phase == "train":
            level, levels = board.get("level", 0), levels_for(kind)

            def tick():
                ev.tick(mfile)
                return an.check(mfile)

            while True:
                p = plan(kind, level, target, start_from=start_from, extra=board.get("plan_extra"))
                if latest_saved(ckpt) >= 0 and latest_saved(ckpt) in ev.seen:
                    p["val_before"] = False  # 起点那一步已经验证过，续训时不再重复验证
                ok, err = fw.validate(p)
                if not ok:
                    return pause_for_claude(board, f"配置校验失败：{err[-600:]}")
                start_ckpt = latest_saved(ckpt)
                board.post(NAME, "decision", {"why": f"{kind}：从 step {start_ckpt if start_ckpt >= 0 else start_from} 训到 {target}",
                                              "level": levels[level], "resumes": board.get("resumes")})
                ok_space, free_gb, need_gb = save_space(ckpt)
                if not ok_space:
                    return pause_for_claude(board, f"数据盘空间不够下一次存盘：可用 {free_gb:.1f} GB，一份 checkpoint {need_gb:.1f} GB（需另留 1.5 GB 余量）", level=level)
                r = trainer.train(p, hang_min=120 if kind == "stage16k" else 90, on_tick=tick)
                ev.wait()
                ev.tick(mfile)
                ev.wait()
                if r["status"] == "done":
                    if latest_saved(ckpt) >= target:
                        break
                    return pause_for_claude(board, f"训练进程正常退出，但最新 checkpoint 只到 step {latest_saved(ckpt)}", r)
                latest = latest_saved(ckpt)
                ok_latest, problems = ckpt_complete(ckpt / f"global_step_{latest}") if latest >= 0 else (False, ["没有 checkpoint"])
                why = (f"结果分析发现致命异常：{r['fatal']}" if r["status"] == "fatal"
                       else f"训练中断（{r['status']}/{r['cls']}，本次最后完成第 {r['last_step']} 步）")
                return pause_for_claude(board, why, r, level=level, latest=latest, latest_complete=ok_latest, latest_problems=problems)
            board.set(phase="final_eval")
            phase = "final_eval"

        if phase == "final_eval":
            level, results = board.get("level", 0), []
            ekind = "eval16k" if kind == "stage16k" else "eval"
            best, last = board.get("best"), latest_saved(ckpt)
            targets = ([("best", best["path"])] if best else []) + (
                [("last", str(ckpt / f"global_step_{last}"))] if not best or best["step"] != last else [])
            r2b = board.get("r2_best_eval_v2") or {}
            done_paths = {r2b.get("path")} | {(fe or {}).get("path") for ps in (board.get("prev_stages") or []) for fe in (ps.get("final_eval") or [])}
            targets = [(t, pth) for t, pth in targets if pth not in done_paths]  # 已经完整评测过的（沿用的最佳、上一阶段的结果）不重复评
            for tag, path in targets:
                board.post(NAME, "decision", {"why": f"完整验证集评估 {tag}（{ekind}）：{path}"})
                results.append(ev.final_eval(trainer, f"r3_{board.get('stage')}_{tag}", path, level, kind=ekind))
            board.set(final_eval=results)
            if board.get("after") == "pause":
                # 分阶段实验：这一阶段跑完不关机，停下来等 Claude 看结果、决定下一阶段（看门狗宽限内没人处理才关机）
                try:
                    an.report("stage_done", f"{board.get('stage')} 阶段完成（到 step {target}）", results)
                except Exception as e:  # noqa: BLE001
                    board.post(NAME, "report_error", {"error": repr(e)})
                return pause_for_claude(board, f"{board.get('stage')} 阶段完成（到 step {target}），等 Claude 判断下一步", stage_done=True)
            return finish(board, an, "completed", f"第三轮完成（到 step {target}），最终评估 {sum(1 for x in results if x)}/{len(targets)} 个成功", results)
    except Exception as e:  # noqa: BLE001
        board.post(NAME, "exception", {"error": repr(e), "trace": traceback.format_exc()[-2000:]})
        return pause_for_claude(board, f"loop3 异常：{e!r}", tail=traceback.format_exc()[-1500:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
