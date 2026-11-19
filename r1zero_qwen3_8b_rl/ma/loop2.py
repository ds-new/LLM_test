"""第二轮多智能体 loop（决策智能体主导）：
  预检（环境、数据）→ 从最近 checkpoint 续训到 150 步（每 2 步验证+存盘，保留最新与最佳）
  → 用完整验证集评估最佳与最终模型 → 结果分析报告 → 关机。

分工：
  env（环境）      依赖检查与修复、GPU 空闲、磁盘余量
  data（数据）     训练集/验证集已在第一轮建好；这里抽评选集 val_select
  framework（框架）verl/Megatron/SGLang 配置、显存档位、checkpoint 与奖励策略、Hydra 预校验
  trainer（训练）  启动/监控训练进程，判断崩溃/卡死类别
  eval（测试评估） 选最佳 checkpoint、清理旧 checkpoint、最终完整评估
  analysis（分析） 每步健康检查与告警、最终报告
  decision（决策） 本文件：编排上述智能体，决定续训/降档/停止，完成或无法恢复时关机

自动恢复：显存不足 → 降一档（最多到最后一档）；崩溃/卡死（NCCL/磁盘/其他）→ 清理后从最新 checkpoint 续训，
最多 8 次，且连续两次没有新 checkpoint（没有进展）就判定无法恢复。
"""
import shutil
import sys
import time
import traceback

from .analysis_agent import AnalysisAgent
from .common import RUN, Board, now, read_json, shutdown
from .data_agent import DataAgent
from .env_agent import EnvAgent
from .eval_agent import CKPT_DIR, EvalAgent, ckpt_complete, latest_saved, save_space
from .framework_agent import LEVELS, EXP, FrameworkAgent, plan
from .trainer_agent import TrainerAgent

NAME = "decision"
FORMAL_STEPS = 150
MAX_RESUMES = 8
MFILE = RUN / f"metrics_{EXP}.jsonl"


def init_round2(board):
    if board.get("round") == 2:
        return
    old = read_json(RUN / "state.json", {}) or {}
    if not (RUN / "state_round1.json").exists():
        shutil.copy2(RUN / "state.json", RUN / "state_round1.json")
    board.state = {"round": 2, "phase": "preflight", "env": old.get("env"), "data": old.get("data"),
                   "level": old.get("level", 1), "round2_start": now(), "resumes": 0, "no_progress": 0}
    board.set()
    board.post(NAME, "round2_init", {"from_step": latest_saved(), "level": board.get("level")})


def finish(board, analysis, status, reason, final_eval=None):
    board.set(phase="done" if status == "completed" else "failed", final_status=status, final_reason=reason)
    board.post(NAME, "finish", {"status": status, "reason": reason})
    try:
        p = analysis.report(status, reason, final_eval)
        board.post(NAME, "report", {"path": str(p)})
    except Exception as e:  # noqa: BLE001
        board.post(NAME, "report_error", {"error": repr(e), "trace": traceback.format_exc()[-1200:]})
    if status == "completed":
        shutdown(board, reason, delay=60)
    else:
        # 无法恢复：交给 watchdog —— 10 分钟内没人开始修复（touch run/REPAIRING）就关机
        board.post(NAME, "failure_grace", {"reason": reason, "shutdown_in_min": 10, "hold_file": str(RUN / "REPAIRING")})
    return 0 if status == "completed" else 1


def pause_for_claude(board, reason, r=None, **info):
    """2026-11-16 起：异常不再按规则自动处理（降档、回退、续训都由 Claude 判断）。
    停下来写 run/WAITING_CLAUDE.json 并退出；看门狗 20 分钟内没看到 Claude 开始处理（touch run/REPAIRING）就关机，不删任何文件。"""
    from .common import write_json
    rec = {"t": now(), "reason": reason, **info}
    if r:
        rec.update(status=r.get("status"), cls=r.get("cls"), last_step=r.get("last_step"), log=r.get("log"),
                   tail=(r.get("tail") or "")[-1500:])
    write_json(RUN / "WAITING_CLAUDE.json", rec)
    board.set(phase="waiting_claude", waiting={k: v for k, v in rec.items() if k != "tail"})
    board.post(NAME, "waiting_claude", {k: v for k, v in rec.items() if k != "tail"})
    return 1


def main():
    board = Board()
    init_round2(board)
    env, data, fw, trainer = EnvAgent(board), DataAgent(board), FrameworkAgent(board), TrainerAgent(board)
    ev, an = EvalAgent(board), AnalysisAgent(board)
    phase = board.get("phase")
    if phase in ("done", "failed", "waiting_claude"):
        return 0  # 失败或等待中的状态只由 Claude 处理后改回 train 再启动
    board.post(NAME, "start", {"phase": phase})
    try:
        if phase == "preflight":
            ok, msg = env.preflight()
            if not ok:
                return pause_for_claude(board, f"预检：{msg}")
            ok, stats = data.build_select()
            if not ok:
                return pause_for_claude(board, f"评选集：{stats}")
            phase = "train"
            board.set(phase=phase)

        if phase == "train":
            level = board.get("level", 1)

            def tick():
                ev.tick(MFILE)
                return an.check(MFILE)

            while True:
                p = plan("formal2", level, FORMAL_STEPS)
                if latest_saved() in ev.seen:
                    p["val_before"] = False  # 起点那一步已经验证过（评选分数已记录），续训时不再重复验证（省约 13 分钟）
                ok, err = fw.validate(p)
                if not ok:
                    return pause_for_claude(board, f"配置校验失败：{err[-600:]}")
                start_ckpt = latest_saved()
                board.post(NAME, "decision", {"why": f"从 step {start_ckpt} 续训", "level": level, "resumes": board.get("resumes")})
                ok_space, free_gb, need_gb = save_space(CKPT_DIR)
                if not ok_space:  # 启动前核算：放不下"旧的 + 新的"两份就不启动，免得训完两步才在存盘时崩
                    return pause_for_claude(board, f"数据盘空间不够下一次存盘：可用 {free_gb:.1f} GB，一份 checkpoint {need_gb:.1f} GB（需另留 1.5 GB 余量）", level=level)
                r = trainer.train(p, hang_min=90, on_tick=tick)
                ev.wait()
                ev.tick(MFILE)
                ev.wait()
                if r["status"] == "done":
                    if latest_saved() >= FORMAL_STEPS:
                        break
                    return pause_for_claude(board, f"训练进程正常退出，但最新 checkpoint 只到 step {latest_saved()}", r)
                latest = latest_saved()
                ok_latest, problems = ckpt_complete(CKPT_DIR / f"global_step_{latest}") if latest >= 0 else (False, ["没有 checkpoint"])
                why = (f"结果分析发现致命异常：{r['fatal']}" if r["status"] == "fatal"
                       else f"训练中断（{r['status']}/{r['cls']}，本次最后完成第 {r['last_step']} 步）")
                return pause_for_claude(board, why, r, level=level, latest=latest, latest_complete=ok_latest, latest_problems=problems)
            phase = "final_eval"
            board.set(phase=phase)

        if phase == "final_eval":
            level, results = board.get("level", 1), []
            best, last = board.get("best"), latest_saved()
            targets = []
            if best:
                targets.append(("best", best["path"]))
            if not best or best["step"] != last:
                targets.append(("last", str(CKPT_DIR / f"global_step_{last}")))
            for tag, path in targets:
                board.post(NAME, "decision", {"why": f"完整验证集评估 {tag}：{path}"})
                results.append(ev.final_eval(trainer, tag, path, level))
            board.set(final_eval=results)
            return finish(board, an, "completed", f"训练完成 {FORMAL_STEPS} 步，最终评估 {sum(1 for x in results if x)}/{len(targets)} 个成功", results)
    except Exception as e:  # noqa: BLE001
        board.post(NAME, "exception", {"error": repr(e), "trace": traceback.format_exc()[-2000:]})
        return pause_for_claude(board, f"loop 异常：{e!r}", tail=traceback.format_exc()[-1500:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
