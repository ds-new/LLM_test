"""R1-Zero 多智能体 loop（决策智能体主导）：环境 → 数据 → 冒烟测试 → 正式训练 → 报告 → 关机。

用户要求：出现异常（无法自动恢复）或任务完成后 10 分钟内停止并关机。
自动恢复只做三类：显存不足（逐级降档，最多 2 次）、缺 flash_attn（装预编译包，1 次）、正式训练中断（从检查点续训，1 次）。
"""
import json
import sys
import time
import traceback

from .common import RUN, Board, now, read_json, shutdown
from .data_agent import DataAgent
from .env_agent import EnvAgent
from .trainer_agent import LEVELS, TrainerAgent, plan

NAME = "decision"
FORMAL_STEPS = 150
FORMAL_WALL_HOURS = 26


def report(board, final_status):
    st = board.state
    lines = [f"# Qwen3-8B-Base R1-Zero（verl 0.6.1 + Megatron + SGLang）运行报告", "",
             f"- 结束时间：{now()}，状态：**{final_status}**", ""]
    lines += ["## 环境", "", "```", json.dumps(st.get("env", {}), ensure_ascii=False, indent=1)[:3000], "```", ""]
    lines += ["## 数据", "", "```", json.dumps(st.get("data", {}), ensure_ascii=False, indent=1), "```", ""]
    for exp in ("smoke", "r1zero_qwen3_8b_dapo"):
        mf = RUN / f"metrics_{exp}.jsonl"
        if not mf.exists():
            continue
        rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
        lines += [f"## {exp}：每步指标（节选）", "", "| 步 | 奖励均值 | 平均长度 | 截断率 | 熵 | 每步秒 |", "| --- | --- | --- | --- | --- | --- |"]
        pick = rows if len(rows) <= 30 else rows[::max(1, len(rows) // 30)] + rows[-1:]
        for r in pick:
            lines.append(f"| {r.get('step')} | {r.get('critic/score/mean', float('nan')):.3f} | {r.get('response_length/mean', float('nan')):.0f} | "
                         f"{r.get('response_length/clip_ratio', float('nan')):.3f} | {r.get('actor/entropy', float('nan')):.3f} | {r.get('timing_s/step', float('nan')):.0f} |")
        lines.append("")
    vals = [json.loads(l) for l in (RUN / "events.jsonl").read_text().splitlines() if '"kind": "val"' in l]
    if vals:
        lines += ["## 验证集（acc/mean@1；AIME/AMC 每题复制 16 份 = avg@16）", "", "| 指标 | 值 | 时间 |", "| --- | --- | --- |"]
        lines += [f"| {v['metric']} | {v['value']:.4f} | {v['t']} |" for v in vals]
        lines.append("")
    (RUN / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finish(board, status, reason):
    board.set(phase="done" if status == "completed" else "failed", final_status=status, final_reason=reason)
    board.post(NAME, "finish", {"status": status, "reason": reason})
    try:
        report(board, status)
    except Exception as e:  # noqa: BLE001
        board.post(NAME, "report_error", {"error": repr(e)})
    if status == "completed":
        shutdown(board, reason, delay=60)
    else:
        # 报错停止：不立即关机，交给 watchdog —— 10 分钟内没人开始修复（run/REPAIRING）才关机
        board.post(NAME, "failure_grace", {"reason": reason, "shutdown_in_min": 10, "hold_file": str(RUN / "REPAIRING")})
    return 0 if status == "completed" else 1


def main():
    board = Board()
    env, data, trainer = EnvAgent(board), DataAgent(board), TrainerAgent(board)
    phase = board.get("phase") or "env"
    if phase in ("done", "failed"):
        return 0
    board.post(NAME, "start", {"phase": phase})
    try:
        if phase == "env":
            ok, msg = env.ensure()
            board.post(NAME, "env_result", {"ok": ok, "msg": msg})
            if not ok:
                return finish(board, "failed", f"环境：{msg}")
            phase = "data"
            board.set(phase=phase)
        if phase == "data":
            ok, stats = data.build()
            if not ok:
                return finish(board, "failed", f"数据：{stats}")
            phase = "smoke"
            board.set(phase=phase)
        if phase == "smoke":
            level, fa_tried = board.get("level", 0), False
            while True:
                p = plan("smoke", level)
                ok, err = trainer.validate(p)
                if not ok:
                    return finish(board, "failed", f"配置校验失败：{err[-600:]}")
                r = trainer.train(p, hang_min=30)
                if r["status"] == "done" and r["last_step"] >= p["steps"]:
                    break
                if r["cls"] == "oom" and level + 1 < len(LEVELS):
                    level += 1
                    board.post(NAME, "decision", {"why": f"冒烟测试显存不足，降到档位 {level}", "level": LEVELS[level]})
                    continue
                if r["cls"] == "flash_attn" and not fa_tried:
                    fa_tried = True
                    if env.try_flash_attn():
                        continue
                return finish(board, "failed", f"冒烟测试失败（{r['status']}/{r['cls']}）：{r['tail'][-500:]}")
            rows = [json.loads(l) for l in open(r["metrics"]) if l.strip()]
            board.set(level=level, smoke={"step_seconds": [x.get("timing_s/step") for x in rows], "level": level})
            board.post(NAME, "decision", {"why": "冒烟测试通过，进入正式训练", "level": level,
                                          "step_seconds": [x.get("timing_s/step") for x in rows]})
            phase = "formal"
            board.set(phase=phase)
        if phase == "formal":
            level, resumed = board.get("level", 0), board.get("formal_resumes", 0)
            deadline = board.get("formal_deadline") or (time.time() + FORMAL_WALL_HOURS * 3600)
            board.set(formal_deadline=deadline)
            while True:
                p = plan("formal", level, FORMAL_STEPS)
                if board.get("baseline_done"):  # 基线验证已有结果：续训时不再重复训练前验证（省约 25 分钟）
                    p["val_before"] = False
                ok, err = trainer.validate(p)
                if not ok:
                    return finish(board, "failed", f"正式训练配置校验失败：{err[-600:]}")
                left = (deadline - time.time()) / 3600
                if left <= 0.2:
                    return finish(board, "completed", "达到正式训练时长上限（检查点与验证结果已保存）")
                r = trainer.train(p, hang_min=45, wall_hours=left)
                if r.get("last_step", 0) >= 1:
                    board.set(baseline_done=True)
                if r["status"] == "done":
                    return finish(board, "completed", f"正式训练完成 {r['last_step']} 步")
                if r["status"] == "deadline":
                    return finish(board, "completed", f"达到 {FORMAL_WALL_HOURS} 小时上限，停在第 {r['last_step']} 步")
                if r["cls"] == "oom" and level + 1 < len(LEVELS):
                    level += 1
                    board.set(level=level)
                    board.post(NAME, "decision", {"why": f"正式训练显存不足，降到档位 {level} 并从检查点续训"})
                    continue
                if resumed < 3 and r["status"] in ("crash", "hang") and r["cls"] in ("nccl", "other"):
                    resumed += 1
                    board.set(formal_resumes=resumed)
                    board.post(NAME, "decision", {"why": f"训练中断（{r['status']}/{r['cls']}），从最近检查点续训（第 {resumed}/3 次）"})
                    continue
                return finish(board, "failed", f"正式训练失败（{r['status']}/{r['cls']}，第 {r['last_step']} 步）：{r['tail'][-500:]}")
    except Exception as e:  # noqa: BLE001
        board.post(NAME, "exception", {"error": repr(e), "trace": traceback.format_exc()[-2000:]})
        return finish(board, "failed", f"loop 异常：{e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
