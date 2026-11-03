"""多智能体 GRPO loop 入口（在工程根目录 qwen3vl_4b_geo3k_grpo/ 下运行）。

  python -m multiagent.loop run        # 由 run_loop.sh 调用（可断点恢复）
  python -m multiagent.loop status     # 查看进度
  python -m multiagent.loop stop       # 当前训练在下一次轮询时结束并保存状态后停止
  python -m multiagent.loop restart    # 保存后用最新代码重启
  MA_DRY_RUN=1 python -m multiagent.loop run   # 干跑：合成数据 + 模拟 verl，几分钟走完全流程

退出码：0 完成或停止；75 请求重启；2 致命错误（同一阶段连续失败 5 次）；3 已有实例在运行；1 可重试错误。
"""

import argparse
import json
import os
import sys
import traceback

from .common import (RUN_DIR, Blackboard, Bus, RestartRequested, StopRequested, load_config, now, read_json, stamp,
                     write_json)

LOCK = RUN_DIR / "loop.pid"


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def run():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        other = LOCK.read_text().strip()
        if other != str(os.getpid()) and pid_alive(other):
            print(f"another loop instance is running (pid {other})", flush=True)
            return 3
    LOCK.write_text(str(os.getpid()))
    config = load_config()
    bus, board = Bus(), Blackboard()
    from .decision_agent import DecisionAgent
    bus.post("decision", "all", "start", {"pid": os.getpid(), "python": sys.executable, "run_dir": str(RUN_DIR)})
    agent = None
    try:
        agent = DecisionAgent(config, bus, board)
        agent.run()
        return 0
    except StopRequested as exc:
        bus.post("decision", "all", "stopped", {"reason": str(exc)})
        if agent:
            agent.write_report()
        return 0
    except RestartRequested as exc:
        try:
            (RUN_DIR / "RESTART").unlink()
        except OSError:
            os.replace(RUN_DIR / "RESTART", RUN_DIR / f"RESTART.consumed.{int(now())}")
        bus.post("decision", "all", "restarting", {"reason": str(exc)})
        return 75
    except Exception as exc:
        tb = traceback.format_exc()
        failures = board.get("phase_failures", 0) + 1
        board.update(phase_failures=failures, last_error={"time": stamp(), "phase": board.get("phase"),
                                                          "error": f"{type(exc).__name__}: {exc}", "traceback": tb[-4000:]})
        bus.post("decision", "all", "error", {"phase": board.get("phase"), "failures": failures,
                                              "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
        print(tb, flush=True)
        if agent is not None:
            agent.write_report()
        if failures >= 5:
            write_json(RUN_DIR / "FAILED.json", board.get("last_error"))
            return 2
        return 1
    finally:
        try:
            if LOCK.read_text().strip() == str(os.getpid()):
                LOCK.unlink()
        except OSError:
            pass


def status():
    st = read_json(RUN_DIR / "state.json", {}) or {}
    lock = LOCK.read_text().strip() if LOCK.exists() else None
    cur = st.get("current") or {}
    out = {
        "running": bool(lock and pid_alive(lock)), "pid": lock, "run_dir": str(RUN_DIR), "phase": st.get("phase"),
        "deadline": st.get("deadline_text"),
        "remaining_hours": round((st["deadline"] - now()) / 3600, 2) if st.get("deadline") else None,
        "baseline": st.get("baseline"),
        "current_round": {"round": cur["plan"]["round"], "preset": cur["plan"]["preset"], "best": cur.get("best"),
                          "val_history": cur.get("val_history", [])[-8:]} if cur else None,
        "rounds": [{k: r.get(k) for k in ("round", "preset", "status", "val_acc", "val_gain")} for r in st.get("rounds", [])],
        "final": st.get("final"), "last_error": (st.get("last_error") or {}).get("error"),
        "stop_file": (RUN_DIR / "STOP").exists(), "failed": (RUN_DIR / "FAILED.json").exists(),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    for e in Bus().tail(12):
        c = e["content"]
        text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        print(f"{e['time']} {e['from']}->{e['to']} [{e['type']}] {text[:220]}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "status", "stop", "restart"), nargs="?", default="status")
    args = parser.parse_args()
    if args.command == "run":
        sys.exit(run())
    if args.command == "status":
        sys.exit(status())
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    name = "STOP" if args.command == "stop" else "RESTART"
    (RUN_DIR / name).write_text(stamp())
    print(f"{name} 已放置：{RUN_DIR / name}")


if __name__ == "__main__":
    main()
