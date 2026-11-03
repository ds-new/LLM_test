"""多智能体预训练 loop 入口。

  python -m decoder_from_scratch.multiagent.loop run       # 由 run_loop.sh 调用（可断点恢复）
  python -m decoder_from_scratch.multiagent.loop status    # 查看进度
  python -m decoder_from_scratch.multiagent.loop stop      # 当前段结束后保存并停止
  python -m decoder_from_scratch.multiagent.loop restart   # 保存后用最新代码重启

退出码：0 完成或停止；75 请求重启；2 致命错误（同一阶段连续失败 5 次）；3 已有实例在运行；1 可重试错误。
"""

import argparse
import json
import os
import subprocess
import sys
import traceback

from .common import (RUN_DIR, Blackboard, Bus, RestartRequested, StopRequested, load_config, now, read_json, stamp,
                     write_json)

LOCK = RUN_DIR / 'loop.pid'


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def other_training_processes():
    """发现其他占用 GPU 的旧训练进程时给出警告（避免两个训练抢同一块 GPU）。"""
    try:
        out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    mine = {os.getpid(), os.getppid()}
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in mine or 'multiagent' in cmd:
            continue
        if 'decoder_from_scratch' in cmd and 'python' in cmd:
            hits.append(f'{pid} {cmd[:160]}')
    return hits


def run():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and pid_alive(LOCK.read_text().strip()) and LOCK.read_text().strip() != str(os.getpid()):
        print(f'another loop instance is running (pid {LOCK.read_text().strip()})', flush=True)
        return 3
    LOCK.write_text(str(os.getpid()))
    config = load_config()
    bus, board = Bus(), Blackboard()
    from .decision_agent import DecisionAgent
    others = other_training_processes()
    if others:
        bus.post('decision', 'all', 'warning', {'message': '检测到其他 decoder 训练进程，可能争用 GPU', 'processes': others})
    bus.post('decision', 'all', 'start', {'pid': os.getpid(), 'python': sys.executable, 'run_dir': str(RUN_DIR)})
    agent = None
    try:
        agent = DecisionAgent(config, bus, board)
        agent.run()
        return 0
    except StopRequested as exc:
        bus.post('decision', 'all', 'stopped', {'reason': str(exc)})
        return 0
    except RestartRequested as exc:
        try:
            (RUN_DIR / 'RESTART').unlink()
        except OSError:
            os.replace(RUN_DIR / 'RESTART', RUN_DIR / f'RESTART.consumed.{int(now())}')
        bus.post('decision', 'all', 'restarting', {'reason': str(exc)})
        return 75
    except Exception as exc:
        tb = traceback.format_exc()
        failures = board.get('phase_failures', 0) + 1
        board.update(phase_failures=failures, last_error={'time': stamp(), 'phase': board.get('phase'),
                                                          'error': f'{type(exc).__name__}: {exc}', 'traceback': tb[-4000:]})
        bus.post('decision', 'all', 'error', {'phase': board.get('phase'), 'failures': failures,
                                              'error': f'{type(exc).__name__}: {exc}'})
        print(tb, flush=True)
        if agent is not None:
            agent.write_report()
        if failures >= 5:
            write_json(RUN_DIR / 'FAILED.json', board.get('last_error'))
            return 2
        return 1
    finally:
        try:
            if LOCK.read_text().strip() == str(os.getpid()):
                LOCK.unlink()
        except OSError:
            pass


def status():
    board = read_json(RUN_DIR / 'state.json', {}) or {}
    seg = board.get('last_segment') or {}
    plan = board.get('plan') or {}
    hist = board.get('eval_history') or []
    lock = LOCK.read_text().strip() if LOCK.exists() else None
    out = {
        'running': bool(lock and pid_alive(lock)), 'pid': lock, 'phase': board.get('phase'),
        'started': board.get('started_text'), 'deadline': board.get('deadline_text'),
        'remaining_hours': round((board.get('deadline', now()) - now()) / 3600, 2) if board else None,
        'active_domains': board.get('active_domains'),
        'plan': {k: plan.get(k) for k in ('name', 'params', 'precision', 'total_steps', 'peak_lr')} if plan else None,
        'step': seg.get('step'), 'tokens_seen': seg.get('tokens_seen'), 'tok_s': seg.get('segment_tok_s'),
        'train_loss': seg.get('train_loss'), 'last_eval': hist[-1] if hist else None,
        'best_val': board.get('best_val'), 'last_error': (board.get('last_error') or {}).get('error'),
        'stop_file': (RUN_DIR / 'STOP').exists(), 'failed': (RUN_DIR / 'FAILED.json').exists(),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    events = Bus().tail(12)
    for e in events:
        c = e['content']
        text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        print(f'{e["time"]} {e["from"]}->{e["to"]} [{e["type"]}] {text[:200]}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('run', 'status', 'stop', 'restart'), nargs='?', default='status')
    args = parser.parse_args()
    if args.command == 'run':
        sys.exit(run())
    if args.command == 'status':
        sys.exit(status())
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    name = 'STOP' if args.command == 'stop' else 'RESTART'
    (RUN_DIR / name).write_text(stamp())
    print(f'{name} 已放置：{RUN_DIR / name}；当前训练段在下一步检查时保存并{"停止" if name == "STOP" else "重启"}。')


if __name__ == '__main__':
    main()
