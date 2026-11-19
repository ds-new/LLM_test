"""A800 多智能体预训练 loop 入口（由 run_loop.sh 守护运行）。

  python -m decoder_from_scratch.a800.loop run       # 守护脚本调用；可断点恢复
  python -m decoder_from_scratch.a800.loop status    # 查看进度与最近的智能体消息
  python -m decoder_from_scratch.a800.loop stop      # 当前步完成后保存并停止（不关机）
  python -m decoder_from_scratch.a800.loop restart   # 保存后用最新代码重启

退出码：0 训练完成；4 用户停止；75 请求重启；2 致命错误（同一阶段连续失败 5 次、FatalError，或硬件/磁盘类错误重启后仍出现）；3 已有实例在运行；1 可重试错误。
"""

import argparse
import json
import os
import sys
import traceback

from .common import (RUN_DIR, Blackboard, Bus, FatalError, RestartRequested, StopRequested, load_config, now, read_json, stamp,
                     write_json)

LOCK = RUN_DIR / 'loop.pid'
# 硬件 / 驱动 / 磁盘类错误：重启一次后仍出现，就判定无法自动恢复，直接关机，不再空烧机时重试 20 次
FATAL_PATTERNS = ('No space left on device', 'no CUDA-capable device', 'CUDA driver', 'Found no NVIDIA driver',
                  'uncorrectable ECC', 'NVML', 'an illegal memory access', 'device-side assert', 'Input/output error')


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def run():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and LOCK.read_text().strip() != str(os.getpid()) and pid_alive(LOCK.read_text().strip()):
        print(f'another loop instance is running (pid {LOCK.read_text().strip()})', flush=True)
        return 3
    LOCK.write_text(str(os.getpid()))
    config = load_config()
    bus, board = Bus(), Blackboard()
    if config.get('mode') == 'cpt':  # 路线 A：继续预训练 + 退火
        from .cpt import CPTDecisionAgent as DecisionAgent
    else:
        from .decision_agent import DecisionAgent
    bus.post('decision', 'all', 'start', {'pid': os.getpid(), 'python': sys.executable, 'run_dir': str(RUN_DIR)})
    agent = None
    try:
        agent = DecisionAgent(config, bus, board)
        agent.run()
        return 0
    except StopRequested as exc:
        bus.post('decision', 'all', 'stopped', {'reason': str(exc)})
        return 4
    except RestartRequested as exc:
        (RUN_DIR / 'RESTART').unlink(missing_ok=True)
        bus.post('decision', 'all', 'restarting', {'reason': str(exc)})
        return 75
    except FatalError as exc:
        err = {'time': stamp(), 'phase': board.get('phase'), 'error': f'FatalError: {exc}', 'traceback': traceback.format_exc()[-5000:]}
        board.update(last_error=err)
        bus.post('decision', 'all', 'fatal', {'phase': board.get('phase'), 'error': str(exc)[:500]})
        write_json(RUN_DIR / 'FAILED.json', err)
        if agent is not None:
            agent.write_report()
        return 2
    except Exception as exc:
        tb = traceback.format_exc()
        failures = board.get('phase_failures', 0) + 1
        board.update(phase_failures=failures, last_error={'time': stamp(), 'phase': board.get('phase'),
                                                          'error': f'{type(exc).__name__}: {exc}', 'traceback': tb[-5000:]})
        bus.post('decision', 'all', 'error', {'phase': board.get('phase'), 'failures': failures,
                                              'error': f'{type(exc).__name__}: {str(exc)[:500]}'})
        print(tb, flush=True)
        if agent is not None:
            agent.write_report()
        hint = next((p for p in FATAL_PATTERNS if p in tb), None)
        if hint and failures >= 2:
            write_json(RUN_DIR / 'FAILED.json', {**board.get('last_error'), 'fatal_pattern': hint})
            return 2
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
    hb = (RUN_DIR / 'heartbeat').read_text().split() if (RUN_DIR / 'heartbeat').exists() else None
    out = {
        'running': bool(lock and pid_alive(lock)), 'phase': board.get('phase'), 'started': board.get('started_text'),
        'heartbeat_age_s': round(now() - float(hb[0])) if hb else None, 'phase_hours': board.get('phase_hours'),
        'plan': {k: plan.get(k) for k in ('name', 'params', 'total_steps', 'planned_tokens', 'micro_batch', 'accum')} if plan else None,
        'step': seg.get('step'), 'tokens_B': round(seg.get('tokens_seen', 0) / 1e9, 3) if seg else None,
        'tok_s': seg.get('segment_tok_s'), 'train_loss': seg.get('train_loss'),
        'last_eval': {k: hist[-1][k] for k in ('step', 'val_loss', 'cloze_acc', 'sample_rep4')} if hist else None,
        'best_val': board.get('best_val'), 'last_error': (board.get('last_error') or {}).get('error'),
        'stop_file': (RUN_DIR / 'STOP').exists(), 'failed': (RUN_DIR / 'FAILED.json').exists(),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    for e in Bus().tail(15):
        c = e['content']
        text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        print(f'{e["time"]} {e["from"]}->{e["to"]} [{e["type"]}] {text[:240]}')
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
    print(f'已放置 {RUN_DIR / name}：训练会在下一步检查时保存并{"停止（不关机）" if name == "STOP" else "用最新代码重启"}。')


if __name__ == '__main__':
    main()
