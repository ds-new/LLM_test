"""定时检查用（第三轮）：状态、进程、磁盘、告警、各步指标。用法：python ma/check_r3.py [最近几步]"""
import json, subprocess, sys, statistics as S
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
d = json.load(open('run/state.json'))
print('round', d.get('round'), 'stage', d.get('stage'), 'phase', d.get('phase'), 'target', d.get('target_steps'), 'best', (d.get('best') or {}).get('step'), (d.get('best') or {}).get('score'))
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print('procs:', sh("pgrep -af 'ma.loop3|watchdog3|main_ppo' | cut -c1-40 | tr '\n' ';'"))
print('latest ckpt:', sh('cat run/ckpt/r1zero_qwen3_8b_dapo/latest_checkpointed_iteration.txt'), '| dirs:', sh('ls run/ckpt/r1zero_qwen3_8b_dapo | tr "\n" " "'))
print('disk MiB free: sys', sh("df -B1M / | tail -1 | awk '{print $4}'"), 'data', sh("df -B1M /root/autodl-tmp | tail -1 | awk '{print $4}'"), '| waiting file:', sh('ls run/WAITING_CLAUDE.json 2>/dev/null') or 'none')
rows = [json.loads(l) for l in open('run/metrics_r1zero_qwen3_8b_dapo.jsonl')]
rows = [r for r in rows if 'timing_s/step' in r]
r2 = [r for r in rows if '2026-11-16 21:42' <= r.get('t', '') < '2026-11-17 09:05']; r3 = [r for r in rows if r.get('t', '') >= '2026-11-17 09:05']
for r in r3[-n:]:
    print(r['step'], r['t'][11:16], 'step_s', round(r['timing_s/step']), 'gen', round(r.get('timing_s/gen', 0)), 'len', round(r.get('response_length/mean', 0)), 'trunc', round(r.get('response_length/clip_ratio', 0), 3),
          'score', round(r['critic/score/mean'], 3), 'ent', round(r.get('actor/entropy', 0), 3), 'gn', round(r.get('actor/grad_norm', 0), 3), 'clipfrac', round(r.get('actor/pg_clipfrac', 0), 4),
          *(['rc_kl', round(r['rollout_corr/kl'], 5), 'rc_chi2', round(r.get('rollout_corr/chi2_token', 0), 4), 'lr', r.get('actor/lr')] if 'rollout_corr/kl' in r else []))
m = lambda L, k: round(S.mean(x[k] for x in L), 4) if L else None
print('R2 last10 score', m(r2[-10:], 'critic/score/mean'), '| R3 post-warmup (n=%d) score' % len(r3[3:]), m(r3[3:], 'critic/score/mean'), 'ent', m(r3[3:], 'actor/entropy'), 'len', m(r3[3:], 'response_length/mean'))
for l in open('run/select_history.jsonl'):
    x = json.loads(l)
    if x['t'] >= '2026-11-17 09:05': print('select', x['step'], x['t'][11:16], x['score'], x.get('v2'))
ev = [json.loads(l) for l in open('run/events.jsonl')][-400:]
for x in ev:
    if x.get('t', '') >= '2026-11-17 09:05' and (x.get('kind') in ('warning', 'ckpt_incomplete', 'best_skipped_incomplete', 'rescore_error', 'best_copy_failed', 'new_best', 'waiting_claude', 'exception') or (x.get('kind') == 'result' and x.get('status') != 'done')):
        print('EVENT', json.dumps(x, ensure_ascii=False)[:260])
