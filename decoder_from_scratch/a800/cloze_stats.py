"""练习 14：选择题（cloze）准确率的不确定性 —— 置信区间与配对检验。

cloze 只有几十道题，一题就是几个百分点。这个脚本只读已有的评估记录（不需要 GPU）：
  * RUN_DIR/eval_history.jsonl：每次评估的整体准确率；
  * RUN_DIR/samples/step_*.json：每次评估逐题的预测与答案（用来做配对比较）。
输出：
  1. 每次评估的 95% 置信区间：Wilson 区间（解析）与 bootstrap 区间（对题目重抽样 10000 次）；
  2. 相邻两次评估之间"翻转"的题数（对→错、错→对），以及精确 McNemar 检验的 p 值
     —— 同一批题在两个 checkpoint 上的对比要用配对检验，不能把两个区间是否重叠当判据；
  3. 指定几对 checkpoint（最早 vs 最新、中点 vs 最新）的配对检验；
  4. 题目稳定性：一直对 / 一直错 / 来回翻的题各有多少；
  5. 图：准确率曲线 + Wilson 置信带（cloze_ci.png）。
结果写入 RUN_DIR/stats/。用法：python -m decoder_from_scratch.a800.cloze_stats [--min-step 3000]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .common import RUN_DIR, stamp, write_json, write_text

Z = 1.959964


def wilson(k, n, z=Z):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap(bits, reps=10000, seed=0):
    bits = np.asarray(bits, float)
    rng = np.random.default_rng(seed)
    means = bits[rng.integers(0, len(bits), (reps, len(bits)))].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcnemar_exact(a, b):
    """a、b：同一批题在两个模型上的对错（0/1）。返回 (只有 a 对的题数, 只有 b 对的题数, 双侧精确 p 值)。"""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    m = n10 + n01
    if m == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return n10, n01, min(1.0, p)


def load(run_dir):
    hist = [json.loads(l) for l in (run_dir / 'eval_history.jsonl').open(encoding='utf-8')]
    evals = []
    for h in hist:
        f = run_dir / 'samples' / f'step_{h["step"]:07d}.json'
        if not f.exists():
            continue
        cl = json.loads(f.read_text(encoding='utf-8')).get('cloze') or []
        if not cl:
            continue
        bits = [int(d['pred'] == d['answer']) for d in cl]
        evals.append({'step': h['step'], 'val_loss': h.get('val_loss'), 'acc': h.get('cloze_acc'), 'bits': bits,
                      'cats': [d['cat'] for d in cl], 'contexts': [d['context'] for d in cl]})
    return evals


def analyse(evals, min_step=0):
    evals = [e for e in evals if e['step'] >= min_step]
    n = len(evals[0]['bits'])
    rows = []
    for e in evals:
        k = sum(e['bits'])
        lo, hi = wilson(k, n)
        blo, bhi = bootstrap(e['bits'])
        rows.append({'step': e['step'], 'val_loss': e['val_loss'], 'correct': k, 'n': n, 'acc': round(k / n, 4),
                     'wilson': [round(lo, 4), round(hi, 4)], 'bootstrap': [round(blo, 4), round(bhi, 4)]})
    flips = []
    for a, b in zip(evals, evals[1:]):
        n10, n01, p = mcnemar_exact(a['bits'], b['bits'])
        flips.append({'from': a['step'], 'to': b['step'], 'right_to_wrong': n10, 'wrong_to_right': n01,
                      'acc_change': round((sum(b['bits']) - sum(a['bits'])) / n, 4), 'mcnemar_p': round(p, 4)})
    pairs = []
    for name, a, b in (('最早 vs 最新', evals[0], evals[-1]), ('中点 vs 最新', evals[len(evals) // 2], evals[-1]),
                       ('倒数第 5 次 vs 最新', evals[max(0, len(evals) - 5)], evals[-1])):
        n10, n01, p = mcnemar_exact(a['bits'], b['bits'])
        pairs.append({'pair': name, 'a_step': a['step'], 'b_step': b['step'], 'a_acc': round(sum(a['bits']) / n, 4),
                      'b_acc': round(sum(b['bits']) / n, 4), 'only_a_right': n10, 'only_b_right': n01, 'mcnemar_p': round(p, 4)})
    mat = np.array([e['bits'] for e in evals])
    tail = mat[-10:] if len(mat) >= 10 else mat
    always_right = int((tail.min(0) == 1).sum())
    always_wrong = int((tail.max(0) == 0).sum())
    flipping = n - always_right - always_wrong
    flip_items = [{'context': evals[-1]['contexts'][i][:40], 'cat': evals[-1]['cats'][i],
                   'right_rate_last10': round(float(tail[:, i].mean()), 2)}
                  for i in np.where((tail.min(0) == 0) & (tail.max(0) == 1))[0]]
    sd = math.sqrt(0.8 * 0.2 / n)
    return {'time': stamp(), 'n_items': n, 'n_evals': len(evals), 'rows': rows, 'flips': flips, 'pairs': pairs,
            'stability_last10': {'always_right': always_right, 'always_wrong': always_wrong, 'flipping': flipping,
                                 'flipping_items': flip_items},
            'mean_flips_per_eval': round(float(np.mean([f['right_to_wrong'] + f['wrong_to_right'] for f in flips])), 2) if flips else None,
            'one_item_pct': round(100 / n, 2), 'sd_at_80pct': round(sd, 4)}


def plot(out_dir, res):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    s = [r['step'] for r in res['rows']]
    acc = [r['acc'] for r in res['rows']]
    lo = [r['wilson'][0] for r in res['rows']]
    hi = [r['wilson'][1] for r in res['rows']]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(s, lo, hi, alpha=0.25, label='95% Wilson CI')
    ax.plot(s, acc, marker='o', ms=3, label=f'cloze accuracy (n={res["n_items"]})')
    ax.set_xlabel('step'), ax.set_ylabel('accuracy'), ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(s, [r['val_loss'] for r in res['rows']], color='tab:red', lw=1, label='val loss')
    ax2.set_ylabel('val loss')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'cloze_ci.png', dpi=140)
    plt.close(fig)
    return 'cloze_ci.png'


def render(res):
    L = ['# 选择题准确率的置信区间与配对检验', '',
         f'- 题数 n = {res["n_items"]}：一题 = {res["one_item_pct"]} 个百分点；准确率 80% 附近的标准误 ≈ '
         f'{res["sd_at_80pct"]:.3f}，95% 区间半宽约 ±{1.96 * res["sd_at_80pct"]:.0%}',
         f'- 相邻两次评估平均有 {res["mean_flips_per_eval"]} 道题翻转（对→错或错→对）',
         f'- 最近 10 次评估：一直对 {res["stability_last10"]["always_right"]} 题，一直错 {res["stability_last10"]["always_wrong"]} 题，'
         f'来回翻 {res["stability_last10"]["flipping"]} 题', '',
         '## 配对检验（精确 McNemar）', '', '| 对比 | 步数 | 准确率 | 只有前者对 | 只有后者对 | p 值 |', '|---|---|---|---|---|---|']
    L += [f'| {p["pair"]} | {p["a_step"]} → {p["b_step"]} | {p["a_acc"]:.0%} → {p["b_acc"]:.0%} | {p["only_a_right"]} | '
          f'{p["only_b_right"]} | {p["mcnemar_p"]} |' for p in res['pairs']]
    L += ['', 'p < 0.05 才能说两次评估"确实不同"；只有翻转的题数够多且方向一边倒时才会显著。', '',
          '## 每次评估（最近 15 次）', '', '| 步数 | val loss | 准确率 | Wilson 95% | bootstrap 95% |', '|---|---|---|---|---|']
    L += [f'| {r["step"]} | {r["val_loss"]} | {r["correct"]}/{r["n"]} = {r["acc"]:.0%} | '
          f'{r["wilson"][0]:.0%}–{r["wilson"][1]:.0%} | {r["bootstrap"][0]:.0%}–{r["bootstrap"][1]:.0%} |'
          for r in res['rows'][-15:]]
    L += ['', '## 相邻评估的翻转（最近 10 对）', '', '| 步数 | 对→错 | 错→对 | 准确率变化 | McNemar p |', '|---|---|---|---|---|']
    L += [f'| {f["from"]} → {f["to"]} | {f["right_to_wrong"]} | {f["wrong_to_right"]} | {f["acc_change"]:+.1%} | {f["mcnemar_p"]} |'
          for f in res['flips'][-10:]]
    if res['stability_last10']['flipping_items']:
        L += ['', '## 来回翻的题（最近 10 次中的答对率）', '']
        L += [f'- [{x["cat"]}] {x["context"]}… 答对率 {x["right_rate_last10"]:.0%}' for x in res['stability_last10']['flipping_items']]
    L += ['', '## 结论怎么下', '',
          '- 单次评估的准确率要带区间报告；两次评估差几个百分点通常落在区间内，不能当作能力变化。',
          '- 比较两个 checkpoint 用配对检验（同一批题），比较两个区间是否重叠过于保守。',
          '- 想让这个指标有分辨力，要么加题（误差 ∝ 1/√n，题数 ×4 区间才减半），要么改看逐题 log-prob 的平均（连续量，噪声小得多）。',
          '- val loss 是约 23 万 token（7 个来源 × 32 段 × 1024）的平均，噪声远小于选择题，所以它比 cloze 更适合跟踪训练进展。']
    if res.get('plot'):
        L += ['', f'![{res["plot"]}]({res["plot"]})']
    return '\n'.join(L) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--min-step', type=int, default=0)
    p.add_argument('--run-dir', default=None)
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir) if a.run_dir else RUN_DIR
    out = run_dir / 'stats'
    out.mkdir(parents=True, exist_ok=True)
    res = analyse(load(run_dir), a.min_step)
    res['plot'] = plot(out, res)
    write_json(out / 'cloze_stats.json', res)
    write_text(out / 'REPORT.md', render(res))
    print(render(res))


if __name__ == '__main__':
    main()
