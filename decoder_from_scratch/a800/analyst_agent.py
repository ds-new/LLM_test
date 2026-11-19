"""结果分析智能体：把评估智能体测出的原始数字变成有统计依据的结论交给决策智能体；训练结束后写最终分析报告和下一步建议。

分工：评估智能体只负责"测"（固定口径产出数字），分析智能体负责"读"——
  * 噪声：用相邻两次评估差值的稳健标准差（1.4826 * MAD / sqrt(2)）估计每个来源单次评估的噪声 sigma；
  * 趋势：最近 K 次（默认 8 次）评估对已训练 token 做线性回归，得到斜率与 t 值：
      t <= -3 → improving；t >= 3 且窗口内拟合上升超过 3 sigma → regressing（真的在变差）；否则 plateau。
    决策智能体的"遗忘 → 提高采样权重"规则只认 regressing，不再被 0.001 级的评估噪声反复触发；
  * 训练-验证差距、吞吐和梯度异常、选择题 Wilson 置信区间、预计剩余时间；
  * 训练结束后：画曲线（analysis/curves.png），汇总分来源结果和各项探测结论，写 analysis/FINAL_ANALYSIS.md。
每次结论都写进 analysis/history.jsonl 和消息总线，便于复盘。
"""

import json
import math
from pathlib import Path

import numpy as np

from .common import RUN_DIR, stamp, write_json, write_text

NAME = 'analyst'
LN2 = math.log(2)


def read_jsonl(path):
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def robust_sigma(values):
    """单次评估噪声。相邻差值含两份噪声，所以除以 sqrt(2)；先减去差值中位数，去掉趋势的影响。"""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) < 5:
        return None
    d = np.diff(v)
    mad = float(np.median(np.abs(d - np.median(d))))
    return 1.4826 * mad / math.sqrt(2)


def linear_trend(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return None
    xm = x - x.mean()
    sxx = float((xm ** 2).sum())
    if sxx <= 0:
        return None
    slope = float((xm * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + slope * xm)
    se = math.sqrt(max(float((resid ** 2).sum()) / (n - 2), 1e-12) / sxx)
    return {'slope': slope, 't': slope / se, 'n': n, 'span': float(x[-1] - x[0])}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


class AnalystAgent:
    def __init__(self, config, bus, run_dir=None):
        self.config, self.bus = config, bus
        self.run_dir = Path(run_dir or RUN_DIR)
        self.out = self.run_dir / 'analysis'
        self.cfg = {'window': 8, 't_threshold': 3.0, 'sigma_mult': 3.0, **(config.get('analysis') or {})}

    def say(self, recipient, kind, content):
        return self.bus.post(NAME, recipient, kind, content) if self.bus else None

    def history(self):
        by_step = {}
        for h in read_jsonl(self.run_dir / 'eval_history.jsonl'):
            if h.get('val_by_source') and 'step' in h:
                by_step[h['step']] = h
        return [by_step[s] for s in sorted(by_step)]

    def cloze_count(self):
        try:
            data = json.loads((self.run_dir / 'samples' / 'latest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        c = data.get('cloze')
        return len(c) if isinstance(c, (list, dict)) and c else None

    def classify(self, x, ys, sigma):
        tr = linear_trend(x, ys)
        if tr is None:
            return 'unknown', None, None
        fitted = tr['slope'] * tr['span']
        if tr['t'] >= self.cfg['t_threshold'] and sigma is not None and fitted > self.cfg['sigma_mult'] * sigma:
            return 'regressing', tr, fitted
        if tr['t'] <= -self.cfg['t_threshold']:
            return 'improving', tr, fitted
        return 'plateau', tr, fitted

    @staticmethod
    def train_loss_near(train, step, width=300):
        w = [r['loss'] for r in train if step - width < r['step'] <= step]
        return round(float(np.mean(w)), 4) if w else None

    def analyze(self, segment=None, plan=None, notify=True):
        """每次评估后调用：返回结构化结论，并把要点作为报告发给决策智能体。"""
        hist = self.history()
        if len(hist) < 4:
            return {'enough_data': False, 'regressing_sources': [], 'findings': []}
        W = int(self.cfg['window'])
        recent = hist[-W:]
        x = [h['tokens_seen'] / 1e9 for h in recent]
        result = {'time': stamp(), 'step': hist[-1]['step'], 'tokens_B': round(hist[-1]['tokens_seen'] / 1e9, 3),
                  'window_start_step': recent[0]['step'], 'enough_data': True, 'sources': {}}
        sigma = robust_sigma([h['val_loss'] for h in hist][-4 * W:])
        status, tr, _ = self.classify(x, [h['val_loss'] for h in recent], sigma)
        result['overall'] = {'latest': hist[-1]['val_loss'], 'best': min(h['val_loss'] for h in hist), 'status': status,
                             'sigma': round(sigma, 4) if sigma else None,
                             'per_B': round(tr['slope'], 4) if tr else None, 't': round(tr['t'], 2) if tr else None}
        for name in sorted(recent[-1]['val_by_source']):
            series = [h['val_by_source'].get(name) for h in hist]
            ys = [h['val_by_source'].get(name) for h in recent]
            if None in ys:
                continue
            s = robust_sigma(series[-4 * W:])
            st, t2, f2 = self.classify(x, ys, s)
            result['sources'][name] = {'latest': ys[-1], 'best': min(v for v in series if v is not None), 'status': st,
                                       'sigma': round(s, 4) if s else None,
                                       'per_B': round(t2['slope'], 4) if t2 else None, 't': round(t2['t'], 2) if t2 else None,
                                       'window_change': round(f2, 4) if f2 is not None else None}
        reg = [n for n, s in result['sources'].items() if s['status'] == 'regressing']
        result['regressing_sources'] = reg
        o = result['overall']
        findings = []
        if o['per_B'] is not None:
            findings.append(f'最近 {len(recent)} 次评估总体验证 loss 每 1B token 变化 {o["per_B"]:+.4f}（t={o["t"]}，单次噪声 σ≈{o["sigma"]}），判定 {status}')
        if reg:
            findings.append('显著变差的来源：' + '，'.join(
                f'{n}（窗口内拟合 +{result["sources"][n]["window_change"]}，σ≈{result["sources"][n]["sigma"]}）' for n in reg))
        else:
            findings.append('没有来源显著变差（单次的小幅上升都在噪声范围内）')
        train = read_jsonl(self.run_dir / 'train_log.jsonl')
        if train:
            t_now, t_old = self.train_loss_near(train, recent[-1]['step']), self.train_loss_near(train, recent[0]['step'])
            if t_now is not None and t_old is not None:
                gap_now, gap_old = round(recent[-1]['val_loss'] - t_now, 4), round(recent[0]['val_loss'] - t_old, 4)
                result['train_val_gap'] = {'now': gap_now, 'window_start': gap_old}
                if gap_now - gap_old > 0.05:
                    findings.append(f'训练-验证差距从 {gap_old} 扩大到 {gap_now}：注意数据重复或训练配比偏离验证配比')
        acc, n_q = hist[-1].get('cloze_acc'), self.cloze_count()
        if acc is not None and n_q:
            ci = wilson(round(acc * n_q), n_q)
            result['cloze'] = {'acc': acc, 'n': n_q, 'ci95': ci}
            findings.append(f'选择题 {acc:.0%}（n={n_q}，95% 区间 {ci[0]:.0%}–{ci[1]:.0%}），只作参考')
        if segment:
            prev = [r['tok_s'] for r in read_jsonl(self.out / 'history.jsonl')[-8:] if r.get('tok_s')]
            tok_s = segment.get('segment_tok_s')
            anomalies = []
            if prev and tok_s and tok_s < 0.9 * float(np.median(prev)):
                anomalies.append(f'吞吐 {tok_s} tok/s 低于近期中位数 {float(np.median(prev)):.0f} 的 90%')
            gm, gx = segment.get('grad_norm_median'), segment.get('grad_norm_max')
            if gm and gx and gx > 5 * gm:
                anomalies.append(f'梯度范数最大值 {gx} 超过中位数 {gm} 的 5 倍')
            if segment.get('nonfinite_steps'):
                anomalies.append(f'本段出现 {segment["nonfinite_steps"]} 个非有限步')
            result['segment'] = {'tok_s': tok_s, 'anomalies': anomalies}
            findings += anomalies
            if plan and tok_s:
                left = max(0, plan['total_steps'] - segment['step'])
                hours = left * plan['tokens_per_step'] / tok_s / 3600 * 1.03
                result['eta_hours'] = round(hours, 2)
                findings.append(f'剩余 {left:,} 步，预计 {hours:.1f} 小时（含评估开销）')
        result['findings'] = findings
        write_json(self.out / 'latest.json', result)
        with (self.out / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time': result['time'], 'step': result['step'], 'overall': o, 'regressing': reg,
                                'tok_s': (result.get('segment') or {}).get('tok_s'), 'eta_hours': result.get('eta_hours')},
                               ensure_ascii=False) + '\n')
        if notify:
            self.say('decision', 'report', {'action': 'analysis', 'step': result['step'], 'overall': o,
                                            'regressing_sources': reg, 'findings': findings[:6]})
        return result

    def final_report(self, state):
        """训练和各项探测结束后：曲线图 + 分来源表 + 探测结论 + 规则化的下一步建议。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        res = self.analyze(notify=False)
        hist = self.history()
        if not hist:
            raise RuntimeError('没有评估记录，无法做最终分析')
        train = read_jsonl(self.run_dir / 'train_log.jsonl')
        plan = state.get('plan') or {}
        final = state.get('final') or {}
        tps = plan.get('tokens_per_step', 524288)
        decay_start = int(plan['total_steps'] * (1 - plan.get('decay_fraction', 0.2))) if plan.get('total_steps') else None
        names = sorted(hist[-1]['val_by_source'])
        head = hist[min(3, len(hist) - 1)]

        def at_step(step):
            return min(hist, key=lambda h: abs(h['step'] - step))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        ax = axes[0]
        if len(train) > 30:
            k = 15
            loss = np.array([r['loss'] for r in train])
            tok = np.array([r['tokens'] for r in train])[k - 1:] / 1e9
            ax.plot(tok, np.convolve(loss, np.ones(k) / k, mode='valid'), lw=0.8, alpha=0.7, label='train (moving avg)')
        ax.plot([h['tokens_seen'] / 1e9 for h in hist], [h['val_loss'] for h in hist], lw=1.8, label='val (weighted)')
        ax.set_ylim(min(h['val_loss'] for h in hist) - 0.25, head['val_loss'] + 0.2)
        ax.set_title('overall loss')
        ax = axes[1]
        for n in names:
            ax.plot([h['tokens_seen'] / 1e9 for h in hist], [h['val_by_source'].get(n) for h in hist], lw=1.2, label=n)
        ax.set_ylim(min(min(h['val_by_source'].values()) for h in hist) - 0.1, max(head['val_by_source'].values()) + 0.2)
        ax.set_title('val loss by source')
        for a in axes:
            if decay_start:
                a.axvline(decay_start * tps / 1e9, ls='--', c='gray', lw=0.8)
            a.set_xlabel('tokens (B)')
            a.legend(fontsize=8)
        fig.tight_layout()
        self.out.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.out / 'curves.png', dpi=130)
        plt.close(fig)

        cpt = (state.get('tokenizer') or {}).get('chars_per_token') or {}
        epochs = (state.get('last_segment') or {}).get('epochs') or {}
        val_final = final.get('val_by_source') or hist[-1]['val_by_source']
        test_final = final.get('test_by_source') or {}
        final_val = final.get('val_loss', hist[-1]['val_loss'])
        h_mid = at_step(hist[-1]['step'] // 2)
        h_decay = at_step(decay_start) if decay_start else None
        decay_gain = round(h_decay['val_loss'] - final_val, 4) if h_decay and h_decay is not hist[-1] else None
        pre = [h for h in hist if decay_start and h['step'] <= decay_start][-int(self.cfg['window']):]
        stable = linear_trend([h['tokens_seen'] / 1e9 for h in pre], [h['val_loss'] for h in pre]) if len(pre) >= 4 else None

        L = ['# 最终结果分析（结果分析智能体）', '', f'- 生成时间：{stamp()}',
             f'- 已训练 {hist[-1]["tokens_seen"] / 1e9:.2f}B token、{hist[-1]["step"]:,} 步；最终验证 loss {final_val}'
             f'（ppl {math.exp(final_val):.2f}），测试 loss {final.get("test_loss")}']
        if decay_gain is not None:
            L.append(f'- WSD 衰减阶段（第 {h_decay["step"]:,} 步起）验证 loss 下降 {decay_gain}')
        if stable:
            L.append(f'- 恒定学习率阶段末期：每 1B token 验证 loss 变化 {stable["slope"]:+.4f}（t={stable["t"]:.1f}）')
        L += ['', '![curves](curves.png)', '', '## 分来源', '',
              '| 来源 | 中段 | 衰减起点 | 最终 | 衰减阶段下降 | 单次噪声 σ | bits/字符 | 已过遍数 | 测试 loss |',
              '|---|---|---|---|---|---|---|---|---|']
        for n in names:
            v = val_final.get(n)
            dv = h_decay['val_by_source'].get(n) if h_decay else None
            s = (res.get('sources') or {}).get(n, {})
            bpc = round(v / cpt[n] / LN2, 3) if v and cpt.get(n) else '-'
            gain = round(dv - v, 4) if dv is not None and v is not None else '-'
            L.append(f'| {n} | {h_mid["val_by_source"].get(n)} | {dv if dv is not None else "-"} | {v} | {gain} | '
                     f'{s.get("sigma", "-")} | {bpc} | {epochs.get(n, "-")} | {test_final.get(n, "-")} |')

        L += ['', '## 其他探测结论', '']
        pk = state.get('passk') or {}
        ready = []
        if pk.get('summary'):
            for m, rows in pk['summary'].items():
                L.append(f'- pass@k（{m}）：' + '；'.join(
                    f'{r["task"]} pass@1 {r.get("pass@1")} / pass@16 {r.get("pass@16")} → {r.get("verdict")}' for r in rows))
            ready = [r['task'] for r in next(iter(pk['summary'].values())) if r.get('verdict') == '适合直接 RLVR']
        elif pk.get('error'):
            L.append(f'- pass@k 探测出错：{pk["error"]}')
        for key, label in (('token_probe', 'token 探测'), ('weight_edit', '参数空间编辑')):
            item = state.get(key) or {}
            if item:
                L.append(f'- {label}：{item.get("verdict", item.get("error"))}')
        pf = state.get('profile') or {}
        if pf:
            L.append(f'- profiler：MFU {pf.get("mfu")}，{pf.get("tok_s")} tok/s' if 'mfu' in pf else f'- profiler 出错：{pf.get("error")}')

        recs = []
        max_ep = max(epochs.values()) if epochs else 0
        if stable and stable['slope'] > -0.01 and max_ep >= 1.5:
            recs.append(f'恒定学习率阶段末期每 1B token 验证 loss 只变化 {stable["slope"]:+.4f}，数据已重复约 {max_ep:.1f} 遍：'
                        '同一份数据继续加训收益很低；若继续预训练，应先扩充唯一数据（尤其是中文和数学），而不是加步数。')
        if decay_gain is not None and stable is not None:
            recs.append(f'衰减阶段带来 {decay_gain} 的下降，明显大于恒定阶段末期的进展：下一轮可以做"衰减比例 20% vs 30%"的对照，'
                        '或在衰减阶段换入更高质量的数据（退火数据）。')
        if pk.get('summary'):
            if ready:
                recs.append(f'pass@k 显示 {ready} 已有足够多"部分答对"的分组：可以直接用 verl 做 RLVR（GRPO）。')
            else:
                recs.append('pass@k 显示没有任务适合直接 RLVR（pass@16 太低，或分组全对/全错没有学习信号）：先做 SFT（指令 + 数学/代码推理数据），再做 RL。')
        g = final.get('greedy_rep4', hist[-1].get('greedy_rep4'))
        if g and g > 0.3:
            recs.append(f'贪心解码 4-gram 重复率 {g}：基座模型直接对话会复读，做生成类评测或部署前需要 SFT，或者加 repetition penalty。')
        if res.get('regressing_sources'):
            recs.append(f'训练末期仍在显著变差的来源：{res["regressing_sources"]}，下一轮应提高这些来源的唯一数据量或配比。')
        recs.append('38 道选择题的 95% 区间约 ±13 个百分点，区分不了 checkpoint：建议补一轮正式基准（C-Eval / CMMLU / HellaSwag / ARC / '
                    'GSM8K / HumanEval，基座模型用 log-prob 打分）。')
        L += ['', '## 下一步建议（按规则自动生成，需要人工判断）', ''] + [f'{i}. {r}' for i, r in enumerate(recs, 1)]
        L += ['', '## 训练末期的分析结论', ''] + [f'- {f}' for f in res.get('findings', [])]
        write_text(self.out / 'FINAL_ANALYSIS.md', '\n'.join(L) + '\n')
        summary = {'final_val': final_val, 'test_loss': final.get('test_loss'), 'decay_gain': decay_gain, 'next_steps': recs}
        write_json(self.out / 'final_summary.json', summary)
        self.say('decision', 'report', {'action': 'final_analysis', **summary})
        return summary
