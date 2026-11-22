"""F 第一阶段（step 20）加 presence_penalty 的评测（2026-11-22 15:17 用户要求"进行 F 加惩罚的评测"，约 40 元）。
设置和 F 的完整评测（f/eval_s20，sft/eval_e.py：temperature 0.6、top_p 0.95、最长 16384 token、同样提示、同样判分）相同，只多 presence_penalty：
1.5 和 2.0 两组（SGLang 允许 -2 到 2）；AMC23 / AIME24 / AIME25 / AIME 2026 每题每组 8 次，MATH-500 每题每组 2 次，共 4,064 条。
比较：和 F 的完整评测按题配对（每题：试验组均值 - 基线均值），bootstrap 10000 次；四集平均（MATH-500、AMC23、AIME24、AIME25 等权）是主指标。
参考：E 的完整评测（sft/eval_full）和 E 加 presence_penalty 1.5 的 AIME 测试（f/eval_reppen_E），用来看惩罚和 RL 能不能叠加。
  gen  ：python f/eval_pp_F.py gen --model DIR --out DIR --shard i --nshards 4 [--dry]
  score：python f/eval_pp_F.py score --out DIR [--selftest] [--nproc 48]  -> run/audit/F_PRESENCE_PENALTY_RESULT.md"""
import argparse
import collections
import glob
import json
import os
import sys
import zlib

import numpy as np

R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/sft')
sys.path.insert(0, R + '/ma')
import eval_e  # noqa: E402

SETS = ['math500', 'amc23', 'aime24', 'aime25', 'aime26']
SETS4 = ['math500', 'amc23', 'aime24', 'aime25']
AIME = ['aime24', 'aime25', 'aime26']
ARMS = [('pp1.5', 1.5), ('pp2.0', 2.0)]
BASE = R + '/f/eval_s20'
EBASE = R + '/sft/eval_full'
EPP = R + '/f/eval_reppen_E'
BUCKETS = ['0（基线从没做对）', '(0, 25%]', '(25%, 75%)', '[75%, 100%]']
NAMES = {'math500': 'MATH-500', 'amc23': 'AMC23', 'aime24': 'AIME24', 'aime25': 'AIME25', 'aime26': 'AIME 2026（留出）'}


def items_with_arms():
    cnt, out = collections.Counter(), []
    for it in eval_e.eval_rows():
        if it['ds'] not in SETS:
            continue
        k = (it['ds'], it['q'])
        i = cnt[k]
        cnt[k] += 1
        if it['ds'] == 'math500':
            for name, pp in ARMS:
                for j in range(2):
                    out.append(dict(it, arm=name, pp=pp, rep_idx=j))
        else:
            name, pp = ARMS[0] if i < 8 else ARMS[1]
            out.append(dict(it, arm=name, pp=pp, rep_idx=i))
    return out


def gen(a):
    items = items_with_arms()[a.shard::a.nshards]
    if a.dry:
        print('shard', a.shard, 'items', len(items), sorted(collections.Counter((it['ds'], it['arm']) for it in items).items()))
        return
    from transformers import AutoTokenizer
    import sglang as sgl
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': it['q'] + '\n\n' + eval_e.INSTR}], tokenize=False, add_generation_prompt=True)
               for it in items]
    eng = sgl.Engine(model_path=a.model, tp_size=1, mem_fraction_static=0.85, context_length=a.max_new + 2048,
                     random_seed=777 + a.shard, log_level='warning')
    sps = [{'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': a.max_new, 'stop_token_ids': [151645, 151643],
            'skip_special_tokens': False, 'presence_penalty': it['pp']} for it in items]
    outs = eng.generate(prompt=prompts, sampling_params=sps)
    tmp = os.path.join(a.out, 'gen_%d.jsonl.tmp' % a.shard)
    with open(tmp, 'w') as fo:
        for it, o in zip(items, outs):
            mi = o.get('meta_info', {}) or {}
            fr = mi.get('finish_reason')
            fr = fr.get('type') if isinstance(fr, dict) else fr
            fo.write(json.dumps(dict(it, output=o['text'], finish=fr, n_tok=mi.get('completion_tokens')), ensure_ascii=False) + '\n')
    os.replace(tmp, os.path.join(a.out, 'gen_%d.jsonl' % a.shard))
    print('GEN_DONE', len(items), flush=True)
    eng.shutdown()


_ROWS = []


def _judge_idx(i):
    r = _ROWS[i]
    o = r['output'].replace('<|im_end|>', '').replace('<|endoftext|>', '')
    tail = o.split('</think>')[-1] if '</think>' in o else o[-3000:]
    return eval_e.judge(r['ds'], tail, r['gt'])


def load(d, nproc):
    """判分和 eval_e.load_e 完全相同（正确率、写完、含中文、行重复）；另加重复循环和 token 数。"""
    global _ROWS
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'gen_*.jsonl'))):
        rows += [json.loads(l) for l in open(f)]
    _ROWS = rows
    # 不能用多进程：reward.py 的 math_verify 判等自己开子进程池，放在 Pool 的守护进程里会静默失败（MATH-500 少判对 5 个点）。
    # 逐条判分和 eval_e.load_e 一模一样，2,564 条约 10 秒。nproc 参数保留但不再使用。
    accs = [_judge_idx(i) for i in range(len(rows))]
    for r, acc in zip(rows, accs):
        o = r['output'].replace('<|im_end|>', '').replace('<|endoftext|>', '')
        r['acc'] = float(acc)
        r['complete'] = float(r.get('finish') == 'stop' and '</think>' in o)
        r['cjk'] = float(bool(eval_e.CJK.search(o)))
        r['rep'] = float(eval_e.rep_line(o))
        lines = [l.strip() for l in r['output'].split('\n') if len(l.strip()) > 20]
        c = collections.Counter(lines)
        top = c.most_common(1)[0][1] if c else 0
        t8 = r['output'][-8000:].encode('utf-8', 'ignore')
        r['loop'] = float(top >= 20 or len(zlib.compress(t8, 6)) / max(1, len(t8)) < 0.12)
        r['tok'] = float(r.get('n_tok') or 0)
    return rows


def group(rows, arm=None):
    G = collections.defaultdict(list)
    for r in rows:
        if arm is None or r.get('arm') == arm:
            G[(r['ds'], r['q'])].append(r)
    return G


def mean_of(X, k, f):
    return float(np.mean([r[f] for r in X[k]]))


def set_mean(X, ks, f):
    return float(np.mean([mean_of(X, k, f) for k in ks])) if ks else float('nan')


def bucket(p):
    if p == 0:
        return BUCKETS[0]
    if p <= 0.25:
        return BUCKETS[1]
    if p < 0.75:
        return BUCKETS[2]
    return BUCKETS[3]


def f1(x):
    return '—' if x != x else '%.1f%%' % x


def score(a):
    rng = np.random.default_rng(0)

    def paired(X, Y, ks, col):
        d = np.array([mean_of(X, k, col) - mean_of(Y, k, col) for k in ks])
        bt = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        return float(d.mean()), bt

    def ci(m, bt):
        return '%+.1f [%+.1f, %+.1f]' % (100 * m, 100 * np.percentile(bt, 2.5), 100 * np.percentile(bt, 97.5))

    def pct(X, ds, col):
        return 100 * set_mean(X, [k for k in X if k[0] == ds], col)

    base = load(BASE, a.nproc)
    ebase = load(EBASE, a.nproc)
    epp = [r for r in load(EPP, a.nproc) if r['ds'] in AIME]
    if a.selftest:
        new, cnt = [], collections.Counter()
        for r in base:
            k = (r['ds'], r['q'])
            arm = ARMS[0][0] if (r['ds'] == 'math500' or cnt[k] < 8) else ARMS[1][0]
            cnt[k] += 1
            new.append(dict(r, arm=arm))
    else:
        new = load(a.out, a.nproc)
    B, EB, EP = group(base), group(ebase), group(epp, 'pp1.5')
    AR = {arm: group(new, arm) for arm, _ in ARMS}
    L = ['# F 第一阶段（step 20）加 presence_penalty 的评测', '',
         '除 presence_penalty 以外的设置和 F 的完整评测相同（temperature 0.6、top_p 0.95、最长 16384 token、同样判分）。'
         '基线 = F 的完整评测（f/eval_s20：MATH-500 每题 1 次，其他每题 16 次）；试验组 MATH-500 每题 2 次，其他每题 8 次。'
         '差值按题目配对，方括号是 bootstrap 95% 区间；四集平均 = MATH-500、AMC23、AIME24、AIME25 等权。'
         '"重复循环" = 同一行出现 20 次以上或结尾高度重复；"行重复" = 同一行出现 5 次以上（完整评测表的口径）。'
         'E + presence_penalty 1.5 来自上午的 AIME 测试（f/eval_reppen_E，每题 8 次），这里用同一个判分重新判过。', '',
         '## 总览（正确率）', '',
         '| 模型 | 四集平均 | MATH-500 | AMC23 | AIME24 | AIME25 | AIME 2026 | AIME 三套写完比例 |', '|---|---|---|---|---|---|---|---|']
    models = [('E（SFT）', EB), ('F 第一阶段', B)] + [('F + presence_penalty %s' % pp, AR[arm]) for arm, pp in ARMS] + [('E + presence_penalty 1.5（只测了 AIME）', EP)]
    for name, X in models:
        v = [pct(X, ds, 'acc') for ds in SETS]
        four = float(np.mean(v[:4])) if all(x == x for x in v[:4]) else float('nan')
        comp = 100 * set_mean(X, [k for k in X if k[0] in AIME], 'complete')
        L.append('| %s | %s | %s |' % (name, ' | '.join(f1(x) for x in [four] + v), f1(comp)))
    L += ['', '## 基线：F 第一阶段（presence_penalty 0）；括号里是 E', '',
          '| 评测集 | 正确率 | 写完的比例 | 重复循环 | 行重复 | 含中文 | 平均 token |', '|---|---|---|---|---|---|---|']
    for ds in SETS:
        L.append('| %s | %s（%s） | %s（%s） | %s（%s） | %s（%s） | %s | %.0f（%.0f） |' % (
            NAMES[ds], f1(pct(B, ds, 'acc')), f1(pct(EB, ds, 'acc')), f1(pct(B, ds, 'complete')), f1(pct(EB, ds, 'complete')),
            f1(pct(B, ds, 'loop')), f1(pct(EB, ds, 'loop')), f1(pct(B, ds, 'rep')), f1(pct(EB, ds, 'rep')), f1(pct(B, ds, 'cjk')),
            pct(B, ds, 'tok') / 100, pct(EB, ds, 'tok') / 100))

    def aime_rows(X):
        return [r for k, v in X.items() if k[0] in AIME for r in v]

    def comp_line(name, X):
        rows = aime_rows(X)
        comp = [r for r in rows if r['complete']]
        unf = [r for r in rows if not r['complete']]
        return '- %s（AIME 三套）：写完的回答里正确的比例 %.1f%%；没写完的回答里重复循环占 %.1f%%；平均 token %.0f' % (
            name, 100 * np.mean([r['acc'] for r in comp]) if comp else 0, 100 * np.mean([r['loop'] for r in unf]) if unf else 0,
            np.mean([r['tok'] for r in rows]))

    for arm, pp in ARMS:
        A = AR[arm]
        n = sum(len(v) for v in A.values())
        L += ['', '## presence_penalty %s（%d 条）对 F' % (pp, n), '',
              '| 评测集 | 正确率 | 写完的比例 | 重复循环 | 行重复 | 含中文 | 平均 token | 正确率差值 | 写完比例差值 |',
              '|---|---|---|---|---|---|---|---|---|']
        boots, diffs = {}, {}
        for ds in SETS:
            ks = [k for k in B if k[0] == ds and k in A]
            if not ks:
                continue
            row = '| %s | %s | %s | %s | %s | %s | %.0f |' % (
                NAMES[ds], f1(100 * set_mean(A, ks, 'acc')), f1(100 * set_mean(A, ks, 'complete')), f1(100 * set_mean(A, ks, 'loop')),
                f1(100 * set_mean(A, ks, 'rep')), f1(100 * set_mean(A, ks, 'cjk')), set_mean(A, ks, 'tok'))
            for col in ('acc', 'complete'):
                m, bt = paired(A, B, ks, col)
                if col == 'acc':
                    boots[ds], diffs[ds] = bt, m
                row += ' %s |' % ci(m, bt)
            L.append(row)
        L.append('')
        if all(ds in boots for ds in SETS4):
            bm = np.mean([boots[ds] for ds in SETS4], axis=0)
            L.append('- 四集平均正确率差值：%s' % ci(float(np.mean([diffs[ds] for ds in SETS4])), bm))
        ks = [k for k in B if k[0] in AIME and k in A]
        for col, nm in (('acc', '正确率'), ('complete', '写完比例')):
            m, bt = paired(A, B, ks, col)
            L.append('- AIME 三套合计%s差值：%s（%d 题）' % (nm, ci(m, bt), len(ks)))
        L.append(comp_line('presence_penalty %s' % pp, A))
        L += ['', '按 F 基线每题正确率分组（AMC23 + AIME 三套，基线每题 16 次）：', '',
              '| F 基线每题正确率 | 题数 | 正确率差值 | 写完比例差值 |', '|---|---|---|---|']
        bk = collections.defaultdict(list)
        for k in A:
            if k[0] != 'math500' and k in B:
                bk[bucket(mean_of(B, k, 'acc'))].append(k)
        for b in BUCKETS:
            kk = bk.get(b, [])
            if kk:
                L.append('| %s | %d | %+.1f | %+.1f |' % (b, len(kk), 100 * np.mean([mean_of(A, k, 'acc') - mean_of(B, k, 'acc') for k in kk]),
                                                       100 * np.mean([mean_of(A, k, 'complete') - mean_of(B, k, 'complete') for k in kk])))
    L += ['', comp_line('基线 F', B), comp_line('E', EB), comp_line('E + presence_penalty 1.5', EP)]
    A1, A2 = AR[ARMS[0][0]], AR[ARMS[1][0]]
    L += ['', '## presence_penalty 2.0 对 1.5（同一个 F，按题配对）', '', '| 评测集 | 正确率差值 | 写完比例差值 |', '|---|---|---|']
    boots, diffs = {}, {}
    for ds in SETS:
        ks = [k for k in A1 if k[0] == ds and k in A2]
        if not ks:
            continue
        ma_, bta = paired(A2, A1, ks, 'acc')
        mc, btc = paired(A2, A1, ks, 'complete')
        boots[ds], diffs[ds] = bta, ma_
        L.append('| %s | %s | %s |' % (NAMES[ds], ci(ma_, bta), ci(mc, btc)))
    if all(ds in boots for ds in SETS4):
        L += ['', '- 四集平均正确率差值（2.0 − 1.5）：%s' % ci(float(np.mean([diffs[ds] for ds in SETS4])), np.mean([boots[ds] for ds in SETS4], axis=0))]
    ks = [k for k in B if k[0] in AIME and k in EB and k in EP and k in A1]
    L += ['', '## 惩罚和 RL 能不能叠加（AIME 三套合计，%d 题，按题配对）' % len(ks), '',
          '"双重差值" = （F 加惩罚 − F）−（E 加惩罚 − E）：接近 0 说明两者效果可以相加；明显小于 0 说明 RL 已经拿走了惩罚能带来的一部分好处。', '',
          '| 比较 | 正确率差值 | 写完比例差值 |', '|---|---|---|']
    comps = [('E + 1.5 − E', lambda k, c: mean_of(EP, k, c) - mean_of(EB, k, c)),
             ('F + 1.5 − F', lambda k, c: mean_of(A1, k, c) - mean_of(B, k, c)),
             ('双重差值（1.5）', lambda k, c: (mean_of(A1, k, c) - mean_of(B, k, c)) - (mean_of(EP, k, c) - mean_of(EB, k, c))),
             ('F + 1.5 − E + 1.5（同样加惩罚时 RL 的作用）', lambda k, c: mean_of(A1, k, c) - mean_of(EP, k, c)),
             ('F + 2.0 − E + 1.5', lambda k, c: mean_of(A2, k, c) - mean_of(EP, k, c)),
             ('F + 1.5 − E（合计）', lambda k, c: mean_of(A1, k, c) - mean_of(EB, k, c)),
             ('F + 2.0 − E（合计）', lambda k, c: mean_of(A2, k, c) - mean_of(EB, k, c))]
    for nm, fn in comps:
        cells = []
        for col in ('acc', 'complete'):
            d = np.array([fn(k, col) for k in ks if k in A2])
            bt = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            cells.append(ci(float(d.mean()), bt))
        L.append('| %s | %s | %s |' % (nm, cells[0], cells[1]))
    out = '/tmp/F_PP_selftest.md' if a.selftest else R + '/run/audit/F_PRESENCE_PENALTY_RESULT.md'
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('SCORE_DONE', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--model')
    p.add_argument('--out', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--max-new', type=int, default=16384)
    p.add_argument('--nproc', type=int, default=48)
    p.add_argument('--dry', action='store_true')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gen(a) if a.cmd == 'gen' else score(a)
