"""验证器审计：用独立的宽松判分和格式检查，找出可能的误判（正确判成错 / 错误判成对），导出给人工核对。"""
import json, re, glob, collections, sys
sys.path.insert(0, 'ma')
from verl.utils.reward_score import math_dapo
import reward as R

OUT = 'run/audit'
import os; os.makedirs(OUT, exist_ok=True)

def norm(s):
    if s is None: return None
    s = str(s).strip()
    for _ in range(3):
        s = re.sub(r'^(\*\*|answer\s*:|Answer\s*:|\\\(|\\\[|\$)\s*', '', s.strip())
        s = re.sub(r'\s*(\*\*|\\\)|\\\]|\$)$', '', s.strip())
    s = re.sub(r'\\boxed\s*\{(.*)\}', r'\1', s)
    s = s.strip().strip('$').strip()
    s = re.sub(r'\\(text|mathrm|textbf|mbox)\{([^}]*)\}', r'\2', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac').replace('\\left', '').replace('\\right', '')
    s = s.replace('^\\circ', '').replace('^{\\circ}', '').replace('\\%', '').replace('%', '').replace('\\!', '').replace('\\,', '').replace('\\ ', '')
    s = re.sub(r'\s+', '', s).rstrip('.').lower()
    s = re.sub(r'^[a-z]\s*=', '', s)  # x=5 -> 5
    s = re.sub(r'(?<=\d),(?=\d{3}(\D|$))', '', s)  # 1,000 -> 1000
    s = re.sub(r'(degrees?|units?|cm|meters?|inches|dollars?|cents?)$', '', s)
    s = re.sub(r'^(-?\d+)\.0+$', r'\1', s)
    return s

def answers(txt):
    return [m.strip() for m in re.findall(r'Answer\s*:\s*(.+)', txt)]

def boxed_all(txt):
    out = []
    for m in re.finditer(r'\\boxed\{', txt):
        j, d = m.end(), 1
        while j < len(txt) and d:
            d += {'{': 1, '}': -1}.get(txt[j], 0); j += 1
        if d == 0: out.append(txt[m.end():j-1])
    return out

rows = []
for f in sorted(glob.glob('/root/r1zero_valgen/*.jsonl')):
    for l in open(f):
        d = json.loads(l); rows.append(d)

src_of = {}
import pandas as pd
v = pd.read_parquet('run/data/val_select.parquet')
for _, r in v.iterrows():
    q = r['prompt'][0]['content']
    src_of[q[-300:]] = r['data_source']

C = collections.Counter(); fn, fp, misc = [], [], []
for d in rows:
    out, gt, acc = d['output'], str(d['gts']), float(d['acc']) > 0.5
    src = next((s for k, s in src_of.items() if d['input'].find(k[-200:]) >= 0), '?')
    ans = answers(out); last_ans = ans[-1] if ans else None
    tail_ans = answers(out[-600:])
    bx = boxed_all(out)
    md = math_dapo.compute_score(out, gt)
    C['total'] += 1; C['acc'] += acc
    C['no_answer_line'] += (not ans)
    C['no_answer_in_tail600'] += (not tail_ans)
    C['judged_by_math_dapo'] += bool(md.get('acc'))
    C['judged_by_mathverify_fallback'] += acc and not md.get('acc')
    distinct = {norm(a) for a in ans}
    C['multi_distinct_answers'] += len(distinct) > 1
    ng = norm(gt)
    loose_last = norm(last_ans) == ng if last_ans else False
    loose_box = bool(bx) and norm(bx[-1]) == ng
    any_match = any(norm(a) == ng for a in ans) or any(norm(b) == ng for b in bx)
    item = dict(step=d['step'], src=src, gt=gt, pred=d['pred'], last_answer_line=(last_ans or '')[:200],
                last_boxed=(bx[-1] if bx else '')[:200], n_answer_lines=len(ans), tail=out[-400:])
    if not acc and (loose_last or loose_box):
        fn.append(dict(kind='loose_match_but_judged_wrong', **item)); C['FN_candidate'] += 1
    elif not acc and any_match:
        C['wrong_but_gt_appeared_earlier'] += 1; misc.append(dict(kind='gt_appeared_earlier', **item))
    if acc:
        hedge = len(distinct) > 1 and len(tail_ans) > 1
        multi_in_line = last_ans and re.search(r'(,|\\text\{ or \}|\bor\b|;)', last_ans) and not re.search(r'(,|\bor\b|;)', gt)
        if hedge or multi_in_line or not (loose_last or loose_box):
            fp.append(dict(kind='hedge' if hedge else ('multi_values' if multi_in_line else 'accepted_by_symbolic_only'), **item))
            C['FP_candidate_' + fp[-1]['kind']] += 1

json.dump(dict(C), open(OUT + '/verifier_summary.json', 'w'), indent=1)
for name, L in [('fn', fn), ('fp', fp), ('misc', misc)]:
    with open(OUT + '/verifier_%s.jsonl' % name, 'w') as fo:
        for x in L: fo.write(json.dumps(x, ensure_ascii=False) + '\n')
print(json.dumps(dict(C), indent=1)); print('fn', len(fn), 'fp', len(fp), 'misc', len(misc))
R._kill_pool()
