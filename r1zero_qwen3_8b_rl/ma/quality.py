"""验证生成的质量检查：中英混杂、重复、没写最终答案、长度、正确率。用法：python ma/quality.py 96 116 120 124"""
import json, re, sys, collections
import numpy as np
cjk = re.compile(r'[一-鿿]')
for s in sys.argv[1:]:
    rows = [json.loads(l) for l in open('/root/r1zero_valgen/%s.jsonl' % s)]
    o = [r['output'] for r in rows]
    acc = np.mean([float(r.get('acc', 0)) > 0.5 for r in rows])
    mix = np.mean([bool(cjk.search(x)) for x in o])
    def rep(x):
        lines = [l.strip() for l in x.split('\n') if len(l.strip()) > 20]
        c = collections.Counter(lines)
        return bool(c) and c.most_common(1)[0][1] >= 5
    reps = np.mean([rep(x) for x in o])
    noans = np.mean([not re.search(r'Answer\s*:', x[-800:]) for x in o])
    words = [len(x) for x in o]
    nonascii = np.mean([sum(ord(ch) > 127 for ch in x) / max(1, len(x)) for x in o])
    wait = np.mean([len(re.findall(r'\b(wait|Wait|hmm|Hmm|let me check|Let me verify|double-check)\b', x)) for x in o])
    print('step %s  acc %.3f  中英混杂 %.1f%%  重复行 %.1f%%  结尾没写Answer %.1f%%  平均字符 %d  非ASCII字符占比 %.2f%%  反思词/条 %.1f' % (s, acc, 100 * mix, 100 * reps, 100 * noans, np.mean(words), 100 * nonascii, wait))
