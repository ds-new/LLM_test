import json
from pathlib import Path

import numpy as np

D = Path('/root/autodl-tmp/a800_mech')


def J(n):
    p = D / f'{n}.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


s, le, la, ind, pr, tr, de, qu, hd = (J(n) for n in ('sinks', 'lens', 'layers', 'induction', 'probes', 'trace', 'decode', 'quant', 'heads'))
f2 = lambda x: f'{x:.2f}'
f3 = lambda x: f'{x:.3f}'
sg = lambda x: f'{x:+.3f}'
out = []
w = out.append

first = np.array(s['attention']['first'])
mass = s['massive']
win = s['window']
w('# v1（0.48B，40 层）机理分析与推理侧实验（2026-11-18）')
w('')
w('对象是 v1 的 final.pt：隐藏维 1024、40 层、16 个查询头共用 4 组 KV（GQA）、SwiGLU、RoPE、输入输出共享词嵌入，训练了 25B token。'
  '数据用 v1 自己的验证集（7 个来源，取法和训练时的验证口径一致）。全部只做前向，和路线 A 的训练共用一张卡。'
  '代码在 decoder_from_scratch/a800/mech.py，每项结果的 JSON 和图都在本目录。')
w('')
w('基准：bf16 下按评测权重加权的验证 loss 为 ' + f3(qu['main']['bf16']['weighted']) + '，与 v1 训练时报告的 2.2879 一致，说明这套分析代码的模型和数据都对得上。')
w('')

w('## 1. Attention sink 与超大激活')
w('')
w('![](figs/sinks_attention.png)')
w('![](figs/massive_activations.png)')
w('')
w('- 第 0–8 层几乎不看第一个 token（每层平均注意力不超过 ' + f2(float(first[:9].mean(1).max())) + '）；从第 9 层起，平均每个头有 '
  + f2(float(first[9:39].mean(1).min())) + '–' + f2(float(first[9:39].mean(1).max())) + ' 的注意力落在第一个 token 上，一直持续到第 38 层，最后一层又降到 ' + f2(float(first[39].mean())) + '。')
m9, m38, m40 = mass[9], mass[38], mass[40]
w(f'- 同一时刻残差流里出现超大激活：第 9 个状态（第 8 层的输出）起，第 {m9["dims"][0][0]} 维出现约 {m9["top_mean"]:.0f} 的值，'
  f'而同层所有元素绝对值的中位数只有 {m9["median_mean"]:.2f}。它落在序列第一个 token 和 BOS/EOS 这类分隔符上（{m9["at_pos0"]:.0%} 的序列落在第一个位置），'
  f'原样保持到第 38 个状态（{m38["top_mean"]:.0f}），最后两层把它抵消到约 {m40["top_mean"]:.0f}。')
dn = qu['activation_outlier_ratio']['ffn.down_proj']
w(f'- 来源：第 8 层 MLP 的 down_proj 输入里，最大值是中位数的 {dn[8]:,.0f} 倍（各层这一比值的中位数是 {np.median(dn):.0f}）。第 3 节里，去掉第 8 层的 MLP 会让 loss 上升 {la["mlp"][8]["weighted"] - la["base"]["weighted"]:.2f}。')
w('- 去掉开头 token 会怎样（只算第 512–1023 个位置的 loss，按评测权重加权）：')
w('')
w('| 注意力范围 | loss | 相比完整注意力 |')
w('|---|---|---|')
names = {'full': '完整的因果注意力', 'window256': '只看最近 256 个 token', 'window256+first4': '最近 256 个 + 开头 4 个', 'mask_first': '完整注意力但屏蔽第一个 token'}
for k, v in win.items():
    w(f'| {names[k]} | {f3(v["weighted"])} | {sg(v["weighted"] - win["full"]["weighted"])} |')
w('')
w('解读：sink 是给"这一步不需要从上下文取信息"的头准备的停靠点。超大激活和 sink 在同一层出现，它很可能就是让第一个 token 的 key 对几乎所有 query 都得高分的原因。'
  '滑动窗口一旦把开头 token 挤出去，这些头就没处停，只能把注意力摊到无关的 token 上，loss 大涨；把开头 4 个 token 留在窗口里就基本恢复（StreamingLLM 的做法）。'
  '直接屏蔽第一个 token 几乎没影响，很可能是第二个 token 接替了 sink：屏蔽后它也只能看到自己，处境和原来的第一个 token 一样。')
w('')

lw = le['weighted']
w('## 2. Logit lens：每一层"已经想好"了多少')
w('')
w('![](figs/logit_lens.png)')
w('')
w('把每一层的残差流直接接最终的 RMSNorm 和输出层解码：')
w('')
w('| 层 | 0 | 10 | 20 | 30 | 35 | 38 | 39 | 40（输出） |')
w('|---|---|---|---|---|---|---|---|---|')
w('| 交叉熵 | ' + ' | '.join(f2(lw['loss'][i]) for i in (0, 10, 20, 30, 35, 38, 39, 40)) + ' |')
w('| 与最终 top-1 一致率 | ' + ' | '.join(f2(lw['agree'][i]) for i in (0, 10, 20, 30, 35, 38, 39, 40)) + ' |')
w('')
half = {n: next((i for i, x in enumerate(v['agree']) if x >= 0.5), None) for n, v in le['by_source'].items()}
w('- 第 0 层解码出来的就是输入 token 本身（输入输出共享词嵌入），所以 loss 极高；之后 loss 平稳下降，没有哪一层突然成形；最后一层单独又把 loss 从 '
  + f2(lw['loss'][39]) + ' 降到 ' + f2(lw['loss'][40]) + '。')
w('- 一致率过半的层：' + '、'.join(f'{n} 第 {k} 层' for n, k in half.items()) + '。代码和数学最早成形，中文最晚。')
w('- 局限：前面的层表示不一定在输出层的坐标系里，logit lens 会低估它们，训练一组逐层的线性"翻译器"（tuned lens）会更准。')
w('')

b = la['base']['weighted']
d_all = [r['weighted'] - b for r in la['all']]
w('## 3. 逐层消融：每一层有多重要')
w('')
w('![](figs/layer_ablation.png)')
w('')
w(f'基准 loss {f3(b)}（每个来源 8 块）。去掉整层 / 只去注意力 / 只去 MLP 后 loss 的增量：')
w('')
w('| 层 | 整层 | 只去注意力 | 只去 MLP |')
w('|---|---|---|---|')
shown = sorted(range(len(d_all)), key=lambda i: -d_all[i])[:6]
for i in shown:
    w(f'| {i} | {sg(d_all[i])} | {sg(la["attn"][i]["weighted"] - b)} | {sg(la["mlp"][i]["weighted"] - b)} |')
mid = [d_all[i] for i in range(len(d_all)) if i not in shown]
w(f'| 其余 {len(mid)} 层 | {min(mid):+.3f} 到 {max(mid):+.3f} | | |')
w('')
w('- 第 0 层的 MLP 不可缺：它把词嵌入加工成后面各层能用的表示，相当于"扩展的词嵌入"。')
w('- 第 8 层的 MLP 正是写入超大激活的那一层（down_proj 输入的离群值也在这里），只去掉它 loss 就涨 ' + f2(la['mlp'][8]['weighted'] - b) + '，比相邻各层大一个数量级。')
w('- 累计剪层（按单层增量从小到大一起去掉）：' + '，'.join(f'{p["k"]} 层 {sg(p["weighted"] - b)}' for p in la['prune']) + '。这个模型没有可以白送的层。')
w('')

if hd:
    dh = np.array(hd['delta'])
    w('## 4. 逐头消融：每个注意力头有多重要')
    w('')
    w('![](figs/head_ablation.png)')
    w('')
    w(f'每次把一个头的输出置零（每个来源 2 块），基准 loss {f3(hd["base"])}。640 个头里，增量超过 0.01 的有 {int((dh > 0.01).sum())} 个，超过 0.005 的有 {int((dh > 0.005).sum())} 个，'
      f'{int((dh < 0.001).sum())} 个不到 0.001。最重要的 10 个：')
    w('')
    w('| 层 | 头 | 增量 | induction 分数 | 前一 token 分数 | 落在首 token 的注意力 |')
    w('|---|---|---|---|---|---|')
    isc, psc = np.array(ind['induction_score']), np.array(ind['prev_score'])
    for t in hd['top'][:10]:
        li, h = t['layer'], t['head']
        w(f'| {li} | {h} | {sg(t["delta"])} | {f2(isc[li, h])} | {f2(psc[li, h])} | {f2(first[li, h])} |')
    top_ind = [(x['layer'], x['head']) for x in ind['top_induction'][:8]]
    rank = {(j // dh.shape[1], j % dh.shape[1]): r for r, j in enumerate(np.argsort(-dh.flatten()))}
    w('')
    w('- induction 分数最高的 8 个头在这张重要性排名里的位置：' + '、'.join(f'L{a}H{c} 第 {rank[(a, c)] + 1}' for a, c in top_ind) + '（共 640 个）。')
    circuit = {(7, 6), (8, 10), (8, 4), (10, 5), (10, 4), (13, 4), (16, 1), (16, 3)}
    n_c = sum((t['layer'], t['head']) in circuit for t in hd['top'][:6])
    w(f'- 单个头几乎都可有可无，这和第 3 节里一整层（16 个头加 MLP）+0.02–0.05 的代价对照来看，说明头之间有大量冗余。排在最前面的 6 个头里有 {n_c} 个属于第 5 节的照抄电路：前一 token 头 L7H6、L8H10、L8H4 和 induction head L10H5。')
    corr = np.corrcoef(first.flatten(), dh.flatten())[0, 1]
    w(f'- 头的重要性和"落在首 token 的注意力"几乎无关（相关系数 {corr:.2f}）：L10H5 有 {first[10, 5]:.0%} 的注意力停在 sink 上，仍然排第 2。停在 sink 上只说明它在多数位置不需要取信息，不说明它没用。')
    w('')

br = ind['base_rep']
w('## 5. Induction head 与上下文学习')
w('')
w('![](figs/induction.png)')
w('')
w(f'- 把一段 256 个随机 token 的序列重复一遍：第一遍的 loss 是 {f2(br["first_copy"])}（比均匀乱猜的 10.37 还高），第二遍降到 {f2(br["second_copy"])}，模型几乎能逐字照抄。')
w('- induction 分数（第二遍里，每个位置把注意力放在"它上一次出现之后的那个 token"上的比例）最高的头：'
  + '、'.join(f'L{x["layer"]}H{x["head"]} {f2(x["score"])}' for x in ind['top_induction'][:8]) + '，集中在第 9–16 层。')
w('- 前一 token 头（注意力放在紧挨着的前一个 token 上）最高的是：' + '、'.join(f'L{x["layer"]}H{x["head"]} {f2(x["score"])}' for x in ind['top_prev'][:5])
  + '，主要在第 7–8 层，正好在 induction head 之前：前一个头先把"我前面是什么"写进每个位置，induction head 再拿当前 token 去找"前面是它"的位置，复制它后面的 token。')
w('- 高分的 induction head 经常成组出现在同一组 KV 里（如 L10H4/H5、L13H4/H5、L16H1/H2/H3）。这符合 GQA 的结构：同组的头共用 key，"前一个 token 是什么"只要写进这组 key，组内几个头都能用。')
w('')
w('| 去掉的头 | 第二遍 loss | 验证 loss | 上下文学习分数 |')
w('|---|---|---|---|')
w(f'| 不去掉 | {f2(br["second_copy"])} | {f3(ind["base_val"])} | {f3(ind["base_icl"])} |')
for a in ind['ablate']:
    label = f'induction 分数最高的 {a["k"]} 个' if a['kind'] == 'induction' else f'随机 {a["k"]} 个（3 次平均）'
    w(f'| {label} | {f2(a["second_copy"])} | {f3(a["val"])} | {f3(a["icl"])} |')
w('')
w('上下文学习分数 = 文档里第 480–520 个位置的平均 loss 减去第 40–60 个位置的，越负说明后文从前文得到的帮助越大。')
w('只去掉 2 个 induction head，第二遍就从 0.37 回到 8.76，几乎完全不会照抄；随机去掉同样多的头没有影响。在正常文本上，去掉 16 个 induction head 后，长距离带来的收益从 '
  + f3(-ind['base_icl']) + ' 缩小到 ' + f3(-[a for a in ind['ablate'] if a['kind'] == 'induction' and a['k'] == 16][0]['icl']) + '，说明上下文学习有相当一部分就是这套"查找并复制"的电路。')
w('')

w('## 6. 线性探针：残差流里能读出什么')
w('')
w('![](figs/probes.png)')
w('')
sa, pr2 = pr['source_acc'], pr['position_r2']
w(f'- 数据来源（7 类，随机猜是 {pr["source_chance"]:.2f}）：第 0 层 {f2(sa[0])}，第 2 层 {f2(sa[2])}，最后一层 {f2(sa[-1])}。区分三类中文网页要靠上下文，越往后越准。')
k = int(np.argmax(pr2))
w(f'- 绝对位置（log 位置的 R²）：词嵌入层 {f2(pr2[0])}（RoPE 本身只给相对位置），第 {k} 层最高 {f2(pr2[k])}，之后逐层下降到最后一层 {f2(pr2[-1])}。'
  '模型自己推出了"我在第几个位置"（很可能借助 sink：落在首 token 上的注意力随位置变化），用完又逐渐丢掉。')
ar = pr['arith']
w(f'- 两位数加法（三个示例之后问 "a+b="）：模型只答对 {pr["arith_model_acc"]:.1%}。但在 "=" 处，进位从第 1 层起就几乎完全可读（{f2(max(ar["carry"]))}），'
  f'和的十位最高 {f2(max(ar["tens"]))}，个位最高 {f2(max(ar["units"]))}（随机都是 0.10）。中间量一部分已经算出来了，但没有可靠地用到输出上。'
  'v1 基座的训练数据里没有合成算术题；tokenizer 对照里加了 3% 合成算术后，用 v1 分词器的 1.25 亿参数模型两位数加法也是 100%，所以这里主要缺的是数据；逐位切分数字（v2）的优势出现在更难的题上（四位数加法 82.5% → 100%，两位乘两位 48% → 79%）。')
w('')

ie, nf = tr['ie_norm'], tr['n_facts']
fp = tr['facts']
w('## 7. 知识存在哪：因果追踪')
w('')
w('![](figs/causal_trace.png)')
w('')
w(f'- 用了 {nf} 条模型答对的事实（中英文的首都、作者、地标、人物等）。给主语的词嵌入加噪声（噪声标准差是词嵌入标准差的 3 倍）后，答案概率从平均 '
  f'{np.mean([x["p_clean"] for x in fp]):.2f} 降到 {np.mean([x["p_corrupt"] for x in fp]):.3f}；再逐层、逐位置把干净的激活恢复回去，看概率回升多少（下面都按"干净 − 加噪"归一化）。')
r_sl, r_last = np.array(ie['resid']['subj_last']), np.array(ie['resid']['last'])
m_sf, m_sl = np.array(ie['mlp']['subj_first']), np.array(ie['mlp']['subj_last'])
a_last = np.array(ie['attn']['last'])
w(f'- 残差流：在主语的最后一个 token 上恢复，第 0–16 层都能找回 {r_sl[:17].min():.0%}–{r_sl[:17].max():.0%}（第 {int(r_sl.argmax())} 层最高），第 28 层后降到 {r_sl[28]:.0%} 以下；'
  f'在最后一个 token 上恢复，第 24 层起才开始有用（第 28 层 {r_last[28]:.0%}，第 32 层 {r_last[32]:.0%}）。信息在第 24–32 层从主语位置搬到了最后位置。')
w(f'- MLP（一次恢复相邻 5 层）：主语 token 上越靠前作用越大，第 0–2 层约 {m_sl[0]:.0%}，第 8 层 {m_sl[8]:.0%}，第 16 层 {m_sl[16]:.0%}。')
w(f'- 注意力（一次恢复相邻 5 层）：在最后一个 token 上第 {int(a_last.argmax())} 层附近最大（{a_last.max():.0%}），正是"搬运"发生的地方；在主语位置上几乎不起作用。')
w('- 和 ROME 的图景一致：前中段的 MLP 在主语位置上把事实"读出来"，后段的注意力把它搬到要预测的位置。不同的是这里起作用的 MLP 更靠前（第 0–16 层）。')
w('')

st = de['strategies']
hu = st['human_reference']
w('## 8. 解码策略与复读')
w('')
w(f'{de["n_prompts"]} 个文档开头（7 个来源各约 20 个），给前 {de["prompt_tokens"]} 个 token，续写 {de["new_tokens"]} 个。困惑度一栏是模型自己给续写文本的平均负对数似然；人类原文的值是 {f2(hu["nll"])}，越接近它越像正常文本。')
w('')
w('| 解码方式 | 4-gram 重复率 | 不同二元组占比 | 负对数似然 | 主动结束的比例 |')
w('|---|---|---|---|---|')
w(f'| 人类原文 | {f3(hu["rep4"])} | | {f2(hu["nll"])} | |')
for k2, v in st.items():
    if k2 != 'human_reference':
        w(f'| {k2} | {f3(v["rep4"])} | {f2(v["distinct2"])} | {f2(v["nll"])} | {v["eos_rate"]:.0%} |')
w('')
rf = de['reinforce']
w('- 贪心解码选的总是最可能的路：负对数似然 ' + f2(st['greedy']['nll']) + '，远低于人类原文的 ' + f2(hu['nll']) + '，4-gram 重复率是人类的 '
  + f'{st["greedy"]["rep4"] / max(1e-9, hu["rep4"]):.0f} 倍；例如 zh_cci3 的一条续写把提示里的整句又抄了一遍。')
w('- 自我强化：把同一个 24-token 的片段重复 6 遍，模型给第 1–6 遍的平均概率是 ' + ' / '.join(f2(r['mean_prob']) for r in rf)
  + '，从第 2 遍起有 ' + f'{rf[1]["argmax_hit"]:.0%}' + ' 的 token 正好是概率最高的那个。一旦出现重复，第 5 节的照抄电路就会让它越来越确定，贪心解码于是锁进循环。')
w('- 重复惩罚 1.2 把重复率压到人类水平，代价是负对数似然升到 ' + f2(st['greedy+rep_penalty1.2']['nll']) + '；top-p 0.9 的负对数似然（'
  + f2(st['top_p0.9']['nll']) + '）最接近人类，重复率也接近；纯采样（温度 1）的负对数似然 ' + f2(st['sample_t1.0']['nll']) + ' 超过人类原文，说明长尾里的 token 被抽得太多。')
w('')

mq = qu['main']
bb = mq['bf16']['weighted']
w('## 9. 量化')
w('')
w('![](figs/quant.png)')
w('')
w('| 方案 | 加权验证 loss | 增量 |')
w('|---|---|---|')
for k2, v in mq.items():
    w(f'| {k2} | {f3(v["weighted"])} | {sg(v["weighted"] - bb)} |')
w('')
w('权重量化都是对称取整；per-channel 是每个输出通道一个缩放，g128 是每 128 个输入权重一组。激活量化是把每个线性层的输入取整到 int8，per-tensor 整个张量一个缩放，per-token 每个 token 一个。')
w('')
pk = qu['per_kind_w3g128']
pl = qu['per_layer_w3g128']
w4 = mq['W4 g128']
zh = np.mean([w4[n] - mq['bf16'][n] for n in ('zh_cci3', 'zh_fwedu', 'zh_industry')])
ot = np.mean([w4[n] - mq['bf16'][n] for n in ('en_fwedu', 'en_cosmo', 'code', 'math')])
w(f'- int8 权重基本无损；int4 分组 128 只 +{w4["weighted"] - bb:.3f}；int3 开始明显变差；int2 直接失效。中文比其他来源更敏感（int4 分组 128：中文平均 +{zh:.3f}，英文、代码、数学平均 +{ot:.3f}）。')
w('- 只把某一类矩阵量化到 int3（分组 128）：' + '，'.join(f'{k2.split(".")[-1]} {sg(v)}' for k2, v in pk.items())
  + '。v_proj 最敏感：GQA 下每组 KV 被 4 个查询头共用，它又只有 256 个输出通道，误差被放大。')
top3 = sorted(range(len(pl)), key=lambda i: -pl[i])[:3]
w('- 只把某一层量化到 int3：最敏感的是第 ' + '、'.join(f'{i} 层（{sg(pl[i])}）' for i in top3) + '，其余多在 +0.001–0.008。第 8 层又是写入超大激活的那一层。'
  f'单层增量加起来是 {sum(pl):.2f}，比全部一起量化的 {mq["W3 g128"]["weighted"] - bb:.2f} 小，误差会叠加放大。')
w(f'- 激活：按张量量化到 int8 +{mq["A8 per-tensor"]["weighted"] - bb:.3f}，按 token +{mq["A8 per-token"]["weighted"] - bb:.3f}。'
  '离群值让整个张量的缩放被一个元素撑大，其余元素都被量化成 0 附近；按 token 缩放就把影响限制在那几个 token 上。这是 LLM.int8()、SmoothQuant 一类方法要解决的问题。')
w('')

w('## 和下一步的衔接')
w('')
w('- 超大激活和 sink 都在第 8–9 层"打开"：第 1 部分的结构消融里做 QK-norm、logit soft-cap 时，可以顺带看超大激活和 sink 还在不在、量化敏感度会不会变小。')
w('- 加法只有 3.6% 而探针能读出部分结果：可以在逐位数字分词的模型上重做这组探针，看中间量能不能被用到输出上。')
w('- induction head 什么时候出现：需要一次从头训练、早期密集记录 induction 分数的短训练（约 1–2 小时），排在实验室里。')
w('- 复读：SFT 之后再测一次第 8 节，看后训练能不能消掉贪心解码的循环。')
(D / 'REPORT.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('written', len('\n'.join(out)))
