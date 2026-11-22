"""用 E 的模型测 F 训练题的通过率：每题采样 k 次，温度 1.0、top_p 1.0（和 F 训练时一样），一张卡一个分片。
判分用 ma/reward_f.py（和训练、评测一致）。输出只存判分结果和长度，不存全文。
用法（由 f/run_profile_f.sh 调用）：python f/profile_f.py --model HF --out DIR --shard i --nshards 4 --k 4 --max-new 16384"""
import argparse
import json
import os
import sys

R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/ma')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=4)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--max-new', type=int, default=16384)
    ap.add_argument('--data', default=R + '/run/data/f_profile.parquet')
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    rows = pq.read_table(a.data).to_pylist()
    items = [(i, j) for i in range(len(rows)) for j in range(a.k)][a.shard::a.nshards]
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template(rows[i]['prompt'], tokenize=False, add_generation_prompt=True) for i, _ in items]
    import sglang as sgl
    eng = sgl.Engine(model_path=a.model, tp_size=1, mem_fraction_static=0.85, context_length=a.max_new + 2048,
                     random_seed=4321 + a.shard, log_level='warning')
    sp = {'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': a.max_new, 'stop_token_ids': [151645, 151643], 'skip_special_tokens': False}
    outs = eng.generate(prompt=prompts, sampling_params=sp)
    eng.shutdown()
    from reward_f import compute_score
    tmp = os.path.join(a.out, 'prof_%d.jsonl.tmp' % a.shard)
    with open(tmp, 'w') as fo:
        for (i, j), o in zip(items, outs):
            mi = o.get('meta_info', {}) or {}
            fr = mi.get('finish_reason')
            fr = fr.get('type') if isinstance(fr, dict) else fr
            r = rows[i]
            try:
                sc = compute_score(r['data_source'], o['text'], str(r['reward_model']['ground_truth']))
            except Exception as e:  # noqa: BLE001
                sc = {'acc': 0.0, 'src': 'error:' + type(e).__name__}
            fo.write(json.dumps(dict(i=i, j=j, acc=float(sc.get('acc', 0.0)), src=sc.get('src'), finish=fr,
                                     n_tok=mi.get('completion_tokens'), think_end='</think>' in o['text'])) + '\n')
    os.replace(tmp, os.path.join(a.out, 'prof_%d.jsonl' % a.shard))
    print('PROF_DONE', len(items), flush=True)


if __name__ == '__main__':
    main()
