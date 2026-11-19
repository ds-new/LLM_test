"""train_dapo_r3.parquet：与 train_dapo.parquet 行数、顺序相同，只把 6 道泄露题换成别的题的副本（固定随机种子）。"""
import numpy as np, pandas as pd
D = 'run/data/'
tr = pd.read_parquet(D + 'train_dapo.parquet'); dd = pd.read_parquet(D + 'train_dapo_dedup.parquet')
q = tr.extra_info.map(lambda e: e['question']); keep = set(dd.extra_info.map(lambda e: e['question']))
leak = np.where(~q.isin(keep).values)[0]
assert len(leak) == len(tr) - len(dd) == 6, (len(leak), len(tr), len(dd))
rng = np.random.default_rng(0)
donors = rng.choice(np.setdiff1d(np.arange(len(tr)), leak), size=len(leak), replace=False)
idx = np.arange(len(tr)); idx[leak] = donors
out = tr.iloc[idx].reset_index(drop=True)
out.to_parquet(D + 'train_dapo_r3.parquet')
q2 = out.extra_info.map(lambda e: e['question'])
print('rows', len(out), 'leak positions', leak.tolist(), 'donors', donors.tolist(), 'leaked left', int((~q2.isin(keep)).sum()),
      'same elsewhere', bool((q2.values[np.setdiff1d(np.arange(len(tr)), leak)] == q.values[np.setdiff1d(np.arange(len(tr)), leak)]).all()))
