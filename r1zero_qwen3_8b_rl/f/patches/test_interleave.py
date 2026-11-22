import torch, numpy as np
from verl import DataProto
P, NS, W = 48, 16, 4
ids = torch.arange(P).unsqueeze(1).repeat(1, 3)
gen = DataProto.from_dict(tensors={'input_ids': ids}, non_tensors={'pid': np.arange(P).astype(object)})
gen = gen.repeat(repeat_times=NS, interleave=True)
N = len(gen)
ref = gen.non_tensor_batch['pid'].copy()
perm = torch.arange(N).view(-1, W).t().reshape(-1)
gen.reorder(perm)
chunks = gen.chunk(W)
for k, c in enumerate(chunks):
    pids = list(c.non_tensor_batch['pid'])
    cnt = {p: pids.count(p) for p in set(pids)}
    assert len(cnt) == P and all(v == NS // W for v in cnt.values()), (k, cnt)
outs = []
for c in chunks:
    resp = c.batch['input_ids'][:, :1] * 10
    outs.append(DataProto.from_dict(tensors={'input_ids': c.batch['input_ids'], 'responses': resp}, non_tensors={'pid': c.non_tensor_batch['pid']}))
out = DataProto.concat(outs)
out.reorder(torch.argsort(perm))
want = torch.tensor([int(x) for x in ref], dtype=torch.long)
assert (out.non_tensor_batch['pid'] == ref).all()
assert torch.equal(out.batch['input_ids'][:, 0], want)
assert torch.equal(out.batch['responses'][:, 0], want * 10)
print('interleave test OK:', W, 'chunks x', NS // W, 'samples of each of', P, 'prompts; order restored over', N, 'samples')
