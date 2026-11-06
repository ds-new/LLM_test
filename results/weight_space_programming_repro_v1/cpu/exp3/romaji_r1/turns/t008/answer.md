```python
import numpy as np
import torch
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)
torch.set_num_threads(3)

# A purely parameter-level check: does recognizable initialization noise
# remain in the least-trained query head? No network inputs or layers.
seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,17,21,23,27,31,42,43,64,69,
       99,100,101,123,124,256,314,420,512,666,777,999,1024,1234,1337,
       1729,2023,2024,2025,2026,2718,3141,3407,4096,12345,31415,54321]
target=W["W_q.0"][320:324].astype(np.float64)
target/=np.linalg.norm(target,axis=1,keepdims=True)
nrows=32000
records=[]
for seed in seeds:
    gen=torch.Generator(device="cpu").manual_seed(seed)
    a=torch.randn((nrows,384),generator=gen,dtype=torch.float32).numpy()
    dots=a@target[0].astype(np.float32)
    norms=np.sqrt(np.einsum("ij,ij->i",a,a))
    scores=dots/norms
    inds=np.argsort(-scores)[:2]
    for i in inds:
        if i+4<=nrows:
            consecutive=np.sum(a[i:i+4]*target,axis=1)/norms[i:i+4]
            records.append((float(scores[i]),seed,int(i),consecutive.tolist()))
records.sort(reverse=True)
print("Best initialization-noise matches: cosine, seed, row offset, four consecutive-row cosines")
for row in records[:12]:
    print(row[0],row[1],row[2],np.round(row[3],4))
```
