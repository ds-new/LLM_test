```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4,suppress=True,linewidth=150)

print("Final MLP output spectrum (W_proj.3), first 60")
u,s,vt=la.svd(np.asarray(W["W_proj.3"],dtype=np.float64),full_matrices=False)
print(s[:60])
print("Adjacent singular-value ratios, indices 10..39 (index is zero-based)")
print([(i,round(float(s[i]/s[i+1]),3)) for i in range(10,40)])

print("\nSpectra of first-layer character-processing candidates")
for name in ("W_fc.0","W_proj.0","W_v.0","W_o.0"):
    a=np.asarray(W[name],dtype=np.float64)
    sv=la.svdvals(a)
    print(name,sv[:50])

print("\nLargest-coordinate LayerNorm gains in layer 0 and associated column norms")
g=np.asarray(W["ln1_w.0"],dtype=np.float64)
qk=np.vstack([W["W_q.0"],W["W_k.0"]]).astype(float)
v=np.asarray(W["W_v.0"],dtype=float)
fc=np.asarray(W["W_fc.0"],dtype=float)
for i in np.argsort(-g)[:40]:
    print(i, "gain",round(g[i],4),"qk",round(la.norm(qk[:,i]),3),
          "v",round(la.norm(v[:,i]),3),"fc",round(la.norm(fc[:,i]),3))
print("Correlations of gain with qk/v/fc column norms:",
      [round(float(np.corrcoef(g,la.norm(a,axis=0))[0,1]),3) for a in (qk,v,fc)])

print("\nLayer-0 MLP neuron pair cosine distribution (input weights scaled by ln2_w)")
f=np.asarray(W["W_fc.0"],dtype=float)*W["ln2_w.0"]
f-=f.mean(axis=1,keepdims=True)
f/=la.norm(f,axis=1,keepdims=True)
c=f@f.T
np.fill_diagonal(c,-np.inf)
print("nearest-neighbor cosine quantiles:",np.quantile(c.max(axis=1),[0,.1,.25,.5,.75,.9,.99,1]))
print("Most mutually similar pairs:")
pairs=np.dstack(np.unravel_index(np.argsort(c.ravel())[-30:][::-1],c.shape))[0]
seen=set()
for i,j in pairs:
    if (min(i,j),max(i,j)) in seen: continue
    seen.add((min(i,j),max(i,j)))
    print(int(i),int(j),round(c[i,j],4))

print("\nOutput-subspace concentration for each layer's W_proj and W_o")
for r in (20,24,27,28):
    basis=u[:,:r]
    print("rank",r,[(name,round(float(la.norm(basis.T@W[name])**2/la.norm(W[name])**2),3))
                   for name in ("W_proj.0","W_proj.1","W_proj.2","W_proj.3","W_o.0","W_o.1","W_o.2","W_o.3")])
```
