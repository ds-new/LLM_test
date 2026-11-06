```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)
d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)

print("Raw first-layer per-head value singular values:")
for j in range(6):
    s=svd(W["W_v.0"][64*j:64*(j+1)].astype(float),compute_uv=False)
    print("L0H%d"%j,np.round(s[:40],3))

print("\nRaw L0H3 OV singular values:")
ov3=W["W_o.0"][:,192:256].astype(float)@W["W_v.0"][192:256].astype(float)
print(np.round(svd(ov3,compute_uv=False)[:45],3))

rawpos=svd(np.concatenate([W["W_q.0"][192:256].T,
                          W["W_k.0"][192:256].T],axis=1).astype(float),
           full_matrices=False)[0][:,:32]
v3=W["W_v.0"][192:256].astype(float)
_,sv,tv=svd(v3@(np.eye(d)-rawpos@rawpos.T),full_matrices=False)
print("\nRaw positional-subspace-removed L0H3 V spectrum:",np.round(sv[:40],3))

print("\nCosines of raw first-layer OV matrices:")
ovs=[]
for j in range(6):
    ovs.append(W["W_o.0"][:,j*64:(j+1)*64].astype(float)@
               W["W_v.0"][j*64:(j+1)*64].astype(float))
print(np.round([[np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y)) for y in ovs] for x in ovs],3))

F=(W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float))@P
for rank in [20,23,26]:
    # Map weight-derived directions from the first normalization's coordinate
    # system back to residual coordinates; no layer is evaluated.
    T=tv[:rank].T/np.maximum(g[:,None],.08)
    T=np.linalg.qr(P@T)[0]
    z=F@T
    z-=z.mean(0)
    z/=np.linalg.norm(z,axis=1,keepdims=True)+1e-12
    km=KMeans(n_clusters=28,n_init=20,random_state=123,max_iter=500).fit(z)
    cen=km.cluster_centers_
    cen/=np.linalg.norm(cen,axis=1,keepdims=True)
    cc=cen@cen.T
    np.fill_diagonal(cc,-2)
    count=np.bincount(km.labels_)
    order=np.argsort(cc.max(1))
    print("\nDebiased content rank",rank)
    print("nearest-center cosines",np.round(cc.max(1)[order],3))
    print("corresponding counts",count[order])
    print("median assigned cosine",np.median(np.sum(z*cen[km.labels_],axis=1)))

print("\nMLP first input singular values, first 50:")
print(np.round(svd(W["W_fc.0"].astype(float),compute_uv=False)[:50],3))
```
