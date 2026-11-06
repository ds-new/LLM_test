```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

def ctr(a): return a-a.mean(-1,keepdims=True)
V=W["W_v.0"][128:192].astype(float)
Q=W["W_q.0"][128:192].astype(float)
K=W["W_k.0"][128:192].astype(float)
print("Raw L0H2 V singular values:\n",np.linalg.svd(V,compute_uv=False))
print("Raw L0H2 Q singular values:\n",np.linalg.svd(Q,compute_uv=False))
print("Raw L0H2 K singular values:\n",np.linalg.svd(K,compute_uv=False))

# Cluster MLP input weight directions after projection onto weight-derived
# value-reading subspaces. These are not token vectors or activations.
fc=ctr(W["W_fc.0"].astype(float)*W["ln2_w.0"])
g=W["ln1_w.0"].astype(float)
for kind in ["raw_V", "scaled_V", "OV_read"]:
    if kind=="raw_V":
        _,s,vh=svd(V,full_matrices=False)
        basis=ctr(vh[:28]*g)
        basis=np.linalg.qr(basis.T)[0]
    elif kind=="scaled_V":
        _,s,vh=svd(ctr(V*g),full_matrices=False)
        basis=vh[:28].T
    else:
        ov=W["W_o.0"][:,128:192].astype(float)@ctr(V*g)
        _,s,vh=svd(ov,full_matrices=False)
        basis=vh[:28].T
    z=fc@basis
    norms=np.linalg.norm(z,axis=1)
    z=z/np.maximum(norms[:,None],1e-12)
    km=KMeans(n_clusters=28,n_init=20,random_state=0).fit(z)
    counts=np.bincount(km.labels_,minlength=28)
    coherence=np.linalg.norm(km.cluster_centers_,axis=1)
    ix=np.argsort(-counts)
    print("\n",kind,"mean projected norm fraction=%.3f"%
          np.mean(norms/np.linalg.norm(fc,axis=1)))
    print("silhouette=%.4f"%silhouette_score(z,km.labels_))
    print("cluster sizes:",counts[ix])
    print("cluster mean-direction norms:",coherence[ix])
    print("cluster size entropy=%.4f"%
          (-np.sum((counts/counts.sum())*np.log(counts/counts.sum()))))
```
