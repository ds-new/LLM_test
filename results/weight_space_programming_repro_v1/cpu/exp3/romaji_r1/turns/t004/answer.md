```python
import numpy as np
from scipy.linalg import svd, eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)
d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
q=(W["W_q.0"][192:256].astype(float)*g)@P
k=(W["W_k.0"][192:256].astype(float)*g)@P
pos=svd(np.concatenate([q.T,k.T],axis=1),full_matrices=False)[0][:,:32]
vv=(W["W_v.0"].astype(float)*g)@P
ut,st,_=svd((vv@(np.eye(d)-pos@pos.T)).T,full_matrices=False)
T=ut[:,:27]
B=W["W_o.0"][:,192:256].astype(float)@vv[192:256]
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float))@P
X=F@T
Y=F@B@T

print("Candidate content singular values, first 45:",np.round(st[:45],4))
print("L0H3 OV restricted to candidate content, singular values:",
      np.round(svd(B@T,compute_uv=False),4))

def invsqrt(a):
    e,u=eigh(a)
    return (u*(np.maximum(e,1e-8)**-.5))@u.T

C=invsqrt(X.T@X)@(X.T@Y)@invsqrt(Y.T@Y)
e=np.linalg.eigvals(C)
e=e[np.argsort(-np.abs(e))]
print("\nAlgebraic current/shifted-feature cross-Gram:")
print("canonical singular values",np.round(svd(C,compute_uv=False),4))
print("eigenvalues",np.round(e,4))
print("symmetric-part eigenvalues",np.round(np.linalg.eigvalsh((C+C.T)/2),4))

datasets={"L0_FC_content":X}
u,s,vt=svd(W["W_proj.3"].astype(float),full_matrices=False)
datasets["L3_proj_output_coordinates"]=(s[:27,None]*vt[:27]).T
datasets["L3_proj_whitened_coordinates"]=vt[:27].T

for name,z in datasets.items():
    z=z-z.mean(0)
    z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
    print("\nCLUSTERING",name)
    for n in [16,20,22,24,26,28,32,40]:
        km=KMeans(n_clusters=n,n_init=8,random_state=123,max_iter=400).fit(z)
        centers=km.cluster_centers_
        centers/=np.linalg.norm(centers,axis=1,keepdims=True)
        c=np.sum(z*centers[km.labels_],axis=1)
        counts=np.bincount(km.labels_,minlength=n)
        sil=silhouette_score(z,km.labels_,sample_size=1000,random_state=123)
        print(n,"mean_cos=%.3f silhouette=%.3f"%(c.mean(),sil),
              "counts",sorted(counts.tolist(),reverse=True))
        if n==28:
            print("assigned cosine quantiles",np.round(np.quantile(c,[0,.1,.25,.5,.75,.9,1]),3))
            cc=centers@centers.T
            np.fill_diagonal(cc,-2)
            print("nearest-center cosines",np.round(np.sort(cc.max(1)),3))
```
