```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)

# Additional alphabet-size evidence from independent components of WEIGHTS.
# Components are unlabeled; none is assumed to be a particular character.
d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
pos=svd(np.concatenate([W["W_q.0"][192:256].T,
                       W["W_k.0"][192:256].T],axis=1).astype(float),
        full_matrices=False)[0][:,:32]
v=W["W_v.0"][192:256].astype(float)
_,s,vt=svd(v@(np.eye(d)-pos@pos.T),full_matrices=False)
T=np.linalg.qr(P@(vt[:32].T/np.maximum(g[:,None],.08)))[0]
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float))@P
u,sp,vp=svd(W["W_proj.3"].astype(float),full_matrices=False)

for name,z in [("first MLP input, candidate content coordinates",F@T),
               ("last MLP output, principal coordinates",(sp[:32,None]*vp[:32]).T)]:
    for r in [24,28,32]:
        ica=FastICA(n_components=r,whiten="unit-variance",random_state=17,
                    max_iter=4000,tol=1e-4)
        sources=ica.fit_transform(z)
        norms=np.linalg.norm(ica.mixing_,axis=0)
        sk=np.abs(skew(sources,axis=0))
        ku=kurtosis(sources,axis=0)
        order=np.argsort(-norms)
        print("\n",name,"ICA rank",r)
        print("component: mixing norm, abs skewness, excess kurtosis")
        print(np.round(np.c_[norms,sk,ku][order],3).tolist())
```
