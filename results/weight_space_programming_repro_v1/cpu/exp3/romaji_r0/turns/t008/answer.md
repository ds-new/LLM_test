```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

def eff(a,g):
    a=a.astype(float)*g.astype(float)
    return a-a.mean(1,keepdims=True)
F=eff(W["W_fc.0"],W["ln2_w.0"])
q=eff(W["W_q.0"],W["ln1_w.0"])
k=eff(W["W_k.0"],W["ln1_w.0"])
pos=svd(np.concatenate([q[192:256],k[192:256]],axis=0),full_matrices=False)[2][:40].T
otherO=np.concatenate([W["W_o.0"][:,64*j:64*j+64] for j in [1,2,4,5]],axis=1).astype(float)
other=svd(otherO,full_matrices=False)[0][:,:6]

for r in [28,48,64]:
    o3=svd(W["W_o.0"][:,192:256].astype(float),full_matrices=False)[0][:,:r]
    o0=svd(W["W_o.0"][:,:64].astype(float),full_matrices=False)[0][:,:min(r,32)]
    N=np.concatenate([o3,o0,other,pos,np.ones((384,1))],axis=1)
    U,_,_=svd(N,full_matrices=False)
    residual=F-(F@U)@U.T
    u,s,vt=svd(residual,full_matrices=False)
    print("\nRemoved output rank",r,"residual F spectrum",s[:65])
    if r==64:
        Z=u[:,:28]*np.sqrt(s[:28])
        Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-10)
        for nc in [20,24,28,32]:
            km=KMeans(n_clusters=nc,n_init=20,random_state=19).fit(Z)
            print("clusters",nc,"silhouette",round(silhouette_score(Z,km.labels_,sample_size=1200,random_state=5),3),
                  "sizes",np.sort(np.bincount(km.labels_))[::-1])
        ica=FastICA(n_components=28,whiten="unit-variance",max_iter=2000,tol=1e-4,random_state=9)
        C=ica.fit_transform(Z)
        ix=np.argsort(-np.abs(kurtosis(C,axis=0)))
        print("normalized current-weight ICA kurtoses",kurtosis(C,axis=0)[ix])
        print("ICA skewnesses",skew(C,axis=0)[ix])
```
