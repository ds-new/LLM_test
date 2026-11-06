```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

print("Raw L0 head value spectra:")
for j in [0,1,2,3,4,5]:
    a=W["W_v.0"][64*j:64*j+64].astype(float)
    print(j,svd(a,compute_uv=False))

# Independent-component analysis of weight rows/columns only.
print("\nICA of final-MLP write vectors (columns of W_proj.3):")
Z=W["W_proj.3"].astype(float).T
ica=FastICA(n_components=27,whiten="unit-variance",random_state=11,max_iter=2500,tol=1e-5)
C=ica.fit_transform(Z)
kt=kurtosis(C,axis=0,fisher=True)
order=np.argsort(-kt)
print("sorted excess kurtoses",kt[order])
print("corresponding mixing-column norms",np.linalg.norm(ica.mixing_,axis=0)[order])
for k in order:
    col=C[:,k]
    print("component",int(k),"kurt",round(kt[k],2),
          "quantiles",np.quantile(col,[0,.01,.1,.5,.9,.99,1]),
          "mixnorm",round(np.linalg.norm(ica.mixing_[:,k]),3))

# Whitened output-weight clustering, with moderate singular-value weighting.
u,s,vt=svd(Z-Z.mean(0),full_matrices=False)
for dim in [24,27]:
    X=u[:,:dim]*np.sqrt(s[:dim])
    X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    km=KMeans(n_clusters=28,n_init=30,random_state=3).fit(X)
    counts=np.bincount(km.labels_,minlength=28)
    ix=np.argsort(-counts)
    print("\noutput clusters, dimension",dim,"sizes",counts[ix])
    centers=km.cluster_centers_[ix]
    centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    similarities=centers@centers.T
    print("cluster center nearest-neighbor cosines",np.max(similarities-np.eye(28)*2,axis=1))
    print("largest cluster pair similarities:")
    pairs=sorted([(similarities[a,b],a,b) for a in range(28) for b in range(a)],reverse=True)
    print(pairs[:20])

print("\nFirst-layer MLP neuron input-weight ICA:")
F=W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float)
F-=F.mean(1,keepdims=True)
ica=FastICA(n_components=60,whiten="unit-variance",random_state=11,max_iter=2000,tol=2e-4)
C=ica.fit_transform(F)
kt=kurtosis(C,axis=0)
order=np.argsort(-kt)
print("sorted excess kurtoses:",kt[order])
print("mixing-column norms:",np.linalg.norm(ica.mixing_,axis=0)[order])
```
