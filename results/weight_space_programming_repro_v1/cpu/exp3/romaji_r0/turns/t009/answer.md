```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

def eff(a,g):
    a=a.astype(float)*g.astype(float)
    return a-a.mean(1,keepdims=True)
F=eff(W["W_fc.0"],W["ln2_w.0"])
q=eff(W["W_q.0"],W["ln1_w.0"])
k=eff(W["W_k.0"],W["ln1_w.0"])
v=eff(W["W_v.0"],W["ln1_w.0"])
B=W["W_o.0"][:,192:256].astype(float)@v[192:256]
pos=svd(np.concatenate([q[192:256],k[192:256]],axis=0),full_matrices=False)[2][:40].T
otherO=np.concatenate([W["W_o.0"][:,64*j:64*j+64] for j in [1,2,4,5]],axis=1).astype(float)
other=svd(otherO,full_matrices=False)[0][:,:6]
o3=svd(W["W_o.0"][:,192:256].astype(float),full_matrices=False)[0][:,:64]
o0=svd(W["W_o.0"][:,:64].astype(float),full_matrices=False)[0][:,:32]
U=svd(np.concatenate([o3,o0,other,pos,np.ones((384,1))],axis=1),full_matrices=False)[0]
Fres=F-(F@U)@U.T
u,s,vt=svd(Fres,full_matrices=False)
rank=27
Z=u[:,:rank]*np.sqrt(s[:rank])
row_scale=np.linalg.norm(Z,axis=1)
Z/=row_scale[:,None]
ica=FastICA(n_components=rank,whiten="unit-variance",max_iter=2500,tol=1e-5,random_state=9)
C=ica.fit_transform(Z)
sign=np.where(skew(C,axis=0)>=0,1.,-1.)
X=C*sign*row_scale[:,None]
D=(vt[:rank].T*np.sqrt(s[:rank]))@ica.mixing_@np.diag(sign)
dual=np.linalg.pinv(D.T)
Y=F@B@dual

# Correlations of fitted coefficients of weight rows.
X-=X.mean(0)
Y-=Y.mean(0)
cross=X.T@Y
cross/=np.linalg.norm(X,axis=0)[:,None]*np.linalg.norm(Y,axis=0)[None,:]
print("Anonymous weight-feature diagnostics: these are NOT identified characters or probabilities.")
print("feature: write-basis norm, coefficient skew, self correlation, strongest cross correlations")
for i in range(rank):
    top=np.argsort(-cross[i])[:5]
    bottom=np.argsort(cross[i])[:2]
    print(i,round(np.linalg.norm(D[:,i]),3),round(skew(X[:,i]),3),round(cross[i,i],3),
          "positive",[(int(j),round(cross[i,j],3)) for j in top],
          "negative",[(int(j),round(cross[i,j],3)) for j in bottom])
print("cross-correlation singular values",svd(cross,compute_uv=False))
print("symmetric-part eigenvalues",np.linalg.eigvalsh((cross+cross.T)/2))
print("largest absolute offdiagonal correlations")
pairs=sorted([(abs(cross[i,j]),i,j,cross[i,j]) for i in range(rank) for j in range(rank) if i!=j],reverse=True)
print(pairs[:40])
```
