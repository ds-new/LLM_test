```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)

d=384
P=np.eye(d)-np.ones((d,d))/d
g=W["ln1_w.0"].astype(float)
pos=svd(np.concatenate([W["W_q.0"][192:256].T,
                       W["W_k.0"][192:256].T],axis=1).astype(float),
        full_matrices=False)[0][:,:32]
v=W["W_v.0"][192:256].astype(float)
_,s,vt=svd(v@(np.eye(d)-pos@pos.T),full_matrices=False)
T=np.linalg.qr(P@(vt[:23].T/np.maximum(g[:,None],.08)))[0]
B=W["W_o.0"][:,192:256].astype(float)@(v*g)@P
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float))@P
X=F@T
Y=F@B@T

# ICA is fitted to neuron WEIGHT vectors, never to network activations.
ica=FastICA(n_components=23,whiten="unit-variance",random_state=123,
            max_iter=5000,tol=1e-5)
S=ica.fit_transform(X)
sign=np.sign(skew(S,axis=0))
sign[sign==0]=1
S*=sign
R=ica.components_*sign[:,None]
Z=(Y-Y.mean(0))@R.T
Z/=np.std(Z,axis=0,keepdims=True)
print("ICA weight-component skewness:",np.round(skew(S,axis=0),3))
print("ICA weight-component excess kurtosis:",np.round(kurtosis(S,axis=0),3))
print("Fraction of first-MLP weight variance explained by content subspace:",
      np.linalg.norm(X)**2/np.linalg.norm(F)**2)
print("\nCo-occurrence of current and shifted ICA WEIGHT coefficients:")
corr=S.T@Z/len(S)
for i in range(23):
    inds=np.argsort(-corr[i])[:5]
    print(i,[(int(j),round(float(corr[i,j]),3)) for j in inds],
          "most negative",int(np.argmin(corr[i])),round(float(corr[i].min()),3))

print("\nMean shifted weight coefficients in neurons with extreme current coefficients:")
adj=[]
for i in range(23):
    rows=np.argsort(-S[:,i])[:40]
    z=Z[rows].mean(0)
    adj.append(z)
    inds=np.argsort(-z)[:5]
    print(i,[(int(j),round(float(z[j]),3)) for j in inds])
adj=np.array(adj)
print("\nEigenvalues of centered co-occurrence matrix:",
      np.round(np.linalg.eigvals(corr),3))
print("Top singular values of co-occurrence matrix:",np.round(svd(corr,compute_uv=False),3))
```
