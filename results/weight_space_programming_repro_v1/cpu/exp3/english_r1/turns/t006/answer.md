```python
import numpy as np
from scipy.linalg import svd, eigh
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

def ctr(a):
    return a-a.mean(-1,keepdims=True)
def invsqrt(a):
    e,u=eigh(a)
    return (u*(1/np.sqrt(np.maximum(e,1e-8*max(e.max(),1e-12)))))@u.T

g=W["ln1_w.0"].astype(float)
V=W["W_v.0"][128:192].astype(float)
O=W["W_o.0"][:,128:192].astype(float)
B=O@ctr(V*g)
F=ctr(W["W_fc.0"].astype(float)*W["ln2_w.0"])
q=ctr(W["W_q.0"][128:192].astype(float)*g)
k=ctr(W["W_k.0"][128:192].astype(float)*g)
A=q.T@k/8

print("L0H2 QK eigenvalue-energy / singular-value-energy:",
      np.sum(np.abs(np.linalg.eigvals(k@q.T/8))**2)/np.sum(A*A))
print("L0H2 QK traces of powers (normalized by spectral radius):")
radius=np.max(np.abs(np.linalg.eigvals(k@q.T/8)))
small=k@q.T/(8*radius)
power=np.eye(64)
for i in range(1,9):
    power=power@small
    print(i,round(float(np.trace(power)),4))

print("\nExploratory MLP-weight cross-correlation of current-read and transported-read subspaces.")
print("This is a correlation of coefficient matrices, not a bigram probability or a layer evaluation.")
_,_,vraw=svd(V,full_matrices=False)
_,_,veff=svd(ctr(V*g),full_matrices=False)
_,_,pp=svd(np.concatenate([q,k]),full_matrices=False)
for kind in ["raw", "effective", "raw_minus_QK"]:
    for r in [16,24,28]:
        T=(vraw[:r].T if kind!="effective" else veff[:r].T)
        if kind=="raw_minus_QK":
            T=T-pp[:24].T@(pp[:24]@T)
            T=np.linalg.qr(T)[0]
        X=F@T
        Y=F@B@T
        X=X-X.mean(0)
        Y=Y-Y.mean(0)
        C=invsqrt(X.T@X)@(X.T@Y)@invsqrt(Y.T@Y)
        es=eigh((C+C.T)/2,eigvals_only=True)
        print(kind,r,"symmetric eigenvalues:",es)

print("\nRobustness of L3 key-composition trace cosines through L0H2:")
for mode in ["raw", "gamma", "gamma_centered"]:
    vals=[]
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        q3=W["W_q.3"][sl].astype(float)
        k3=W["W_k.3"][sl].astype(float)
        v0=V.copy()
        if mode!="raw":
            q3*=W["ln1_w.3"]
            k3*=W["ln1_w.3"]
            v0*=g
        if mode=="gamma_centered":
            q3=ctr(q3); k3=ctr(k3); v0=ctr(v0)
        a=q3.T@k3/8
        b=O@v0
        vals.append(np.sum(a*b.T)/(np.linalg.norm(a)*np.linalg.norm(b)))
    print(mode,np.array(vals))
```
