```python
import numpy as np
from scipy.linalg import svd, eigh, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

def eff(name,l,ln="ln1_w"):
    a=W[f"{name}.{l}"].astype(float)*W[f"{ln}.{l}"].astype(float)
    return a-a.mean(1,keepdims=True)
F=eff("W_fc",0,"ln2_w")
V=eff("W_v",0)
B=W["W_o.0"][:,192:256].astype(float)@V[192:256]
_,sv,vt=svd(V[192:256],full_matrices=False)

# Decompose MLP weight rows into paired read/write dictionaries of L0H3.
# This is a least-squares fit to weights, not a layer evaluation.
print("Paired read/write weight-covariance diagnostics:")
for rank in [16,20,24,27,32,40]:
    R=vt[:rank].T
    D=np.concatenate([R,B@R],axis=1)
    coefficients=F@np.linalg.pinv(D.T,rcond=1e-5)
    X=coefficients[:,:rank]
    Y=coefficients[:,rank:]
    X-=X.mean(0); Y-=Y.mean(0)
    cx=X.T@X; cy=Y.T@Y
    def invsqrt(c):
        d,u=eigh(c)
        return (u/np.sqrt(np.maximum(d,1e-6)))@u.T
    # Keep the same coordinate gauge on the two sides.
    H=invsqrt((cx+cy)/2)
    C=H@(X.T@Y)@H
    es=eigh((C+C.T)/2,eigvals_only=True)
    ev=eigvals(C)
    print("rank",rank,
          "explained F energy",round(np.linalg.norm(coefficients@D.T)**2/np.linalg.norm(F)**2,3),
          "sym eigs",es,
          "largest eigs",ev[np.argsort(-np.abs(ev))[:8]])

print("\nFull head-to-head key-composition trace ratios (row receiver; columns prior heads):")
As=[]; Bs=[]
for l in range(4):
    q,k,v=eff("W_q",l),eff("W_k",l),eff("W_v",l)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        As.append(q[sl].T@k[sl]/8)
        Bs.append(W[f"W_o.{l}"][:,sl].astype(float)@v[sl])
for h in range(6,24):
    ratios=[]
    traces=[]
    for z in range((h//6)*6):
        c=As[h]@Bs[z]
        ev=eigvals(c)
        ratios.append(np.trace(c)/(np.abs(ev).sum()+1e-12))
        traces.append(np.trace(c))
    print(f"L{h//6}H{h%6}","signed eig-mass ratios",np.array(ratios),"traces",np.array(traces))
```
