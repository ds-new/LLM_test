```python
import numpy as np
from scipy.linalg import svd, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

print("Detailed spectra: counts of strongly represented directions")
for l in range(4):
    for name in ["W_fc","W_proj"]:
        s=svd(W[f"{name}.{l}"].astype(float),compute_uv=False)
        print(name,l,"first 65:",s[:65])

g0=W["ln1_w.0"].astype(float)
def effective(name,l):
    a=W[f"{name}.{l}"].astype(float)*W[f"ln1_w.{l}"].astype(float)
    return a-a.mean(1,keepdims=True)
qs=[effective("W_q",l) for l in range(4)]
ks=[effective("W_k",l) for l in range(4)]
vs=[effective("W_v",l) for l in range(4)]
As=[]; Bs=[]
for l in range(4):
    for j in range(6):
        s=slice(64*j,64*j+64)
        As.append(qs[l][s].T@ks[l][s]/8)
        Bs.append(W[f"W_o.{l}"][:,s].astype(float)@vs[l][s])

# These are subspaces of weight matrices, not network activations.
poscand=np.concatenate([qs[0][192:256],ks[0][192:256]],axis=0)
_,ss,vt=svd(poscand,full_matrices=False)
P=vt[:32].T
f=W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float)
f-=f.mean(1,keepdims=True)
_,sf,vf=svd(f,full_matrices=False)
T=vf[:28].T
print("\nCandidate weight-subspace overlaps:")
print("singular values of L0H3 Q/K stack:",ss[:60])
print("overlap of L0 MLP top28 and L0H3 Q/K top32:",svd(T.T@P,compute_uv=False))
for h in range(24):
    l,j=divmod(h,6); s=slice(64*j,64*j+64)
    a=As[h]; b=Bs[h]
    q,k=qs[l][s],ks[l][s]
    print(f"L{l}H{j}", "Q,K energy fractions in L0H3 subspace:",
          round(np.linalg.norm(q@P)**2/np.linalg.norm(q)**2,3),
          round(np.linalg.norm(k@P)**2/np.linalg.norm(k)**2,3),
          "Q,K energy fractions in L0MLP subspace:",
          round(np.linalg.norm(q@T)**2/np.linalg.norm(q)**2,3),
          round(np.linalg.norm(k@T)**2/np.linalg.norm(k)**2,3))

print("\nL0H3 -> later QK composition spectra:")
for h in range(6,24):
    c=As[h]@Bs[3]
    ev=eigvals(c)
    ix=np.argsort(-np.abs(ev))
    print(f"L{h//6}H{h%6}", "norm,trace,trace/sumabsEig:",
          round(np.linalg.norm(c),3),round(np.trace(c),3),
          round(np.trace(c)/(np.abs(ev).sum()+1e-12),3),
          "top eigenvalues:",ev[ix[:6]])

print("\nCross-layer positive-copy / negative-copy OV eigenvalue mass:")
for h,b in enumerate(Bs):
    l,j=divmod(h,6); s=slice(64*j,64*j+64)
    ev=eigvals(vs[l][s]@W[f"W_o.{l}"][:,s].astype(float))
    print(f"L{l}H{j}",ev[np.argsort(-np.abs(ev))[:8]])
```
