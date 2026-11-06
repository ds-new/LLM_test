```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

g=W["ln1_w.0"].astype(float)
def centered(a):
    return a-a.mean(axis=-1,keepdims=True)
Q=centered(W["W_q.0"][128:192].astype(float)*g)
K=centered(W["W_k.0"][128:192].astype(float)*g)
V=centered(W["W_v.0"][128:192].astype(float)*g)
Pstack=np.concatenate([Q,K],axis=0)
_,sp,ph=svd(Pstack,full_matrices=False)
_,sv,vh=svd(V,full_matrices=False)
print("L0H2 stacked Q,K singular values:\n",sp[:80])
print("L0H2 V singular values:\n",sv)
print("L0H2 QK-product singular values:\n",svdvals(Q.T@K/8)[:64])
print("Q/K vs V input-subspace squared overlap / dimension:")
for r in [8,16,24,28,32,40]:
    print(r,np.linalg.norm(ph[:r]@vh[:r].T)**2/r)

print("\nL0 value stack, effective MLP input, and final MLP output spectra:")
for name,a in [
    ("V0 effective",centered(W["W_v.0"].astype(float)*g)),
    ("FC0 effective",centered(W["W_fc.0"].astype(float)*W["ln2_w.0"])),
    ("PROJ3",W["W_proj.3"].astype(float)),
    ("O3",W["W_o.3"].astype(float))]:
    s=svdvals(a)
    print(name,s[:70])

print("\nL0 head read-space energy in L0H2 Q/K-derived and V-derived subspaces:")
P=ph[:24].T
T=vh[:28].T
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    values=[]
    for name in ["W_q","W_k","W_v"]:
        a=centered(W[name+".0"][sl].astype(float)*g)
        values.append((np.linalg.norm(a@P)**2/np.linalg.norm(a)**2,
                       np.linalg.norm(a@T)**2/np.linalg.norm(a)**2))
    print("H%d Q,K,V [QK-subspace,V-subspace]:"%j,np.array(values))

print("\nLater heads: QK energy in position-candidate and token-candidate subspaces:")
for l in range(1,4):
    gg=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        q=centered(W[f"W_q.{l}"][sl].astype(float)*gg)
        k=centered(W[f"W_k.{l}"][sl].astype(float)*gg)
        a=q.T@k/8
        den=np.linalg.norm(a)**2
        print("L%dH%d P_query=%.3f P_key=%.3f T_query=%.3f T_key=%.3f" %
              (l,j,np.linalg.norm(P.T@a)**2/den,np.linalg.norm(a@P)**2/den,
               np.linalg.norm(T.T@a)**2/den,np.linalg.norm(a@T)**2/den))
```
