```python
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=4,suppress=True,linewidth=160)

def ctr(a): return a-a.mean(-1,keepdims=True)
As=[]; Bs=[]; cs=[]
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        Q=W[f"W_q.{l}"][s].astype(float)
        K=W[f"W_k.{l}"][s].astype(float)
        V=W[f"W_v.{l}"][s].astype(float)
        O=W[f"W_o.{l}"][:,s].astype(float)
        q=ctr(Q*g); k=ctr(K*g); v=ctr(V*g)
        As.append(q.T@k/8)
        Bs.append(O@v)
        cs.append(k.T@(Q@b)/8)

common=np.mean([c/np.linalg.norm(c) for c in cs[6:12]],axis=0)
common/=np.linalg.norm(common)
u,s,vh=svd(Bs[5],full_matrices=False)
summary_space=u[:,:2]
print("Alignment with the common L1 affine key-selection direction:")
print("head |c| cos(c,common_L1) fraction_c_in_L0H5_top2_output")
for h,c in enumerate(cs):
    print("L%dH%d %.4f %.4f %.4f" %
          (h//6,h%6,np.linalg.norm(c),c@common/np.linalg.norm(c),
           np.linalg.norm(summary_space.T@c)**2/np.linalg.norm(c)**2))

print("\nL0H3 Frobenius cosine with powers of L0H2 QK, and their transposes:")
a=As[2]/np.linalg.norm(As[2],2)
p=np.eye(384)
for exponent in range(1,7):
    p=p@a
    print(exponent,
          np.sum(As[3]*p)/(np.linalg.norm(As[3])*np.linalg.norm(p)),
          np.sum(As[3]*p.T)/(np.linalg.norm(As[3])*np.linalg.norm(p)))

print("\nCoordinate-permutation null for the L3H1 / L0H2 circuit trace cosine.")
print("This permutes a weight product, not network inputs.")
a=As[19]; b=Bs[2]
den=np.linalg.norm(a)*np.linalg.norm(b)
observed=np.sum(a*b.T)/den
rng=np.random.default_rng(2026)
null=[]
for _ in range(1000):
    ix=rng.permutation(384)
    bp=b[np.ix_(ix,ix)]
    null.append(np.sum(a*bp.T)/den)
null=np.array(null)
print("observed=%.6f null_mean=%.6f null_std=%.6f null_max=%.6f"%
      (observed,null.mean(),null.std(),null.max()))
print("This null measures coordinate-alignment specificity, not calibrated behavioral confidence.")
```
