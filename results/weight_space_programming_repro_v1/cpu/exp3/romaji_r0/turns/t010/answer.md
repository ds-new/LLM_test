```python
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4,suppress=True,linewidth=160)
rng=np.random.default_rng(2026)

def eff(name,l):
    a=W[f"{name}.{l}"].astype(float)*W[f"ln1_w.{l}"].astype(float)
    return a-a.mean(1,keepdims=True)
A=[]; B=[]; r=[]
for l in range(4):
    q,k,v=eff("W_q",l),eff("W_k",l),eff("W_v",l)
    b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl=slice(64*j,64*j+64)
        A.append(q[sl].T@k[sl]/8)
        B.append(W[f"W_o.{l}"][:,sl].astype(float)@v[sl])
        r.append(k[sl].T@(W[f"W_q.{l}"][sl].astype(float)@b)/8)

print("First-layer OV top-two singular-value energy fraction:")
for j in range(6):
    s=svd(B[j],compute_uv=False)
    print(f"L0H{j}",np.sum(s[:2]**2)/np.sum(s**2))

C=svd(np.concatenate([B[j] for j in [1,2,4,5]],axis=1),full_matrices=False)[0][:,:4]
ref=r[6]/np.linalg.norm(r[6])
print("\nKey-only affine coefficients: norm; fraction in shared write subspace; cosine to L1H0 coefficient")
for h in range(24):
    print(f"L{h//6}H{h%6}",round(np.linalg.norm(r[h]),4),
          round(np.linalg.norm(C.T@r[h])**2/np.linalg.norm(r[h])**2,4),
          round(r[h]@ref/np.linalg.norm(r[h]),4))

# Coordinate-permutation controls test a particular weight alignment only.
# They are not simulations of the network or statistical tests of corpus identity.
receiver=A[21]
source=B[3]
denom=np.linalg.norm(receiver)*np.linalg.norm(source)
observed=np.sum(receiver*source.T)/denom
controls=[]
for _ in range(400):
    p=rng.permutation(384)
    controls.append(np.sum(receiver*source[np.ix_(p,p)].T)/denom)
controls=np.array(controls)
print("\nL3H3 QK / L0H3 OV signed Frobenius alignment:")
print("observed",observed,"permutation mean/std",controls.mean(),controls.std(),
      "permutation min/max",controls.min(),controls.max())
print("Same alignment for other last-layer heads:",
      [np.sum(A[h]*source.T)/(np.linalg.norm(A[h])*np.linalg.norm(source)) for h in range(18,24)])
```
