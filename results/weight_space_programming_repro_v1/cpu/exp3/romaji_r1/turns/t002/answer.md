```python
import numpy as np
from scipy.linalg import svd, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)

d = 384
P = np.eye(d) - np.ones((d,d))/d
A, B = {}, {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s = slice(j*64,(j+1)*64)
        q = W[f"W_q.{l}"][s].astype(float) * g
        k = W[f"W_k.{l}"][s].astype(float) * g
        v = W[f"W_v.{l}"][s].astype(float) * g
        q, k, v = q @ P, k @ P, v @ P
        A[l,j] = q.T @ k / 8
        B[l,j] = W[f"W_o.{l}"][:,s].astype(float) @ v

print("Singular values of final MLP output matrix, first 60:")
print(np.round(svd(W["W_proj.3"].astype(float),compute_uv=False)[:60],4))
print("\nFirst-layer QK spectra after algebraic gamma/centering factors:")
for j in range(6):
    a = A[0,j]
    s = svd(a,compute_uv=False)
    e = eigvals(a)
    e = e[np.argsort(-np.abs(e))][:40]
    print("L0H%d" % j, "sv:", np.round(s[:40],4))
    print("eigenvalue magnitude,phase:", np.round(np.c_[np.abs(e),np.angle(e)],3).tolist())

print("\nCross-head induction-style products A_lh @ B_earlier:")
print("For each later head, best 5 by trace/nuclear_norm; this is only a matrix diagnostic.")
for l in range(1,4):
    for j in range(6):
        rows = []
        for m in range(l):
            for k in range(6):
                c = A[l,j] @ B[m,k]
                ss = svd(c,compute_uv=False)
                nuc = ss.sum()
                sym = np.sum(c*c.T)/(np.sum(c*c)+1e-30)
                rows.append((np.trace(c)/nuc, m, k, sym,
                             np.linalg.norm(c)/(np.linalg.norm(A[l,j])*np.linalg.norm(B[m,k])),
                             np.trace(c)))
        rows.sort(reverse=True)
        print("L%dH%d" % (l,j), [tuple(round(x,4) if isinstance(x,(float,np.floating)) else x for x in r) for r in rows[:5]])

print("\nFirst-layer source candidates, A_lh @ B_0k: trace/nuclear_norm for k=0..5")
for l in range(1,4):
    for j in range(6):
        vals=[]
        for k in range(6):
            c=A[l,j]@B[0,k]
            vals.append(np.trace(c)/svd(c,compute_uv=False).sum())
        print("L%dH%d" % (l,j), np.round(vals,4))

print("\nQK alignment with first-layer QK matrices (Frobenius cosine):")
for l in range(4):
    for j in range(6):
        a=A[l,j]
        vals=[np.sum(a*A[0,k])/(np.linalg.norm(a)*np.linalg.norm(A[0,k])) for k in range(6)]
        print("L%dH%d" % (l,j),np.round(vals,3))
```
