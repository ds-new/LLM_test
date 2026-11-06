```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=160)

print("Keys:", sorted(W))
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(float)
        print(name, "mean,std,min,max,norm:", np.array([a.mean(), a.std(), a.min(), a.max(), np.linalg.norm(a)]))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(float)
        s = svdvals(a)
        print(name, "norm,stable_rank,top10_s,median_s:",
              round(np.linalg.norm(a), 3), round(np.sum(s*s)/(s[0]*s[0]), 2),
              np.round(s[:10], 3), round(np.median(s), 3))
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:, sl].astype(float)
        qe, ke, ve = q*g, k*g, v*g
        qe -= qe.mean(axis=1, keepdims=True)
        ke -= ke.mean(axis=1, keepdims=True)
        ve -= ve.mean(axis=1, keepdims=True)
        A = qe.T @ ke / 8
        B = o @ ve
        sa, sb = svdvals(A), svdvals(B)
        ea = np.linalg.eigvalsh((A + A.T)/2)
        eb = np.linalg.eigvals(ve @ o)
        print(f"L{l}H{j}",
              "QK norm/top6/stable/asym:", round(np.linalg.norm(A),3), np.round(sa[:6],3),
              round(np.sum(sa*sa)/(sa[0]*sa[0]),2), round(np.linalg.norm(A-A.T)/(2*np.linalg.norm(A)),3),
              "QK sym extremes:", np.round(ea[[0,1,-2,-1]],3),
              "OV norm/top6/tr/posEigFrac:", round(np.linalg.norm(B),3), np.round(sb[:6],3),
              round(np.trace(B),3), round(np.maximum(eb.real,0).sum()/(np.abs(eb).sum()+1e-12),3),
              "q_bias,k_bias:", round(np.linalg.norm(q@b),3), round(np.linalg.norm(k@b),3))
```