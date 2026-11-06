```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(2)

def summary(x):
    return "min %.4g max %.4g mean %.4g std %.4g norm %.4g" % (
        x.min(), x.max(), x.mean(), x.std(), np.linalg.norm(x))

def product_stats(A, B):
    # Pure matrix analysis of A @ B; no network inputs or layer execution.
    qa, ra = np.linalg.qr(A, mode="reduced")
    qb, rb = np.linalg.qr(B.T, mode="reduced")
    s = svdvals(ra @ rb.T)
    ev = np.linalg.eigvals(B @ A)
    f2 = float(s @ s)
    stable_rank = f2 / max(float(s[0] ** 2), 1e-30)
    trace_ratio = float(ev.real.sum() / max(np.abs(ev).sum(), 1e-30))
    symmetry = float(np.trace((B @ A) @ (B @ A)) / max(f2, 1e-30))
    return (np.sqrt(f2), s[0], stable_rank, trace_ratio, symmetry,
            np.max(ev.real), np.min(ev.real))

print("All calculations below concern weights and their matrix products only.")
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        print(name, summary(W[f"{name}.{l}"]))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(np.float64)
        s = svdvals(a)
        print(name, "F=%.4f s[:8]=%s stable_rank=%.3f" %
              (np.linalg.norm(a), np.array2string(s[:8], precision=3),
               np.sum(s*s)/(s[0]*s[0])))
    print("head product: F, s1, stable_rank, trace/sum|eig|, symmetry, max_Re_eig, min_Re_eig")
    for j in range(6):
        sl = slice(j*64, (j+1)*64)
        Q = W[f"W_q.{l}"][sl].astype(np.float64)
        K = W[f"W_k.{l}"][sl].astype(np.float64)
        V = W[f"W_v.{l}"][sl].astype(np.float64)
        O = W[f"W_o.{l}"][:, sl].astype(np.float64)
        b = W[f"ln1_b.{l}"].astype(np.float64)
        g = W[f"ln1_w.{l}"].astype(np.float64)
        qk = product_stats((Q*g).T/8, K*g)
        ov = product_stats(O, V*g)
        print("L%dH%d QK %s OV %s bias_norms(q,k)=(%.3f,%.3f)" %
              (l, j, np.array2string(np.array(qk), precision=3),
               np.array2string(np.array(ov), precision=3),
               np.linalg.norm(Q@b), np.linalg.norm(K@b)))
```
