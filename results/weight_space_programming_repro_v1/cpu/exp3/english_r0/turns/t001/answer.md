```python
import numpy as np
from scipy.linalg import svdvals

np.set_printoptions(precision=4, suppress=True, linewidth=160)
print("ARRAYS", [(k, v.shape) for k, v in sorted(W.items())])
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"].astype(float)
        ix = np.argsort(np.abs(a))[-8:][::-1]
        print(name, "quantiles", np.quantile(a, [0, .1, .5, .9, 1]),
              "largest_abs", list(zip(ix.tolist(), np.round(a[ix], 4).tolist())))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"].astype(float)
        s = svdvals(a)
        print(name, "F", round(np.linalg.norm(a), 4), "sv", np.round(s[:10], 4),
              "stable_rank", round(np.sum(s*s)/(s[0]*s[0]), 3))
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:, sl].astype(float)
        m = q.T @ k / 8
        a = o @ v
        sm, sa = svdvals(m), svdvals(a)
        print(f"L{l}H{j}",
              "QK_F", round(np.linalg.norm(m), 4), "QK_sv", np.round(sm[:6], 4),
              "QK_sym", round(np.linalg.norm(m+m.T)/(2*np.linalg.norm(m)), 4),
              "QK_trace", round(np.trace(m), 4),
              "OV_F", round(np.linalg.norm(a), 4), "OV_sv", np.round(sa[:6], 4),
              "OV_trace", round(np.trace(a), 4))
```
