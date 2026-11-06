```python
import numpy as np
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)

def spectrum(a):
    s = svdvals(np.asarray(a, dtype=np.float64), check_finite=False)
    e = s*s
    return s, e.sum() / e[0], np.exp(-np.sum((e/e.sum()) * np.log(e/e.sum() + 1e-300)))

print("Shapes:", {k: v.shape for k, v in W.items()})
for l in range(4):
    print("\nLAYER", l)
    for name in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        a = W[f"{name}.{l}"]
        print(name, "mean/std/min/max/norm",
              np.round([a.mean(), a.std(), a.min(), a.max(), np.linalg.norm(a)], 5))
    for name in ["W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"]:
        a = W[f"{name}.{l}"]
        s, sr, er = spectrum(a)
        print(name, "Frob=%.3f stable_rank=%.2f entropy_rank=%.2f" %
              (np.linalg.norm(a), sr, er),
              "singular_values[0:8]=", np.round(s[:8], 3),
              "s[32,64,128,256,-1]=", np.round(s[[32,64,128,256,-1]], 3))
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:,sl].astype(float)
        a = q.T @ k / 8
        b = o @ v
        sa, sra, era = spectrum(a)
        sb, srb, erb = spectrum(b)
        ea = np.linalg.eigvals(k @ q.T / 8)
        eb = np.linalg.eigvals(v @ o)
        ea = ea[np.argsort(-np.abs(ea))]
        eb = eb[np.argsort(-np.abs(eb))]
        print("L%dH%d" % (l,j),
              "Q,K,V,O norms", np.round([np.linalg.norm(z) for z in (q,k,v,o)],3),
              "QK norm/srank/symcos", np.round([np.linalg.norm(a),sra,
                  np.sum(a*a.T)/np.sum(a*a)],3),
              "OV norm/srank/trace", np.round([np.linalg.norm(b),srb,np.trace(b)],3))
        print("  QK sv", np.round(sa[:8],3), "eig", np.round(ea[:6],3))
        print("  OV sv", np.round(sb[:8],3), "eig", np.round(eb[:6],3))
```
