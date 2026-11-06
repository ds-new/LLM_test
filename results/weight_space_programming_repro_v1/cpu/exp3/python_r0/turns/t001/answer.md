```python
import numpy as np
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=160)
threadpool_limits(limits=4)

def spectrum(a):
    s = np.linalg.svd(a.astype(np.float64), compute_uv=False)
    e = s * s
    p = e / max(e.sum(), 1e-30)
    r90 = int(np.searchsorted(np.cumsum(p), 0.9) + 1)
    erank = np.exp(-np.sum(p * np.log(np.maximum(p, 1e-30))))
    return s, r90, erank

def centered_columns(a):
    return a - a.mean(axis=1, keepdims=True)

print("All diagnostics below are weight-only linear algebra; no network evaluations.")
print("Shapes:", {k: v.shape for k, v in sorted(W.items())})

for l in range(4):
    print("\nLAYER", l)
    for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
        a = W[f"{name}.{l}"].astype(np.float64)
        print(name, "mean/std/norm =", np.round([a.mean(), a.std(), np.linalg.norm(a)], 5),
              "quantiles =", np.round(np.quantile(a, [0, .1, .5, .9, 1]), 5))
    for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
        a = W[f"{name}.{l}"].astype(np.float64)
        s, r90, erank = spectrum(a)
        print(name, "Frob=%.4f std=%.5f r90=%d effective_rank=%.1f top8=%s" %
              (np.linalg.norm(a), a.std(), r90, erank, np.round(s[:8], 4)))

    gamma = W[f"ln1_w.{l}"].astype(np.float64)
    beta = W[f"ln1_b.{l}"].astype(np.float64)
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        qr = W[f"W_q.{l}"][sl].astype(np.float64)
        kr = W[f"W_k.{l}"][sl].astype(np.float64)
        vr = W[f"W_v.{l}"][sl].astype(np.float64)
        o = W[f"W_o.{l}"][:, sl].astype(np.float64)
        # Fixed matrix coefficients induced by LN's affine parameters,
        # with the constant residual direction projected out.
        q = centered_columns(qr * gamma)
        k = centered_columns(kr * gamma)
        v = centered_columns(vr * gamma)
        A = q.T @ k / 8.0
        B = o @ v
        sa, ra, ea = spectrum(A)
        sb, rb, eb = spectrum(B)
        na, nb = np.linalg.norm(A), np.linalg.norm(B)
        key_bias = (qr @ beta) @ k / 8.0
        print("L%dH%d QK: F=%.4f r90=%d erank=%.1f top3energy=%.3f symcos=%.3f trace/F=%.3f keybias/F=%.3f;"
              " OV: F=%.4f r90=%d erank=%.1f symcos=%.3f trace/F=%.3f" %
              (l, j, na, ra, ea, np.sum(sa[:3]**2)/max(na**2, 1e-30),
               np.sum(A*A.T)/max(na**2, 1e-30), np.trace(A)/max(na, 1e-30),
               np.linalg.norm(key_bias)/max(na, 1e-30),
               nb, rb, eb, np.sum(B*B.T)/max(nb**2, 1e-30),
               np.trace(B)/max(nb, 1e-30)))
        print("    raw Q/K/V/O norms =", np.round(
            [np.linalg.norm(z) for z in (qr, kr, vr, o)], 4),
            "QK top6 =", np.round(sa[:6], 4),
            "OV top6 =", np.round(sb[:6], 4))

    fc = W[f"W_fc.{l}"].astype(np.float64)
    proj = W[f"W_proj.{l}"].astype(np.float64)
    g2 = W[f"ln2_w.{l}"].astype(np.float64)
    b2 = W[f"ln2_b.{l}"].astype(np.float64)
    f = centered_columns(fc * g2)
    dots = np.sum(f * proj.T, axis=1)
    cos = dots / np.maximum(np.linalg.norm(f, axis=1) *
                           np.linalg.norm(proj, axis=0), 1e-30)
    biases = fc @ b2
    print("MLP input/output-neuron cosine quantiles:",
          np.round(np.quantile(cos, [0, .1, .25, .5, .75, .9, 1]), 4))
    print("MLP affine-bias coefficient quantiles:",
          np.round(np.quantile(biases, [0, .1, .25, .5, .75, .9, 1]), 4))
```
