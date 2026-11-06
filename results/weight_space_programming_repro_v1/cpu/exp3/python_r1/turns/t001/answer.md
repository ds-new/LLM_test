```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def norm(x):
    return float(la.norm(np.asarray(x, dtype=np.float64)))

def participation(s):
    z = np.asarray(s)**2
    return float(z.sum()**2 / (z @ z + 1e-30))

print("LayerNorm parameters: mean, std, min, max, L2 norm")
for l in range(4):
    for name in ("ln1_w", "ln1_b", "ln2_w", "ln2_b"):
        a = W[f"{name}.{l}"]
        print(f"{name}.{l}: {a.mean():.5f} {a.std():.5f} {a.min():.5f} {a.max():.5f} {norm(a):.4f}")

print("\nMatrix summaries: Frobenius norm, top 8 singular values, spectral participation rank")
for l in range(4):
    for name in ("W_q", "W_k", "W_v", "W_o", "W_fc", "W_proj"):
        a = np.asarray(W[f"{name}.{l}"], dtype=np.float64)
        s = la.svdvals(a)
        print(f"{name}.{l}: norm={norm(a):.3f} top={np.round(s[:8],3)} rank={participation(s):.2f}")

print("\nHead summaries (linear weight products only)")
print("QK = (W_q diag(ln1_w))^T (W_k diag(ln1_w)); OV = W_o W_v diag(ln1_w)")
print("sym = <M,M^T>/||M||_F^2; eigsign = sum Re(eigenvalues)/sum |eigenvalues|")
for l in range(4):
    g = np.asarray(W[f"ln1_w.{l}"], dtype=np.float64)
    b = np.asarray(W[f"ln1_b.{l}"], dtype=np.float64)
    for j in range(6):
        sl = slice(64*j,64*(j+1))
        q = np.asarray(W[f"W_q.{l}"][sl],dtype=np.float64)
        k = np.asarray(W[f"W_k.{l}"][sl],dtype=np.float64)
        v = np.asarray(W[f"W_v.{l}"][sl],dtype=np.float64)
        o = np.asarray(W[f"W_o.{l}"][:,sl],dtype=np.float64)
        qg, kg, vg = q*g, k*g, v*g
        a, m = qg.T@kg, o@vg
        ea, em = la.eigvals(kg@qg.T), la.eigvals(vg@o)
        sa, sm = la.svdvals(a), la.svdvals(m)
        print(f"L{l}H{j}: q/k/v/o={norm(q):.2f}/{norm(k):.2f}/{norm(v):.2f}/{norm(o):.2f}"
              f" QKnorm={norm(a):.2f} QKrank={participation(sa):.1f}"
              f" QKsym={np.sum(a*a.T)/(norm(a)**2):.3f} QKeigsign={ea.real.sum()/(np.abs(ea).sum()+1e-30):.3f}"
              f" OVnorm={norm(m):.2f} OVrank={participation(sm):.1f}"
              f" OVsym={np.sum(m*m.T)/(norm(m)**2):.3f} OVeigsign={em.real.sum()/(np.abs(em).sum()+1e-30):.3f}"
              f" qb/kb/vb={norm(q@b):.2f}/{norm(k@b):.2f}/{norm(v@b):.2f}"
              f" QKtop={np.round(sa[:5],2)} OVtop={np.round(sm[:5],2)}")
```
