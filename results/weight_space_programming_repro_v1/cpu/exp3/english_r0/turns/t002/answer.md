```python
import numpy as np
from scipy.linalg import svdvals, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

def centered_columns(a):
    return a - a.mean(axis=1, keepdims=True)

QK, OV = {}, {}
for l in range(4):
    w = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    print("\nLAYER", l)
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q, k, v = [W[f"W_{s}.{l}"][sl].astype(float) for s in ["q", "k", "v"]]
        o = W[f"W_o.{l}"][:, sl].astype(float)
        qe, ke, ve = [centered_columns(a*w) for a in [q, k, v]]
        m, a = qe.T@ke/8, o@ve
        QK[l,j], OV[l,j] = m, a
        sm, sa = svdvals(m), svdvals(a)
        ev = eigvals(ke@qe.T/8)
        ev = ev[np.argsort(np.abs(ev))[::-1]]
        key_bias = (q@b)@ke/8
        print(f"L{l}H{j}",
              "effective_QK_F", round(np.linalg.norm(m),4),
              "sv", np.round(sm[:8],4),
              "stable_rank", round(np.sum(sm**2)/sm[0]**2,3),
              "trace", round(np.trace(m),4),
              "eigenvalues", np.round(ev[:8],4),
              "key_bias_norm", round(np.linalg.norm(key_bias),4),
              "effective_OV_F", round(np.linalg.norm(a),4),
              "OV_trace", round(np.trace(a),4))
    for name in ["W_fc", "W_proj"]:
        s = svdvals(W[f"{name}.{l}"].astype(float))
        print(name, "top48_singular_values", np.round(s[:48],3))

print("\nFIRST-LAYER QK pairwise Frobenius cosines")
ms = [QK[0,j] for j in range(6)]
print(np.array([[np.sum(a*b)/np.linalg.norm(a)/np.linalg.norm(b) for b in ms] for a in ms]))

print("\nComposition: later QK times earlier OV; relative norm and signed trace/nuclear norm")
for l in range(1,4):
    for j in range(6):
        m = QK[l,j]
        results = []
        for p in range(l):
            for h in range(6):
                a = OV[p,h]
                c = m@a
                strength = np.linalg.norm(c)/(np.linalg.norm(m)*np.linalg.norm(a))
                sv = svdvals(c)
                positivity = np.trace(c)/sv.sum()
                results.append((strength, p, h, positivity, np.trace(c)))
        results.sort(reverse=True)
        print(f"L{l}H{j}", [(f"L{p}H{h}", round(s,4), round(pos,4), round(tr,4))
                           for s,p,h,pos,tr in results[:6]])
```
