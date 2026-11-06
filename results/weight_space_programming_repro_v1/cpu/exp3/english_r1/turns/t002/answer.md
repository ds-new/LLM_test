```python
import numpy as np
from scipy.linalg import svd, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

def cos(a,b):
    return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30)

As, Bs, us, vs = [], [], [], []
for l in range(4):
    aa, bb, uu, vv = [], [], [], []
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    print("\nLAYER",l)
    for j in range(6):
        s = slice(64*j,64*(j+1))
        Q = W[f"W_q.{l}"][s].astype(float)
        K = W[f"W_k.{l}"][s].astype(float)
        V = W[f"W_v.{l}"][s].astype(float)
        O = W[f"W_o.{l}"][:,s].astype(float)
        Qg, Kg, Vg = Q*g, K*g, V*g
        Qg -= Qg.mean(axis=1,keepdims=True)
        Kg -= Kg.mean(axis=1,keepdims=True)
        Vg -= Vg.mean(axis=1,keepdims=True)
        A = Qg.T@Kg/8
        B = O@Vg
        # The affine query-bias/key-feature term is a weight-derived direction.
        key_linear = Kg.T@(Q@b)/8
        query_linear = Qg.T@(K@b)/8
        u,t,vh = svd(A,full_matrices=False)
        aa.append(A); bb.append(B); uu.append(u[:,0]); vv.append(vh[0])
        ev = eigvals(Kg@Qg.T/8)
        ev = ev[np.argsort(-np.abs(ev))]
        print("H%d QK_s[:8]=%s key_linear_norm=%.4f query_linear_norm=%.4f u.v=%.4f" %
              (j,t[:8],np.linalg.norm(key_linear),np.linalg.norm(query_linear),u[:,0]@vh[0]))
        if l == 0:
            print("   QK_eig[:16]=",ev[:16])
            print("   key_linear alignment with right singular vector=%.4f" %
                  (key_linear@vh[0]/(np.linalg.norm(key_linear)+1e-30)))
    print("QK pairwise cosine:\n",np.array([[cos(a,c) for c in aa] for a in aa]))
    print("OV pairwise cosine:\n",np.array([[cos(a,c) for c in bb] for a in bb]))
    if l == 0:
        print("leading Q-direction cosine:\n",np.array(uu)@np.array(uu).T)
        print("leading K-direction cosine:\n",np.array(vv)@np.array(vv).T)
    As.append(aa); Bs.append(bb); us.append(uu); vs.append(vv)

print("\nCross-layer QK cosine, each head vs the six heads of preceding layer:")
for l in range(1,4):
    print("layer",l,"\n",np.array([[cos(a,c) for c in As[l-1]] for a in As[l]]))
print("\nL0 ln1_w quantiles:",np.quantile(W["ln1_w.0"],[0,.05,.1,.25,.5,.75,.9,.95,1]))
print("L0 per-head raw Q,K,V Frobenius norms:")
for j in range(6):
    s=slice(j*64,(j+1)*64)
    print(j,[round(float(np.linalg.norm(W[f"{name}.0"][s])),4) for name in ["W_q","W_k","W_v"]])
```
