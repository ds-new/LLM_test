```python
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
threadpool_limits(4)
d = 384
P = np.eye(d) - np.ones((d,d))/d

A, B, Q, K, V, O = {}, {}, {}, {}, {}, {}
for l in range(4):
    g = np.asarray(W[f"ln1_w.{l}"],dtype=np.float64)
    for j in range(6):
        h = (l,j)
        sl = slice(64*j,64*(j+1))
        Q[h] = (W[f"W_q.{l}"][sl]*g)@P
        K[h] = (W[f"W_k.{l}"][sl]*g)@P
        V[h] = (W[f"W_v.{l}"][sl]*g)@P
        O[h] = np.asarray(W[f"W_o.{l}"][:,sl],dtype=np.float64)
        A[h], B[h] = Q[h].T@K[h], O[h]@V[h]

def cos(a,b):
    return float(np.sum(a*b)/(la.norm(a)*la.norm(b)+1e-30))

print("L0 QK matrix cosine similarities")
print(np.round([[cos(A[0,i],A[0,j]) for j in range(6)] for i in range(6)],3))
print("L0 OV matrix cosine similarities")
print(np.round([[cos(B[0,i],B[0,j]) for j in range(6)] for i in range(6)],3))

print("\nInduction-style key-composition diagnostics.")
print("For each later head: preceding OV heads with largest trace(A_l B_earlier)/nuclear_norm(A_l B_earlier).")
print("These are matrix signatures, not attention measurements.")
for h in A:
    if h[0] == 0:
        continue
    rows = []
    for e in B:
        if e[0] >= h[0]:
            continue
        # A B = Q^T (K O) V. Nonzero eigenvalues from a 64x64 product.
        mid = K[h]@O[e]
        small = mid@V[e]@Q[h].T
        ev = la.eigvals(small)
        _, rq = la.qr(Q[h].T,mode="economic")
        _, rv = la.qr(V[e].T,mode="economic")
        sv = la.svdvals(rq@mid@rv.T)
        tr = np.trace(small)
        nuclear_ratio = tr/(sv.sum()+1e-30)
        sign = ev.real.sum()/(np.abs(ev).sum()+1e-30)
        strength = la.norm(mid)/(la.norm(K[h])*la.norm(O[e])+1e-30)
        rows.append((nuclear_ratio,e,sign,strength,la.norm(sv)))
    rows.sort(reverse=True)
    print(f"L{h[0]}H{h[1]}: " + "; ".join(
        f"L{e[0]}H{e[1]} trace/nuc={r:.3f} eigsign={s:.3f} overlap={c:.3f} norm={n:.2f}"
        for r,e,s,c,n in rows[:4]))

print("\nL0 joint query/key and value spectra")
qk0 = np.vstack([Q[0,j] for j in range(6)] + [K[0,j] for j in range(6)])
v0 = np.vstack([V[0,j] for j in range(6)])
for name,m in [("QK",qk0),("V",v0),("V_heads1_5",np.vstack([V[0,1],V[0,5]]))]:
    s = la.svdvals(m)
    print(name, "singular values:", np.round(s[:70],3))

_,_,vt = la.svd(np.vstack([V[0,1],V[0,5]]),full_matrices=False)
T = vt[:27].T
print("\nFraction of L0 Q/K/V squared norm in the 27 leading shared value directions of H1,H5")
for j in range(6):
    print(f"L0H{j}",*[f"{name}={la.norm(m@T)**2/la.norm(m)**2:.3f}"
                      for name,m in [("Q",Q[0,j]),("K",K[0,j]),("V",V[0,j])]])
print("\nL0 QK within and outside that value subspace")
R = np.eye(d)-T@T.T
for j in range(6):
    a=A[0,j]
    tt=T.T@a@T
    rr=R@a@R
    print(f"L0H{j}: total={la.norm(a):.3f} tokenblock={la.norm(tt):.3f} otherblock={la.norm(rr):.3f}"
          f" token_sym={cos(tt,tt.T):.3f} other_sym={cos(rr,rr.T):.3f}"
          f" other_eigs={np.round(la.eigvals(rr)[np.argsort(-np.abs(la.eigvals(rr)))[:12]],3)}")
```
