```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

def cent(a):
    return a - a.mean(axis=1, keepdims=True)

def norm(a):
    return np.linalg.norm(a)

A, B = {}, {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl = slice(64*j, 64*(j+1))
        q = cent(W[f"W_q.{l}"][sl].astype(float)*g)
        k = cent(W[f"W_k.{l}"][sl].astype(float)*g)
        v = cent(W[f"W_v.{l}"][sl].astype(float)*g)
        A[l,j] = q.T @ k / 8
        B[l,j] = W[f"W_o.{l}"][:,sl].astype(float) @ v

_, _, vt = np.linalg.svd(np.vstack([B[0,1], B[0,5]]), full_matrices=False)
C = vt[:27].T

print("Final MLP output singular spectrum, first 60:")
print(np.linalg.svd(W["W_proj.3"].astype(float), compute_uv=False)[:60])
print("\nL0 OV singular spectra, first 32:")
for j in range(6):
    print("L0H%d" % j, np.linalg.svd(B[0,j], compute_uv=False)[:32])

# Compare fixed coefficient blocks on weight-derived subspaces.
S = [C, B[0,5]@C, B[0,1]@C]
names = ["current", "via_L0H5", "via_L0H1"]
print("\nMatching-index matrices: tr(S_query.T A S_key) / Frobenius.")
print("Rows/columns:", names)
for l in range(1,4):
    for j in range(6):
        t = np.array([[np.trace(x.T@A[l,j]@y)/max(norm(x.T@A[l,j]@y), 1e-30)
                       for y in S] for x in S])
        print("L%dH%d" % (l,j), np.round(t,3).tolist())

# The following is a covariance of MLP weight rows, not an MLP evaluation.
f = cent(W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float))
f0 = f @ C
f5 = f @ B[0,5] @ C
f1 = f @ B[0,1] @ C

def invsqrt_psd(a, reg=1e-3):
    ev, u = np.linalg.eigh(a)
    return (u * (1/np.sqrt(np.maximum(ev, reg*ev.max())))) @ u.T

print("\nL0 MLP weight-row cross-association.")
for name, other in [("H5",f5),("H1",f1)]:
    x = f0 - f0.mean(axis=0)
    y = other - other.mean(axis=0)
    xx, yy = x.T@x, y.T@y
    pooled = invsqrt_psd((xx+yy)/2)
    cross = pooled @ (x.T@y) @ pooled
    sym = (cross+cross.T)/2
    ev = np.linalg.eigvalsh(sym)
    cc = invsqrt_psd(xx) @ (x.T@y) @ invsqrt_psd(yy)
    print(name, "symmetric eigenvalues:", ev)
    print(name, "canonical correlations:", np.linalg.svd(cc,compute_uv=False))

print("\nStrength of L0 output subspaces in subsequent QK coefficients.")
print("For each head: query/key energy fractions in each top-two L0 output space.")
spaces = [np.linalg.svd(B[0,i],full_matrices=False)[0][:,:2] for i in (0,4)]
for l in range(1,4):
    for j in range(6):
        a = A[l,j]
        vals = [(norm(u.T@a)**2/norm(a)**2, norm(a@u)**2/norm(a)**2) for u in spaces]
        print("L%dH%d H0=(%.3f,%.3f) H4=(%.3f,%.3f)" %
              (l,j,vals[0][0],vals[0][1],vals[1][0],vals[1][1]))

print("\nLargest L0 LN1 gains and corresponding matrix-column energies:")
g = W["ln1_w.0"].astype(float)
ix = np.argsort(g)[-16:][::-1]
for i in ix:
    print("coordinate %3d gamma %.4f: Q %.3f K %.3f V %.3f FC %.3f" %
          (i, g[i], norm(W["W_q.0"][:,i]), norm(W["W_k.0"][:,i]),
           norm(W["W_v.0"][:,i]), norm(W["W_fc.0"][:,i])))
```
