```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
np.set_printoptions(precision=4, suppress=True, linewidth=160)

def cent(a):
    return a - a.mean(axis=1, keepdims=True)

def norm(a):
    return np.linalg.norm(a)

def cos(a, b):
    return np.sum(a*b) / max(norm(a)*norm(b), 1e-30)

A, B, kb = {}, {}, {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s = slice(j*64, (j+1)*64)
        q0 = W[f"W_q.{l}"][s].astype(float)
        k0 = W[f"W_k.{l}"][s].astype(float)
        q, k = cent(q0*g), cent(k0*g)
        v = cent(W[f"W_v.{l}"][s].astype(float)*g)
        o = W[f"W_o.{l}"][:, s].astype(float)
        A[l,j] = q.T @ k / 8
        B[l,j] = o @ v
        kb[l,j] = (q0 @ b) @ k / 8

# Candidate shared character-reading subspace, derived only from value weights.
_, ss, vt = np.linalg.svd(np.vstack([B[0,1], B[0,5]]), full_matrices=False)
C = vt[:27].T
print("Joint L0H1/L0H5 OV input singular values:", ss[:40])
print("\nL0 QK energy in candidate character subspace (query/key/both); OV input energy there")
for j in range(6):
    a, b = A[0,j], B[0,j]
    print("L0H%d" % j,
          np.round([norm(C.T@a)**2/norm(a)**2,
                    norm(a@C)**2/norm(a)**2,
                    norm(C.T@a@C)**2/norm(a)**2,
                    norm(b@C)**2/norm(b)**2], 4))

print("\nL0 pairwise QK Frobenius cosine")
print(np.array([[cos(A[0,i], A[0,j]) for j in range(6)] for i in range(6)]))
print("L0 pairwise OV Frobenius cosine")
print(np.array([[cos(B[0,i], B[0,j]) for j in range(6)] for i in range(6)]))

print("\nL1 pairwise QK cosine")
print(np.array([[cos(A[1,i], A[1,j]) for j in range(6)] for i in range(6)]))
print("L1 key-bias cosine")
print(np.array([[cos(kb[1,i], kb[1,j]) for j in range(6)] for i in range(6)]))

print("\nPotential key-composition circuits A_lh @ B_previous.")
print("Each entry: previous_head, trace/F, symmetric_cosine, composition_strength.")
print("Positive trace and symmetry are only circuit clues, not measured attention.")
for l in range(1,4):
    for j in range(6):
        entries = []
        for m in range(l):
            for i in range(6):
                t = A[l,j] @ B[m,i]
                nt = norm(t)
                tr = np.trace(t)/max(nt, 1e-30)
                sy = cos(t, t.T)
                strength = np.sqrt(384)*nt/max(norm(A[l,j])*norm(B[m,i]), 1e-30)
                entries.append((tr, m, i, sy, strength))
        entries.sort(reverse=True)
        best = ["L%dH%d %.3f %.3f %.2f" % (m,i,tr,sy,strength)
                for tr,m,i,sy,strength in entries[:5]]
        print("L%dH%d:" % (l,j), "; ".join(best))

print("\nL0 character-subspace key-composition scores for all later heads")
print("Columns: upstream L0H0..H5; trace(C.T @ A @ B @ C) / Frobenius")
for l in range(1,4):
    for j in range(6):
        vals = []
        for i in range(6):
            t = C.T @ A[l,j] @ B[0,i] @ C
            vals.append(np.trace(t)/max(norm(t), 1e-30))
        print("L%dH%d" % (l,j), np.round(vals, 3))

print("\nL0 QK split between symmetric and skew-symmetric parts")
for j in range(6):
    a = A[0,j]
    ev = np.linalg.eigvalsh((a+a.T)/2)
    print("L0H%d sym_extremes =" % j, np.r_[ev[:5], ev[-5:]],
          "skew_energy=%.4f" % (norm((a-a.T)/2)**2/norm(a)**2))

print("\nL1 key-bias alignment to L0 OV output spaces (top two directions)")
for i in range(6):
    u, s, _ = np.linalg.svd(B[0,i], full_matrices=False)
    vals = [norm(u[:,:2].T@kb[1,j])**2/max(norm(kb[1,j])**2, 1e-30)
            for j in range(6)]
    print("L0H%d output:" % i, np.round(vals, 4))
```
