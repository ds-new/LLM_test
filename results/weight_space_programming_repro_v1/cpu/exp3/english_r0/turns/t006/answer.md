```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

# All operations below concern weight matrices and their singular subspaces.
p = W["W_proj.3"].astype(float)
u, s, vh = svd(p, full_matrices=False)
print("Leading final-MLP output direction: largest coordinates")
ix = np.argsort(np.abs(u[:, 0]))[-12:][::-1]
print([(int(i), round(float(u[i, 0]), 4)) for i in ix])

# Remove coordinate-localized directions, rather than assuming every large
# singular value corresponds to a separate character.
localized = np.flatnonzero(np.sum(u[:, :40]**4, axis=0) > 0.025)
B = u[:, localized]
clean = p - B @ (B.T @ p)
sc = svdvals(clean)
print("Removed singular-vector indices (1-based)", localized + 1)
print("Remaining leading singular values", np.round(sc[:40], 4))
print("Remaining largest adjacent ratios among first 35",
      sorted([(i+1, float(sc[i]/sc[i+1])) for i in range(35)],
             key=lambda x: x[1], reverse=True)[:8])

print("\nFirst-layer head OV energy in its leading three singular directions")
for j in range(6):
    z = slice(64*j,64*(j+1))
    v = W["W_v.0"][z].astype(float)*W["ln1_w.0"].astype(float)
    v -= v.mean(axis=1,keepdims=True)
    a = W["W_o.0"][:,z].astype(float)@v
    ss = svdvals(a)
    print(f"L0H{j}", "top3_fraction", round(float(np.sum(ss[:3]**2)/np.sum(ss**2)),4),
          "stable_rank", round(float(np.sum(ss**2)/ss[0]**2),4))
```
