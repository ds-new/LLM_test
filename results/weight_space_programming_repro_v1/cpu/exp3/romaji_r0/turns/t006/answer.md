```python
import numpy as np
from scipy.linalg import svd, eig
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

# Compare independently learned final-block write subspaces.
P=W["W_proj.3"].astype(float)
O=W["W_o.3"].astype(float)
u,s,vt=svd(P,full_matrices=False)
uo,so,vto=svd(O,full_matrices=False)
print("Final-block W_o singular values:",so[:55])
print("Final-MLP singular mode: singular value; attention-write energy; earlier-MLP-write energy; overlap with attention top27")
for z in range(45):
    print(z+1,round(s[z],3),
          round(np.linalg.norm(u[:,z]@O),3),
          round(np.linalg.norm(u[:,z]@W["W_proj.2"]),3),
          round(np.linalg.norm(u[:,z]@uo[:,:27])**2,3))
print("Principal-angle cosines, final MLP vs final attention, top27:",svd(u[:,:27].T@uo[:,:27],compute_uv=False))

def eff(name,l):
    a=W[f"{name}.{l}"].astype(float)*W[f"ln1_w.{l}"].astype(float)
    return a-a.mean(1,keepdims=True)
q=eff("W_q",0); k=eff("W_k",0)
As=[q[64*j:64*j+64].T@k[64*j:64*j+64]/8 for j in range(6)]

# Inspect common positional rotation modes without constructing any input.
basis=svd(np.concatenate([q[192:256],k[192:256]],axis=0),full_matrices=False)[2][:32].T
a3=basis.T@As[3]@basis
a0=basis.T@As[0]@basis
ev,R=eig(a3)
L=np.linalg.inv(R)
c=L@a0@R
ix=[i for i in np.argsort(-np.abs(ev)) if ev[i].imag>1e-5]
print("\nL0H3 eigenmodes and L0H0 coefficient in the same biorthogonal basis:")
for i in ix:
    print("H3",ev[i],"phase",round(np.angle(ev[i]),3),"H0",c[i,i],"phase",round(np.angle(c[i,i]),3))
print("H0 diagonal energy fraction in this (nonorthogonal) eigenbasis:",
      np.sum(np.abs(np.diag(c))**2)/np.sum(np.abs(c)**2))

print("\nFirst-layer QK dominant-mode location in the H3-derived subspace:")
for j in range(6):
    a=As[j]
    uu,ss,vv=svd(a,full_matrices=False)
    print("H",j,"top6 s",ss[:6],
          "left overlaps",np.sum((basis.T@uu[:,:6])**2,axis=0),
          "right overlaps",np.sum((basis.T@vv[:6].T)**2,axis=0))
```
