```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

def eff(a, w):
    z = a*w
    return z-z.mean(axis=1, keepdims=True)

M, A, QE, KE, VE = {}, {}, {}, {}, {}
for l in range(4):
    w = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s = slice(64*j,64*j+64)
        q,k,v = [eff(W[f"W_{n}.{l}"][s].astype(float),w) for n in ["q","k","v"]]
        o = W[f"W_o.{l}"][:,s].astype(float)
        QE[l,j],KE[l,j],VE[l,j] = q,k,v
        M[l,j],A[l,j] = q.T@k/8,o@v

print("L0 effective OV spectra and input/output subspace overlap with L0H2 QK")
u,s,vh = svd(M[0,2],full_matrices=False)
P = np.linalg.qr(np.concatenate([u[:,:18],vh[:18].T],axis=1))[0]
for j in range(6):
    a,m = A[0,j],M[0,j]
    print("H",j,"OV_sv",np.round(svdvals(a)[:16],4),
          "QK_fraction_in_H2_space",round(np.linalg.norm(P.T@m@P)**2/np.linalg.norm(m)**2,4),
          "OV_input_fraction_in_H2_space",round(np.linalg.norm(a@P)**2/np.linalg.norm(a)**2,4),
          "OV_output_fraction_in_H2_space",round(np.linalg.norm(P.T@a)**2/np.linalg.norm(a)**2,4))
print("L0 OV pairwise cosines")
print(np.array([[np.sum(A[0,i]*A[0,j])/np.linalg.norm(A[0,i])/np.linalg.norm(A[0,j]) for j in range(6)] for i in range(6)]))
print("L1 effective QK pairwise cosines")
print(np.array([[np.sum(M[1,i]*M[1,j])/np.linalg.norm(M[1,i])/np.linalg.norm(M[1,j]) for j in range(6)] for i in range(6)]))

print("\nL0H2 effective QK full nonzero spectrum")
print(s[:64])
print("L0H2 QK dominant coordinate energy, compared with ln1_w.0")
energy = np.sum(u[:,:18]**2,axis=1)+np.sum(vh[:18]**2,axis=0)
ix=np.argsort(energy)[-25:][::-1]
print([(int(i),round(energy[i],3),round(float(W["ln1_w.0"][i]),3)) for i in ix])

print("\nLast-layer output-space spectra")
for name in ["W_o.3","W_proj.3"]:
    a=W[name].astype(float)
    print(name,np.round(svdvals(a)[:60],3))
p=W["W_proj.3"].astype(float)
f=eff(W["W_fc.3"].astype(float),W["ln2_w.3"].astype(float)).T
alpha=np.sum(p*f)/np.sum(f*f)
print("Projection of W_proj.3 onto effective W_fc.3.T:",alpha)
for c in [0,alpha,-.1,-.2,-.4,-.8]:
    print("subtract",round(c,4),"spectrum",np.round(svdvals(p-c*f)[:40],3))

print("\nNearest-neighbor geometry of final MLP output columns")
norms=np.linalg.norm(p,axis=0)
z=p/np.maximum(norms,1e-20)
sim=z.T@z
np.fill_diagonal(sim,-np.inf)
nn=sim.max(axis=1)
print("column norm quantiles",np.quantile(norms,[0,.1,.25,.5,.75,.9,.99,1]))
print("nearest cosine quantiles",np.quantile(nn,[0,.1,.25,.5,.75,.9,.99,1]))

print("\nNormalized trace of key composition with L0H2 (all later heads)")
for l in range(1,4):
    for j in range(6):
        c=M[l,j]@A[0,2]
        print(f"L{l}H{j}","trace",round(np.trace(c),4),
              "trace/nuclear",round(np.trace(c)/svdvals(c).sum(),4),
              "query-side trace/nuclear",round(np.trace(A[0,2].T@M[l,j])/svdvals(A[0,2].T@M[l,j]).sum(),4))
```
