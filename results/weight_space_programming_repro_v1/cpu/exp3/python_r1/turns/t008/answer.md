```python
import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=160)
d=384
P=np.eye(d)-np.ones((d,d))/d
g=np.asarray(W["ln1_w.0"],float)
V=[];B=[]
for j in range(6):
    sl=slice(64*j,64*j+64)
    v=(W["W_v.0"][sl]*g)@P
    V.append(v); B.append(W["W_o.0"][:,sl].astype(float)@v)
_,_,vt=la.svd(np.vstack([V[1],V[5]]),full_matrices=False)
T=vt[:30].T
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"])@P
U=F/la.norm(F,axis=1,keepdims=True)
n,labels=connected_components(csr_matrix(U@U.T>.9),directed=False)
records=[];excluded=set()
for label in range(n):
    ix=np.flatnonzero(labels==label)
    if len(ix)<2: continue
    c=U[ix].mean(axis=0);c/=la.norm(c)
    if la.norm(c@T)>.4:
        o=W["W_proj.0"][:,ix].astype(float).mean(axis=1)
        o/=la.norm(o)
        records.append((int(ix[0]),ix,c,o))
        excluded.update(ix.tolist())
records.sort(key=lambda r:r[0])
ids=[r[0] for r in records]
C=np.array([r[2] for r in records])
E=np.array([r[3] for r in records])
print("Duplicate character-like features: input/output direction cosine and affine LN-bias contraction")
for fid,ix,c,o in records:
    bias=(W["W_fc.0"][ix].astype(float)@W["ln2_b.0"]).mean()
    print(fid,"input_output_cos",round(float(c@o),3),"ln_bias_contraction",round(float(bias),3))
print("Input/output cross-Gram diagonal:",np.diag(C@E.T))
print("Input/output cross-Gram off-diagonal RMS:",
      round(float(la.norm(C@E.T-np.diag(np.diag(C@E.T)))/np.sqrt(len(C)*(len(C)-1))),3))

for name,D in [("input_dual",la.pinv(C)),("output_directions",E.T),("negative_output_directions",-E.T)]:
    print("\nCoefficient extrema pairings for",name)
    for sign in (1,-1):
        coef=[sign*F@D,sign*F@B[5]@D]
        coef=[m-m.mean(axis=1,keepdims=True) for m in coef]
        peak=[m.argmax(axis=1) for m in coef]
        purity=[m.max(axis=1)/(la.norm(m,axis=1)+1e-30) for m in coef]
        sel=(purity[0]>.45)&(purity[1]>.45)
        sel[list(excluded)]=False
        pairs=Counter((ids[peak[1][i]],ids[peak[0][i]]) for i in np.flatnonzero(sel))
        print("sign",sign,"neurons",int(sel.sum()),"pairs",pairs.most_common(25))

print("\nDirect linear MLP weight products: trace and symmetric-part energy")
for l in range(4):
    f=(W[f"W_fc.{l}"].astype(float)*W[f"ln2_w.{l}"])@P
    m=W[f"W_proj.{l}"].astype(float)@f
    ev=la.eigvals(m)
    print(l,"trace",round(float(np.trace(m)),3),"norm",round(float(la.norm(m)),3),
          "eigsign",round(float(ev.real.sum()/np.abs(ev).sum()),3))
```
