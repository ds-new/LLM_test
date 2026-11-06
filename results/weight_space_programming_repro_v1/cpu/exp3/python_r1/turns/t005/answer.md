```python
import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=160)
d=384
P=np.eye(d)-np.ones((d,d))/d
g=np.asarray(W["ln1_w.0"],dtype=float)
Q=[];K=[];V=[];O=[];A=[];B=[]
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    Q.append((W["W_q.0"][sl]*g)@P)
    K.append((W["W_k.0"][sl]*g)@P)
    V.append((W["W_v.0"][sl]*g)@P)
    O.append(np.asarray(W["W_o.0"][:,sl],dtype=float))
    A.append(Q[-1].T@K[-1])
    B.append(O[-1]@V[-1])

print("L0 head value spectra: raw, and gain-scaled")
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    print("H",j,"raw",la.svdvals(W["W_v.0"][sl].astype(float))[:36])
    print("H",j,"scaled",la.svdvals(V[j])[:36])

# Position-like weight subspace: dominant left/right singular directions of the two rotational QK matrices.
posstack=np.vstack([A[1],A[1].T,A[5],A[5].T])
_,ps,pvt=la.svd(posstack,full_matrices=False)
print("\nJoint rotational-QK spectrum",ps[:40])
for r in (16,24,32):
    U=pvt[:r].T
    print("Rotational subspace rank",r)
    for j in range(6):
        _,sv,vt=la.svd(V[j],full_matrices=False)
        print("H",j,"V_energy",round(float(la.norm(V[j]@U)**2/la.norm(V[j])**2),3),
              "top_value_direction_overlaps",np.round(np.sum((vt[:3]@U)**2,axis=1),3))

print("\nShared value subspace: principal angles between H1 and H5")
for r in (20,24,27,30,36):
    _,_,a=la.svd(V[1],full_matrices=False)
    _,_,b=la.svd(V[5],full_matrices=False)
    print("rank",r,"cosines",la.svdvals(a[:r]@b[:r].T))

print("\nMLP duplicate-feature components, and backward linear coupling through the L0 heads")
f=(np.asarray(W["W_fc.0"],dtype=float)*W["ln2_w.0"])@P
fn=la.norm(f,axis=1)
unit=f/fn[:,None]
cos=unit@unit.T
n,labels=connected_components(csr_matrix(cos>.90),directed=False)
groups=sorted([np.flatnonzero(labels==i) for i in range(n)],key=lambda ix:(-len(ix),int(ix[0])))
_,_,vt=la.svd(np.vstack([V[1],V[5]]),full_matrices=False)
T=vt[:30].T
for ix in groups:
    if len(ix)<2: continue
    c=unit[ix].mean(axis=0); c/=la.norm(c)
    direct=la.norm(c@T)
    coupling=np.array([la.norm(c@bb) for bb in B])
    print("feature",int(ix[0]),"size",len(ix),"direct_char_subspace",round(direct,3),
          "OV_backward_norms",np.round(coupling,3))
```
