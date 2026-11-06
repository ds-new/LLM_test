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
Q=[];K=[];V=[];B=[];A=[]
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    q=(W["W_q.0"][sl]*g)@P
    k=(W["W_k.0"][sl]*g)@P
    v=(W["W_v.0"][sl]*g)@P
    Q.append(q);K.append(k);V.append(v)
    B.append(W["W_o.0"][:,sl].astype(float)@v)
    A.append(q.T@k)
_,_,vt=la.svd(np.vstack([V[1],V[5]]),full_matrices=False)
T=vt[:30].T
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"])@P
U=F/la.norm(F,axis=1,keepdims=True)
n,labels=connected_components(csr_matrix(U@U.T>.90),directed=False)
records=[]
excluded=set()
for label in range(n):
    ix=np.flatnonzero(labels==label)
    if len(ix)<2: continue
    c=U[ix].mean(axis=0); c/=la.norm(c)
    if la.norm(c@T)>.4:
        records.append((int(ix[0]),c))
        excluded.update(ix.tolist())
records.sort()
ids=[r[0] for r in records]
C=np.array([r[1] for r in records])
_,_,pvt=la.svd(np.vstack([A[1],A[1].T,A[5],A[5].T]),full_matrices=False)
R=np.eye(d)-pvt[:16].T@pvt[:16]
_,_,cvt=la.svd(np.vstack([V[1],V[5]])@R,full_matrices=False)
Tc=cvt[:30].T

for name,basis in [("full",np.eye(d)),("shared_value",T),("value_without_positional_modes",Tc)]:
    D=basis@la.pinv(C@basis)
    print("\nBASIS",name,"dual direction norm range",np.round([la.norm(D,axis=0).min(),la.norm(D,axis=0).max()],3))
    coef=[F@D,F@B[5]@D,F@B[1]@D]
    coef=[m-m.mean(axis=1,keepdims=True) for m in coef]
    peaks=[m.argmax(axis=1) for m in coef]
    purity=[m.max(axis=1)/(la.norm(m,axis=1)+1e-30) for m in coef]
    sel=(purity[0]>.4)&(purity[1]>.4)
    sel[list(excluded)]=False
    pairs=Counter((ids[peaks[1][i]],ids[peaks[0][i]]) for i in np.flatnonzero(sel))
    print("Two-sector feature pairings:",pairs.most_common(25))
    sel3=sel&(purity[2]>.4)
    triples=Counter((ids[peaks[2][i]],ids[peaks[1][i]],ids[peaks[0][i]]) for i in np.flatnonzero(sel3))
    print("Three-sector feature pairings:",triples.most_common(25))
    if name!="shared_value": continue
    print("Character-feature coupling to each L0 Q/K weight matrix (largest column norms):")
    for j in range(6):
        qn=la.norm(Q[j]@D,axis=0)
        kn=la.norm(K[j]@D,axis=0)
        print("H",j,"Q",[(ids[t],round(float(qn[t]),3)) for t in np.argsort(-qn)[:7]],
              "K",[(ids[t],round(float(kn[t]),3)) for t in np.argsort(-kn)[:7]])
    print("L0 QK weight products restricted to the candidate feature directions; largest entries:")
    for j in (0,2,3,4):
        m=D.T@A[j]@D
        inds=[np.unravel_index(t,m.shape) for t in np.argsort(-np.abs(m).ravel())[:12]]
        print("H",j,[(ids[i],ids[k],round(float(m[i,k]),3)) for i,k in inds])
```
