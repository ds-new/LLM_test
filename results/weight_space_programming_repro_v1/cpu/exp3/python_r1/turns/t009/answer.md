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
g=W["ln1_w.0"].astype(float)
V=[];B=[];A=[]
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    v=(W["W_v.0"][sl]*g)@P
    V.append(v)
    B.append(W["W_o.0"][:,sl].astype(float)@v)
    q=(W["W_q.0"][sl]*g)@P
    k=(W["W_k.0"][sl]*g)@P
    A.append(q.T@k)
_,_,vt=la.svd(np.vstack([V[1],V[5]]),full_matrices=False)
T=vt[:30].T
F=(W["W_fc.0"].astype(float)*W["ln2_w.0"])@P
U=F/la.norm(F,axis=1,keepdims=True)
n,labels=connected_components(csr_matrix(U@U.T>.9),directed=False)
records=[]
for label in range(n):
    ix=np.flatnonzero(labels==label)
    if len(ix)<2: continue
    c=U[ix].mean(axis=0); c/=la.norm(c)
    if la.norm(c@T)>.4:
        records.append((int(ix[0]),c))
records.sort()
ids=[r[0] for r in records]
C=np.array([r[1] for r in records])
D=la.pinv(C)
_,_,pvt=la.svd(np.vstack([A[1],A[1].T,A[5],A[5].T]),full_matrices=False)
bu,_,_=la.svd(B[4],full_matrices=False)
pos=np.column_stack([pvt[:24].T,bu[:3]])
sectors=[D,B[5]@D,B[1]@D]
coef=[]
print("Residualizing each linear weight-direction sector against the other sectors and shared positional directions.")
for i in range(3):
    other=np.column_stack([pos]+[sectors[j] for j in range(3) if j!=i])
    u,s,_=la.svd(other,full_matrices=False)
    u=u[:,s>s[0]*1e-7]
    e=sectors[i]-u@(u.T@sectors[i])
    e/=la.norm(e,axis=0,keepdims=True)
    m=F@e
    m-=m.mean(axis=1,keepdims=True)
    coef.append(m)
    print("sector",i,"coefficient Frobenius norm",round(float(la.norm(m)),3))

peaks=[m.argmax(axis=1) for m in coef]
purity=[m.max(axis=1)/(la.norm(m,axis=1)+1e-30) for m in coef]
for cutoff in (.4,.5,.6):
    sel=(purity[0]>cutoff)&(purity[1]>cutoff)
    pairs=Counter((ids[peaks[1][i]],ids[peaks[0][i]]) for i in np.flatnonzero(sel))
    print("\ncutoff",cutoff,"selected",int(sel.sum()),"pairs",pairs.most_common(35))
    sel3=sel&(purity[2]>cutoff)
    triples=Counter((ids[peaks[2][i]],ids[peaks[1][i]],ids[peaks[0][i]]) for i in np.flatnonzero(sel3))
    print("triples",triples.most_common(20))
    counts=np.zeros((len(ids),len(ids)))
    for (a,b),v in pairs.items():
        counts[ids.index(a),ids.index(b)]=v
    sym=counts+counts.T
    deg=sym.sum(axis=1)
    normalized=sym/np.sqrt(np.maximum(deg[:,None]*deg[None,:],1))
    ev,vec=la.eigh(normalized)
    part=vec[:,0]>0
    print("smallest normalized eigenvalues",ev[:4],
          "cut fraction",round(float(2*sym[np.ix_(part,~part)].sum()/(sym.sum()+1e-30)),3))
    print("parts",[ids[i] for i in np.flatnonzero(part)],[ids[i] for i in np.flatnonzero(~part)])

print("\nStrongest first-layer neuron coefficient patterns after sector residualization")
strength=np.array([la.norm(m,axis=1) for m in coef])
quality=np.minimum(purity[0],purity[1])*np.minimum(strength[0],strength[1])
for i in np.argsort(-quality)[:30]:
    top=[]
    for m in coef:
        inds=np.argsort(-m[i])[:3]
        top.append([(ids[k],round(float(m[i,k]),2)) for k in inds])
    print("neuron",int(i),"direct/previous/local",top)
```
