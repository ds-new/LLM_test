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
g=np.asarray(W["ln1_w.0"],float)
B=[];VV=[]
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    v=(W["W_v.0"][sl]*g)@P
    VV.append(v)
    B.append(np.asarray(W["W_o.0"][:,sl],float)@v)
_,_,vt=la.svd(np.vstack([VV[1],VV[5]]),full_matrices=False)
T=vt[:30].T
F=(np.asarray(W["W_fc.0"],float)*W["ln2_w.0"])@P
fn=la.norm(F,axis=1)
U=F/fn[:,None]
n,labels=connected_components(csr_matrix(U@U.T>.90),directed=False)
records=[]
excluded=set()
for label in range(n):
    ix=np.flatnonzero(labels==label)
    if len(ix)<2: continue
    c=U[ix].mean(axis=0);c/=la.norm(c)
    if la.norm(c@T)>.4:
        records.append((int(ix[0]),ix,c))
        excluded.update(ix.tolist())
records.sort(key=lambda x:x[0])
C=np.array([r[2] for r in records])
ids=[r[0] for r in records]
print("Candidate character-feature IDs (unlabeled):",ids)
print("Singular values of their direction matrix:",la.svdvals(C))
D=la.pinv(C)

# Only products of trained weight matrices and their derived directions:
# no LayerNorm evaluation, attention computation, or ReLU.
X=F@D
Y=F@B[5]@D
Z=F@B[1]@D
for name,M in [("direct",X),("via_L0H5_OV",Y),("via_L0H1_OV",Z)]:
    spread=M-M.mean(axis=1,keepdims=True)
    purity=np.sort(spread,axis=1)[:,-1]/(la.norm(spread,axis=1)+1e-30)
    print(name,"top-direction concentration quantiles",np.quantile(purity,[.1,.25,.5,.75,.9,.99]))
print("\nPer-feature coefficient strength and duplicate-component geometry")
for k,(fid,ix,c) in enumerate(records):
    print(fid,"copies",len(ix),"input_norm",round(float(np.median(fn[ix])),3),
          "output_norm",round(float(la.norm(W["W_proj.0"][:,ix])),3),
          "direct_column_norm",round(float(la.norm(X[:,k])),3),
          "previous_column_norm",round(float(la.norm(Y[:,k])),3))

for cutoff in (.35,.45,.55):
    M=X-X.mean(axis=1,keepdims=True)
    N=Y-Y.mean(axis=1,keepdims=True)
    px=M.max(axis=1)/(la.norm(M,axis=1)+1e-30)
    py=N.max(axis=1)/(la.norm(N,axis=1)+1e-30)
    sel=(px>cutoff)&(py>cutoff)
    sel[list(excluded)]=False
    a=M.argmax(axis=1); b=N.argmax(axis=1)
    counts=np.zeros((len(C),len(C)))
    for i in np.flatnonzero(sel):
        counts[b[i],a[i]]+=1
    total=counts.sum()
    print("\nWeight-feature pairing graph cutoff",cutoff,"neurons",int(total))
    print("Largest predecessor/current pair counts:",
          [(ids[i],ids[j],int(counts[i,j])) for i,j in
           [np.unravel_index(t,counts.shape) for t in np.argsort(-counts.ravel())[:30]] if counts[i,j]>0])
    print("Diagonal counts:",[(ids[i],int(counts[i,i])) for i in range(len(C)) if counts[i,i]>0])
    sym=counts+counts.T
    deg=sym.sum(axis=1)
    S=sym/np.sqrt(np.maximum(deg[:,None]*deg[None,:],1))
    ev,evec=la.eigh(S)
    part=evec[:,0]>0
    cross=sym[np.ix_(part,~part)].sum()*2/(sym.sum()+1e-30)
    print("Normalized graph smallest eigenvalues",ev[:5],"spectral cut fraction",round(float(cross),3))
    print("Spectral partition", [ids[i] for i in np.flatnonzero(part)],
          [ids[i] for i in np.flatnonzero(~part)])
```
