```python
import numpy as np
from scipy.linalg import polar
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

def cent(a):
    return a-a.mean(axis=1,keepdims=True)

g=W["ln1_w.0"].astype(float)
A=[]; B=[]
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    q=cent(W["W_q.0"][sl].astype(float)*g)
    k=cent(W["W_k.0"][sl].astype(float)*g)
    v=cent(W["W_v.0"][sl].astype(float)*g)
    A.append(q.T@k/8)
    B.append(W["W_o.0"][:,sl].astype(float)@v)
_,_,vt=np.linalg.svd(np.vstack([B[1],B[5]]),full_matrices=False)
C=vt[:27].T
P=np.eye(384)-C@C.T

print("Positional-subspace tests: these compare matrix factors, not position inputs.")
a=P@A[5]@P
u,s,vt=np.linalg.svd(a,full_matrices=False)
for r in (8,12,16,20,24,32):
    angles=np.linalg.svd(u[:,:r].T@vt[:r].T,compute_uv=False)
    print("L0H5 rank",r,"left/right subspace angle cosines:",angles)

# Use the dominant joint row/column space of L0H5's non-character block.
ev,uu=np.linalg.eigh(a@a.T+a.T@a)
for r in (12,16,20,24,32):
    D=uu[:,-r:]
    a5=D.T@a@D
    metric=(a5@a5.T+a5.T@a5)/2
    e,v=np.linalg.eigh(metric)
    G=(v*(np.maximum(e,e.max()*1e-4)**(-.25)))@v.T
    R,_=polar(G@a5@G)
    eig=np.linalg.eigvals(R)
    print("\nRank",r,"polar-factor eigenphases/pi:",np.sort(np.angle(eig)/np.pi))
    for j in (0,1,2,3):
        aj=G@(D.T@P@A[j]@P@D)@G
        scores=[]
        for power in range(-8,9):
            rp=np.linalg.matrix_power(R,power)
            c=np.sum(aj*rp)/(np.linalg.norm(aj)*np.linalg.norm(rp))
            scores.append((c,power))
        scores.sort(reverse=True)
        print("H%d top (matrix cosine, relative power):" % j,
              [(round(c,4),p) for c,p in scores[:5]])

# Redo clustering with view-specific centering and bounded row magnitudes,
# to distinguish token structure from differences in weight scales/means.
f=cent(W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float))
xs=[f@C,f@B[5]@C,f@B[1]@C]
zs=[]
for x in xs:
    x=x-x.mean(axis=0)
    norms=np.linalg.norm(x,axis=1)
    x=x/np.maximum(norms[:,None],np.quantile(norms,.25))
    zs.append(x)
joint=np.vstack(zs)
e,v=np.linalg.eigh(joint.T@joint)
wh=(v*(np.maximum(e,e.max()*1e-3)**(-.5)))@v.T
z=joint@wh
z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
km=KMeans(n_clusters=28,n_init=15,random_state=7).fit(z)
ct=km.cluster_centers_
ct/=np.linalg.norm(ct,axis=1,keepdims=True)
lab=km.labels_.reshape(3,-1)
sim=np.sum(z*ct[km.labels_],axis=1).reshape(3,-1)
print("\nView-centered weight-row clusters:")
for i in range(3):
    print("view",i,"cosine quantiles",np.quantile(sim[i],[.1,.5,.9]),
          "counts",np.bincount(lab[i],minlength=28))
for i in (1,2):
    mask=(sim[0]>.5)&(sim[i]>.5)
    tab=np.zeros((28,28),dtype=int)
    np.add.at(tab,(lab[0,mask],lab[i,mask]),1)
    print("view",i,"clear rows",mask.sum(),"same-cluster fraction",
          np.trace(tab)/max(tab.sum(),1))
    print("largest pairs",sorted([(tab[a,b],a,b) for a in range(28) for b in range(28)],reverse=True)[:20])
```
