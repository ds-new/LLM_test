```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

def cent(a):
    return a-a.mean(axis=1,keepdims=True)
def norm(a):
    return np.linalg.norm(a)

B={}
for l in (0,3):
    g=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        B[l,j]=W[f"W_o.{l}"][:,sl].astype(float) @ cent(W[f"W_v.{l}"][sl].astype(float)*g)

_,_,vt=np.linalg.svd(np.vstack([B[0,1],B[0,5]]),full_matrices=False)
C=vt[:27].T
P3=W["W_proj.3"].astype(float)
T=np.linalg.svd(P3,full_matrices=False)[0][:,:27]
D=T.T@(B[3,1]+B[3,5])@C
print("Candidate shared copying-map singular values:")
print(np.linalg.svd(D,compute_uv=False))

# Cluster the final MLP's actual output-weight columns after whitening.
# This is a weight dictionary, not reconstructed text or a network run.
Y=(T.T@P3).T
mean=Y.mean(axis=0)
Yc=Y-mean
ev,u=np.linalg.eigh(Yc.T@Yc/len(Yc))
wh=(u*(1/np.sqrt(np.maximum(ev,ev.max()*1e-5))))@u.T
unwh=(u*np.sqrt(np.maximum(ev,ev.max()*1e-5)))@u.T
Z=Yc@wh
Zn=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)

saved=None
print("\nOutput-weight dictionary fits:")
for k in (20,24,28,32):
    km=KMeans(n_clusters=k,n_init=15,random_state=100+k).fit(Zn)
    cc=km.cluster_centers_
    radii=np.linalg.norm(cc,axis=1)
    unit=cc/np.maximum(radii[:,None],1e-12)
    assigned=np.sum(Zn*unit[km.labels_],axis=1)
    sil=silhouette_score(Zn,km.labels_,sample_size=1000,random_state=0)
    counts=np.bincount(km.labels_,minlength=k)
    print("k",k,"silhouette",round(sil,4),
          "assignment-cosine quantiles",np.quantile(assigned,[.1,.5,.9]),
          "sorted counts",np.sort(counts)[::-1])
    if k==28:
        saved=(km,cc,unit,counts,assigned)

km,cc,unit,counts,assigned=saved
print("\n28-cluster center cosine quantiles:",
      np.quantile((unit@unit.T)[np.triu_indices(28,1)],[0,.1,.5,.9,1]))
print("Counts in cluster-index order:",counts)
print("Cluster concentration (mean cosine):",np.linalg.norm(cc,axis=1))

# Relate output-weight dictionary directions to early MLP read coefficients
# through the candidate copying map, entirely by matrix products.
# Regularization avoids amplification of nearly unsupported directions.
ud,sd,vtd=np.linalg.svd(D,full_matrices=False)
ridge=(.12*sd[0])**2
Dinv=(vtd.T*(sd/(sd*sd+ridge)))@ud.T
dictionary=C@Dinv@(cc@unwh).T
dictionary/=np.maximum(np.linalg.norm(dictionary,axis=0,keepdims=True),1e-12)

F=cent(W["W_fc.0"].astype(float)*W["ln2_w.0"].astype(float))
read=[F@dictionary,F@B[0,5]@dictionary,F@B[0,1]@dictionary]
read=[x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12) for x in read]
labels=[np.argmax(x,axis=1) for x in read]
print("\nEarly-MLP weight-coefficient maximum counts in that dictionary:")
for i,lab in enumerate(labels):
    print("view",i,np.bincount(lab,minlength=28))
for i in (1,2):
    tab=np.zeros((28,28),dtype=int)
    np.add.at(tab,(labels[0],labels[i]),1)
    print("view",i,"same-index fraction",np.trace(tab)/tab.sum(),
          "top pairs",sorted([(tab[a,b],a,b) for a in range(28) for b in range(28)],reverse=True)[:20])

print("\nOutput dictionary's approximate rank:")
print(np.linalg.svd(dictionary,compute_uv=False))
```
