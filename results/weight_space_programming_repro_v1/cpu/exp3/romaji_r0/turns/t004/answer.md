```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3,suppress=True,linewidth=180)

def eff(a,g):
    a=a.astype(float)*g.astype(float)
    return a-a.mean(1,keepdims=True)

q0=eff(W["W_q.0"],W["ln1_w.0"])
k0=eff(W["W_k.0"],W["ln1_w.0"])
v0=eff(W["W_v.0"],W["ln1_w.0"])
F=eff(W["W_fc.0"],W["ln2_w.0"])
s3=slice(192,256)
u,sv,vt=svd(v0[s3],full_matrices=False)
uo,so,vto=svd(W["W_o.0"][:,s3].astype(float),full_matrices=False)
print("L0H3 value singular values:",sv)
print("L0H3 output singular values:",so)
print("Value-input/output top24 subspace overlap:",svd(vt[:24]@uo[:,:24],compute_uv=False))

print("\nFirst-layer OV principal directions:")
for j in range(6):
    sl=slice(j*64,(j+1)*64)
    b=W["W_o.0"][:,sl].astype(float)@v0[sl]
    uu,ss,vv=svd(b,full_matrices=False)
    for z in range(2):
        idx=np.argsort(-np.abs(uu[:,z]))[:8]
        ix=np.argsort(-np.abs(vv[z]))[:5]
        print(f"H{j} mode{z} s={ss[z]:.3f}",
              "output coordinates",[(int(i),round(uu[i,z],3)) for i in idx],
              "input coordinates",[(int(i),round(vv[z,i],3)) for i in ix])

# Project neuron weight rows onto the main value-read and value-write subspaces.
R=vt[:28].T
S=uo[:,:28]
X=F@R
Y=F@S
print("\nNeuron weight projections: input-read and output-write subspaces.")
print("Energy fractions",np.linalg.norm(X)**2/np.linalg.norm(F)**2,np.linalg.norm(Y)**2/np.linalg.norm(F)**2)
for label,Z in [("read",X),("write",Y)]:
    Z=Z-Z.mean(0)
    Z=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    for nc in [18,22,26,28,32]:
        km=KMeans(n_clusters=nc,n_init=10,random_state=7).fit(Z)
        sil=silhouette_score(Z,km.labels_,sample_size=1000,random_state=7)
        counts=np.bincount(km.labels_,minlength=nc)
        print(label,"clusters",nc,"silhouette",round(sil,3),"sizes",np.sort(counts)[::-1])

# Affine QK coefficients: no LayerNorm or attention is evaluated.
print("\nQK key-only affine term and positional-weight coupling:")
Bs=[]
for j in [1,2,4,5]:
    sl=slice(64*j,64*j+64)
    Bs.append(W["W_o.0"][:,sl].astype(float)@v0[sl])
U,_,_=svd(np.concatenate(Bs,axis=1),full_matrices=False)
C=U[:,:4]
for l in range(4):
    g=W[f"ln1_w.{l}"]; b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl=slice(64*j,64*j+64)
        qr=W[f"W_q.{l}"][sl].astype(float)
        q=eff(qr,g)
        k=eff(W[f"W_k.{l}"][sl],g)
        A=q.T@k/8
        t=k.T@(qr@b)/8
        print(f"L{l}H{j}","key-linear norm",round(np.linalg.norm(t),3),
              "key-linear along shared4",np.round(C.T@t,3),
              "QK key-energy in shared4",round(np.linalg.norm(A@C)**2/np.linalg.norm(A)**2,3),
              "QK query-energy in shared4",round(np.linalg.norm(C.T@A)**2/np.linalg.norm(A)**2,3))
```
