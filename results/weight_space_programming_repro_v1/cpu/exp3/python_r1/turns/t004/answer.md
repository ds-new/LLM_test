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
    sl=slice(64*j,64*j+64)
    q=(W["W_q.0"][sl]*g)@P
    k=(W["W_k.0"][sl]*g)@P
    v=(W["W_v.0"][sl]*g)@P
    o=np.asarray(W["W_o.0"][:,sl],dtype=float)
    Q.append(q);K.append(k);V.append(v);O.append(o)
    A.append(q.T@k);B.append(o@v)

print("Common spectral modes of L0H5 and L0H1")
for label,sel in [("all",np.arange(d)),("gain_gt_0.5",np.flatnonzero(g>.5))]:
    a=A[5][np.ix_(sel,sel)]
    b=A[1][np.ix_(sel,sel)]
    ev,vl,vr=la.eig(a,left=True,right=True)
    print(label,"dimensions",len(sel))
    for ix in np.argsort(-np.abs(ev)):
        if ev[ix].imag<=1e-5 or abs(ev[ix])<0.08: continue
        r=vr[:,ix]; ll=vl[:,ix]
        z=(ll.conj()@b@r)/(ll.conj()@r)
        align=abs(np.vdot(b@r,r))/(la.norm(b@r)*la.norm(r)+1e-30)
        print("H5eig",np.round(ev[ix],3),"H1projection",np.round(z,3),
              "phases",round(np.angle(ev[ix]),3),round(np.angle(z),3),"H1_collinearity",round(align,3))

print("\nLayer-0 MLP high-cosine connected components")
f=(np.asarray(W["W_fc.0"],dtype=float)*W["ln2_w.0"])@P
fn=la.norm(f,axis=1)
unit=f/fn[:,None]
cos=unit@unit.T
for threshold in (.95,.9,.8):
    n,labels=connected_components(csr_matrix(cos>threshold),directed=False)
    groups=sorted([np.flatnonzero(labels==i) for i in range(n)],key=len,reverse=True)
    big=[ix for ix in groups if len(ix)>=3]
    print("threshold",threshold,"component sizes",[len(ix) for ix in big])
    if threshold != .9: continue
    centers=[]
    for ix in big[:35]:
        center=unit[ix].mean(axis=0);center/=la.norm(center)
        centers.append(center)
        out=np.asarray(W["W_proj.0"][:,ix],dtype=float)
        os=la.svdvals(out)
        print("members",ix[:14],"count",len(ix),"input_norm_median",round(float(np.median(fn[ix])),3),
              "input_coherence",round(float(la.norm(unit[ix].mean(axis=0))),3),
              "output_norm",round(float(la.norm(out)),3),"output_rank1_fraction",round(float(os[0]**2/(os@os)),3),
              "pos_gain_energy",round(float(np.sum(center[g>.5]**2)),3))
    if centers:
        centers=np.array(centers)
        print("Center cosine matrix:")
        print(np.round(centers@centers.T,2))

print("\nCommon output directions of L0H4 and their linear coupling to later QK weights")
u,s,_=la.svd(B[4],full_matrices=False)
U=u[:,:2]
print("L0H4 first two singular values",s[:2])
for l in range(1,4):
    gg=np.asarray(W[f"ln1_w.{l}"],dtype=float)
    print("Layer",l)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        q=(W[f"W_q.{l}"][sl]*gg)@P
        k=(W[f"W_k.{l}"][sl]*gg)@P
        aa=q.T@k
        print(f"H{j}: query_energy={la.norm(q@U)**2/la.norm(q)**2:.3f}"
              f" key_energy={la.norm(k@U)**2/la.norm(k)**2:.3f}"
              f" QK_right_energy={la.norm(aa@U)**2/la.norm(aa)**2:.3f}"
              f" QK_left_energy={la.norm(U.T@aa)**2/la.norm(aa)**2:.3f}"
              f" 2x2={np.round(U.T@aa@U,2).tolist()}")
```
