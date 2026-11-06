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
g=W["ln1_w.0"].astype(float)
V=[];B=[];A=[]
for j in range(6):
    sl=slice(64*j,64*(j+1))
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
char=[];other=[]
for label in range(n):
    ix=np.flatnonzero(labels==label)
    if len(ix)<2: continue
    c=U[ix].mean(axis=0);c/=la.norm(c)
    o=W["W_proj.0"][:,ix].astype(float).mean(axis=1);o/=la.norm(o)
    rec=(int(ix[0]),len(ix),c,o)
    (char if la.norm(c@T)>.4 else other).append(rec)
char.sort(key=lambda r:r[0]);other.sort(key=lambda r:r[0])
ci=np.array([r[2] for r in char]); co=np.array([r[3] for r in char])
print("Other duplicate-feature groups: closest input and output character-like groups")
for fid,count,c,o in other:
    si=ci@c;so=co@o
    ii=np.argsort(-np.abs(si))[:3];oi=np.argsort(-np.abs(so))[:3]
    print("group",fid,"size",count,"input_output_cos",round(float(c@o),3),
          "input_matches",[(char[i][0],round(float(si[i]),3)) for i in ii],
          "output_matches",[(char[i][0],round(float(so[i]),3)) for i in oi])

print("\nRotation-phase consistency: H1 compared with integer multiples of H5's phase")
ev,vl,vr=la.eig(A[5],left=True,right=True)
data=[]
for ix in np.argsort(-np.abs(ev)):
    if ev[ix].imag<=1e-5 or abs(ev[ix])<0.8: continue
    r=vr[:,ix];ll=vl[:,ix]
    z=(ll.conj()@A[1]@r)/(ll.conj()@r)
    align=abs(np.vdot(A[1]@r,r))/(la.norm(A[1]@r)*la.norm(r)+1e-30)
    data.append((abs(ev[ix]),np.angle(ev[ix]),np.angle(z),align))
print("Rows: H5 magnitude, H5 phase, H1 projected phase, collinearity")
print(np.array(data))
for skip in (0,1):
    rows=data[skip:]
    print("Exclude largest mode pair:",bool(skip))
    for multiple in (1,2,3,4):
        scores=[np.cos(b-multiple*a) for mag,a,b,align in rows]
        weights=[mag*align for mag,a,b,align in rows]
        print("phase multiple",multiple,"weighted cosine",round(float(np.average(scores,weights=weights)),3))

print("\nMagnitude and positional-mode concentration of the two dominant input value directions of L0H4")
_,_,pvt=la.svd(np.vstack([A[1],A[1].T,A[5],A[5].T]),full_matrices=False)
_,s,v4=la.svd(V[4],full_matrices=False)
for i in range(3):
    r=v4[i]
    print("mode",i,"singular_value",round(float(s[i]),3),
          "rotational_QK_subspace_energy",round(float(la.norm(r@pvt[:24].T)**2),3),
          "character_detector_subspace_energy",round(float(la.norm(r@la.orth(ci.T))**2),3),
          "top_char_detector_cosines",[(char[k][0],round(float((ci@r)[k]),3)) for k in np.argsort(-np.abs(ci@r))[:4]])
```
