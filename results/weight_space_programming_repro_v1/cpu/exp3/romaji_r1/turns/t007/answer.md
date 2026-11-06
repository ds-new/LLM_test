```python
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)

d=384
P=np.eye(d)-np.ones((d,d))/d
A={}; B={}; bias={}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    beta=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        qr=W[f"W_q.{l}"][sl].astype(float)
        kr=W[f"W_k.{l}"][sl].astype(float)
        vr=W[f"W_v.{l}"][sl].astype(float)
        q=(qr*g)@P; k=(kr*g)@P
        A[l,j]=q.T@k/8
        B[l,j]=W[f"W_o.{l}"][:,sl].astype(float)@(vr*g)@P
        bias[l,j]=(qr@beta)@k/8

g=W["ln1_w.0"].astype(float)
q=(W["W_q.0"][192:256].astype(float)*g)@P
k=(W["W_k.0"][192:256].astype(float)*g)@P
u,s,vt=svd(np.concatenate([q,k],axis=0),full_matrices=False)
print("Positional-kernel matrix-power comparisons.")
print("These compare weight matrices, not scores for any input or position.")
for r in [20,28,32,40]:
    t=vt[:r].T/s[:r]
    aa=t.T@A[0,3]@t
    ul,ss,vr=svd(aa)
    rot=ul@vr
    powers={n:np.linalg.matrix_power(rot,n) for n in range(-8,9)}
    print("rank",r,"reference singular range",np.round([ss.min(),ss.max()],4))
    for j in range(6):
        c=t.T@A[0,j]@t
        vals=np.array([np.sum(c*powers[n])/(np.linalg.norm(c)*np.sqrt(r)) for n in range(-8,9)])
        best=np.argsort(-vals)[:4]
        print("L0H%d"%j,[(int(i-8),round(float(vals[i]),3)) for i in best])

print("\nInduction robustness across independently chosen content subspaces:")
rawpos=svd(np.concatenate([W["W_q.0"][192:256].T,
                          W["W_k.0"][192:256].T],axis=1).astype(float),
           full_matrices=False)[0][:,:32]
vv=W["W_v.0"][192:256].astype(float)
_,ss,tt=svd(vv@(np.eye(d)-rawpos@rawpos.T),full_matrices=False)
for r in [16,20,23,26,32]:
    t=np.linalg.qr(P@(tt[:r].T/np.maximum(g[:,None],.08)))[0]
    vals=[]
    for j in range(6):
        c=t.T@A[3,j]@B[0,3]@t
        vals.append(np.trace(c)/svd(c,compute_uv=False).sum())
    print("rank",r,"L3H0..5",np.round(vals,4))

print("\nCommon later-layer key-bias direction:")
ref=sum(bias[1,j] for j in range(6))
ref/=np.linalg.norm(ref)
u5=svd(B[0,5],full_matrices=False)[0][:,:2]
for l in range(4):
    for j in range(6):
        a=A[l,j]
        z=bias[l,j]
        print("L%dH%d"%(l,j),
              "QK_F=%.4f bias_F=%.4f common_bias_cos=%.4f position_key_fraction=%.4f"%
              (np.linalg.norm(a),np.linalg.norm(z),z@ref/(np.linalg.norm(z)+1e-30),
               np.linalg.norm(a@u5)/np.linalg.norm(a)))

print("\nSelected directly stored parameter entries:")
for name in ["ln1_w.0","ln1_b.1","ln1_b.2","ln1_b.3"]:
    a=W[name]
    ids=np.argsort(-np.abs(a))[:6]
    print(name,[(int(i),float(a[i])) for i in ids])
```
