```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4,suppress=True,linewidth=180)

def eff(a,w):
    a=a*w
    return a-a.mean(axis=1,keepdims=True)

M,A,Bias={},{},{}
for l in range(4):
    w=W[f"ln1_w.{l}"].astype(float)
    b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q,k,v=[W[f"W_{n}.{l}"][s].astype(float) for n in ["q","k","v"]]
        qe,ke,ve=[eff(z,w) for z in [q,k,v]]
        M[l,j]=qe.T@ke/8
        A[l,j]=W[f"W_o.{l}"][:,s].astype(float)@ve
        Bias[l,j]=(q@b)@ke/8

u,ss,vh=svd(np.concatenate([A[0,j] for j in [0,1,4,5]],axis=1),full_matrices=False)
B=u[:,:3]
print("Joint output spectrum, L0 H0/H1/H4/H5",ss[:15])
print("Shared three-dimensional write basis: dominant coordinates")
for t in range(3):
    ix=np.argsort(abs(B[:,t]))[-12:][::-1]
    print(t,[(int(i),round(B[i,t],3)) for i in ix])
print("\nHead QK/linear-key-bias energy in that output subspace")
for l in range(1,4):
    for j in range(6):
        m,b=M[l,j],Bias[l,j]
        print(f"L{l}H{j}",
              "QK_both",round(np.linalg.norm(B.T@m@B)**2/np.linalg.norm(m)**2,3),
              "QK_query",round(np.linalg.norm(B.T@m)**2/np.linalg.norm(m)**2,3),
              "QK_key",round(np.linalg.norm(m@B)**2/np.linalg.norm(m)**2,3),
              "bias",round(np.linalg.norm(b@B)**2/np.linalg.norm(b)**2,3),
              "projected_QK",np.round(B.T@m@B,3).tolist(),
              "projected_key_bias",np.round(b@B,3).tolist())

print("\nFull key-composition signed trace/nuclear norm, rows=later head, cols=earlier heads")
for l in range(1,4):
    print("layer",l,"columns",[(p,h) for p in range(l) for h in range(6)])
    for j in range(6):
        vals=[]
        for p in range(l):
            for h in range(6):
                c=M[l,j]@A[p,h]
                vals.append(np.trace(c)/svdvals(c).sum())
        print(f"L{l}H{j}",np.round(vals,3))

print("\nFirst-layer content-carrying map spectra")
for name in ["W_v","W_o"]:
    a=W[f"{name}.0"].astype(float)
    if name=="W_v":
        a=eff(a[128:192],W["ln1_w.0"].astype(float))
    else:
        a=a[:,128:192]
    print(name,"H2",np.round(svdvals(a),4))
print("L0H2 effective OV",np.round(svdvals(A[0,2])[:64],4))

print("\nFinal MLP output spectrum ratios (index is 1-based)")
s=svdvals(W["W_proj.3"].astype(float))
print([(i+1,round(s[i],3),round(s[i]/s[i+1],3)) for i in range(18,35)])
```
