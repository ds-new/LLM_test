```python
import numpy as np
from scipy.linalg import svd, eigvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

def weights(l,j):
    s=slice(64*j,64*j+64)
    g=W[f"ln1_w.{l}"].astype(float)
    q=W[f"W_q.{l}"][s].astype(float)*g
    k=W[f"W_k.{l}"][s].astype(float)*g
    v=W[f"W_v.{l}"][s].astype(float)*g
    q-=q.mean(1,keepdims=True); k-=k.mean(1,keepdims=True); v-=v.mean(1,keepdims=True)
    o=W[f"W_o.{l}"][:,s].astype(float)
    return q,k,v,o

A=[]; B=[]
for l in range(4):
    for j in range(6):
        q,k,v,o=weights(l,j)
        A.append(q.T@k/8)
        B.append(o@v)

print("First-layer QK spectra and query/key alignment:")
for j in range(6):
    q,k,v,o=weights(0,j)
    u,s,vt=svd(A[j],full_matrices=False)
    ev=eigvals(A[j])
    ev=ev[np.argsort(-np.abs(ev))]
    print("\nL0H",j)
    print("QK singular values:",s[:45])
    print("QK eigenvalues:",ev[:30])
    print("Q,K singular values:",svd(q,compute_uv=False)[:16],svd(k,compute_uv=False)[:16])
    print("paired left/right singular-vector cosines:",np.diag(vt[:24]@u[:,:24]))
    print("OV eigenvalues:",eigvals(v@o)[np.argsort(-np.abs(eigvals(v@o)))][:16])

print("\nCosines among all QK matrices (6x6 layer blocks):")
for l in range(4):
    aa=np.stack([a.ravel() for a in A[6*l:6*l+6]])
    aa/=np.linalg.norm(aa,axis=1,keepdims=True)
    print("layer",l, "\n",aa@aa.T)

print("\nCross-head linear composition diagnostics: receiver QK @ earlier OV.")
print("For each receiver: largest normalized composition and largest signed trace, with source.")
for t in range(6,24):
    vals=[]
    for s in range((t//6)*6):
        c=A[t]@B[s]
        denom=np.linalg.norm(A[t])*np.linalg.norm(B[s])
        vals.append((s,np.linalg.norm(c)/denom,np.trace(c)/denom))
    print(f"L{t//6}H{t%6}",
          "norm",[(f"L{s//6}H{s%6}",round(n,4),round(tr,4)) for s,n,tr in sorted(vals,key=lambda z:-z[1])[:4]],
          "positive trace",[(f"L{s//6}H{s%6}",round(tr,4)) for s,n,tr in sorted(vals,key=lambda z:-z[2])[:3]])

print("\nLargest LayerNorm coordinates:")
for l in range(4):
    for name in ["ln1_w","ln1_b","ln2_w","ln2_b"]:
        a=W[f"{name}.{l}"]
        idx=np.argsort(-np.abs(a))[:10]
        print(l,name,list(zip(idx.tolist(),np.round(a[idx],4).tolist())))
```
