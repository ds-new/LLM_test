```python
import numpy as np
from scipy.linalg import eigvals, svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(2)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

A, B, KL = [], [], []
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float); b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s=slice(64*j,64*(j+1))
        Q=W[f"W_q.{l}"][s].astype(float)
        K=W[f"W_k.{l}"][s].astype(float)
        V=W[f"W_v.{l}"][s].astype(float)
        O=W[f"W_o.{l}"][:,s].astype(float)
        q=Q*g; k=K*g; v=V*g
        q-=q.mean(1,keepdims=True)
        k-=k.mean(1,keepdims=True)
        v-=v.mean(1,keepdims=True)
        A.append(q.T@k/8); B.append(O@v)
        KL.append(k.T@(Q@b)/8)

print("Key-composition trace cosine: tr(QK_l OV_earlier)/(|QK_l| |OV_earlier|).")
print("These are matrix circuit diagnostics, not attention or activation measurements.")
for l in range(1,4):
    tab=np.array([[np.sum(A[6*l+j]*B[i].T)/(np.linalg.norm(A[6*l+j])*np.linalg.norm(B[i]))
                   for i in range(6*l)] for j in range(6)])
    print("Layer",l,"vs heads of preceding layers in L,H order:\n",tab)

print("\nDetailed composites through L0H2 (candidate positional transport head):")
for h in range(6,24):
    C=A[h]@B[2]
    e=eigvals(C)
    sv=svdvals(C)
    print("L%dH%d F=%.4f s1=%.4f trace=%.4f trace/sum|eig|=%.4f symmetry=%.4f" %
          (h//6,h%6,np.linalg.norm(C),sv[0],np.trace(C),
           np.trace(C)/(np.abs(e).sum()+1e-30),
           np.sum(C*C.T)/(np.sum(C*C)+1e-30)))

print("\nLayer-0 OV singular spectra and first output-direction alignments:")
outdirs=[]
indirs=[]
for j in range(6):
    u,s,vh=np.linalg.svd(B[j],full_matrices=False)
    outdirs.append(u[:,0]); indirs.append(vh[0])
    print("H%d:"%j,s[:20])
print("Output top-direction cosine:\n",np.array(outdirs)@np.array(outdirs).T)
print("Input top-direction cosine:\n",np.array(indirs)@np.array(indirs).T)

print("\nFraction of later QK norm explained by earlier OV output subspaces:")
for l in range(1,4):
    for j in range(6):
        a=A[6*l+j]
        stats=[]
        for h in [0,1,2,3,4,5]:
            u,s,vh=np.linalg.svd(B[h],full_matrices=False)
            r=20 if h==2 else 2
            U=u[:,:r]
            stats.append((np.linalg.norm(U.T@a)**2/np.linalg.norm(a)**2,
                          np.linalg.norm(a@U)**2/np.linalg.norm(a)**2))
        print("L%dH%d query/key fractions:"%(l,j),np.array(stats).round(3))

print("\nCosines of affine key-selection directions, within each layer:")
for l in range(4):
    k=np.array(KL[6*l:6*l+6])
    k/=np.linalg.norm(k,axis=1,keepdims=True)
    print(l,k@k.T)
```
