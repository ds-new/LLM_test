```python
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
threadpool_limits(limits=3)

d=384
P=np.eye(d)-np.ones((d,d))/d
A={}; B={}; Q={}; K={}; V={}; O={}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl=slice(j*64,(j+1)*64)
        Q[l,j]=(W[f"W_q.{l}"][sl].astype(float)*g)@P
        K[l,j]=(W[f"W_k.{l}"][sl].astype(float)*g)@P
        V[l,j]=(W[f"W_v.{l}"][sl].astype(float)*g)@P
        O[l,j]=W[f"W_o.{l}"][:,sl].astype(float)
        A[l,j]=Q[l,j].T@K[l,j]/8
        B[l,j]=O[l,j]@V[l,j]

def cos(a,b):
    return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30)
def basis(a,r):
    return svd(a,full_matrices=False)[0][:,:r]

print("Within-layer QK matrix cosines:")
for l in range(4):
    print("L",l)
    print(np.round([[cos(A[l,i],A[l,j]) for j in range(6)] for i in range(6)],3))

pos=basis(np.concatenate([Q[0,3].T,K[0,3].T],axis=1),32)
vv=np.concatenate([V[0,j] for j in range(6)],axis=0)
tok=basis((vv@(np.eye(d)-pos@pos.T)).T,27)
print("\nL0 value singular values:",np.round(svd(vv,compute_uv=False)[:45],3))
print("L0H3 positional-subspace / candidate content-subspace fractions:")
for j in range(6):
    a=A[0,j]; b=B[0,j]
    print("L0H%d"%j,"QK both-pos/left-pos/right-pos/content",
          np.round([np.linalg.norm(pos.T@a@pos)/np.linalg.norm(a),
                    np.linalg.norm(pos.T@a)/np.linalg.norm(a),
                    np.linalg.norm(a@pos)/np.linalg.norm(a),
                    np.linalg.norm(tok.T@a@tok)/np.linalg.norm(a)],3),
          "OV source-pos/source-content",
          np.round([np.linalg.norm(b@pos)/np.linalg.norm(b),
                    np.linalg.norm(b@tok)/np.linalg.norm(b)],3))

u5,s5,vt5=svd(B[0,5],full_matrices=False)
print("\nL0H5 OV top singular values:",np.round(s5[:12],3))
print("Top 4 right singular vectors' squared positional/content projection:")
for i in range(4):
    z=vt5[i]
    print(i,round(np.linalg.norm(pos.T@z)**2,4),round(np.linalg.norm(tok.T@z)**2,4))

print("\nLater QK energy on L0H5 top TWO output directions:")
out5=u5[:,:2]
for l in range(1,4):
    for j in range(6):
        a=A[l,j]
        qbias=W[f"W_q.{l}"][j*64:(j+1)*64].astype(float)@W[f"ln1_b.{l}"].astype(float)
        key_bias_direction=qbias@K[l,j]/8
        print("L%dH%d"%(l,j),
              "QK left/right/both fractions",np.round([
                  np.linalg.norm(out5.T@a)/np.linalg.norm(a),
                  np.linalg.norm(a@out5)/np.linalg.norm(a),
                  np.linalg.norm(out5.T@a@out5)/np.linalg.norm(a)],3),
              "bias-key norm",round(np.linalg.norm(key_bias_direction),3),
              "bias-key projection",np.round(key_bias_direction@out5,3))

print("\nMLP output spectra, positions 15..34:")
for l in range(4):
    print(l,np.round(svd(W[f"W_proj.{l}"].astype(float),compute_uv=False)[15:35],3))

print("\nL3H3 compositional diagnostics in candidate content subspace:")
for j in range(6):
    c=tok.T@A[3,j]@B[0,3]@tok
    s=svd(c,compute_uv=False)
    e=np.linalg.eigvalsh((c+c.T)/2)
    print("L3H%d"%j,"trace/nuclear=%.4f symcos=%.4f positive_symmetric_mass=%.4f"%
          (np.trace(c)/s.sum(),cos(c,c.T),e[e>0].sum()/np.abs(e).sum()),
          "top sv",np.round(s[:5],4))
```
