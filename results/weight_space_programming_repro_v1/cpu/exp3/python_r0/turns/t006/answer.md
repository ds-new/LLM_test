```python
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)

def cent(a):
    return a-a.mean(axis=1,keepdims=True)
def norm(a):
    return np.linalg.norm(a)
def cosine(a,b):
    return np.sum(a*b)/max(norm(a)*norm(b),1e-30)

A={};B={};kb={}
for l in range(4):
    g=W[f"ln1_w.{l}"].astype(float)
    beta=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl=slice(64*j,64*(j+1))
        qr=W[f"W_q.{l}"][sl].astype(float)
        kr=W[f"W_k.{l}"][sl].astype(float)
        q=cent(qr*g); k=cent(kr*g)
        v=cent(W[f"W_v.{l}"][sl].astype(float)*g)
        A[l,j]=q.T@k/8
        B[l,j]=W[f"W_o.{l}"][:,sl].astype(float)@v
        kb[l,j]=(qr@beta)@k/8

_,_,vt=np.linalg.svd(np.vstack([B[0,1],B[0,5]]),full_matrices=False)
C=vt[:27].T
U=np.linalg.svd(B[0,4],full_matrices=False)[0][:,:2]
ref=sum(kb[1,j]/norm(kb[1,j]) for j in range(6))
ref=U@(U.T@ref)
ref/=norm(ref)
T=np.linalg.svd(W["W_proj.3"].astype(float),full_matrices=False)[0][:,:27]

print("Shared low-dimensional position-path coefficients.")
print("Columns: key-bias norm; fraction in L0H4 output plane; cosine with L1 reference;")
print("signed reference coefficient / QK norm; QK key energy in that plane.")
for l in range(1,4):
    for j in range(6):
        b=kb[l,j]; a=A[l,j]
        print("L%dH%d" % (l,j),np.round([
            norm(b),norm(U.T@b)**2/max(norm(b)**2,1e-30),
            cosine(b,ref),np.dot(b,ref)/norm(a),
            norm(a@U)**2/norm(a)**2],4))

print("\nOV input energy in candidate character subspace and output energy in final-MLP task subspace:")
for l in range(1,4):
    for j in range(6):
        b=B[l,j]
        print("L%dH%d" % (l,j),np.round([
            norm(b@C)**2/norm(b)**2,
            norm(T.T@b)**2/norm(b)**2,
            norm(T.T@b@C)**2/max(norm(b@C)**2,1e-30)],4))

print("\nOV cosine among candidate copying heads and controls:")
hs=[(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),
    (3,0),(3,1),(3,2),(3,3),(3,4),(3,5)]
print("Order:",hs)
print(np.array([[cosine(B[x],B[y]) for y in hs] for x in hs]))

print("\nRobustness of current-to-L0H5 key matching under alternative source subspaces.")
g=W["ln1_w.0"].astype(float)
for exponent in (-1,0,1):
    stack=[]
    for j in (1,5):
        sl=slice(64*j,64*(j+1))
        factor=np.maximum(g,.04)**exponent
        vv=cent(W["W_v.0"][sl].astype(float)*factor)
        stack.append(W["W_o.0"][:,sl].astype(float)@vv)
    _,_,vv=np.linalg.svd(np.vstack(stack),full_matrices=False)
    for rank in (20,24,27,32):
        c=vv[:rank].T
        vals=[]
        for h in ((2,1),(2,2),(3,0),(3,1),(3,5)):
            m=c.T@A[h]@B[0,5]@c
            vals.append(np.trace(m)/max(norm(m),1e-30))
        print("gamma exponent",exponent,"rank",rank,
              "heads L2H1,L2H2,L3H0,L3H1,L3H5:",np.round(vals,3))

# A distribution-free upper bound using only coefficient norms and the
# centered-normalized residual's norm bound, not a LayerNorm evaluation.
a=A[0,4]; b=kb[0,4]
smax=np.linalg.svd(a,compute_uv=False)[0]
score_range_bound=2*384*smax+2*np.sqrt(384)*norm(b)
max_mass_128=1/(1+127*np.exp(-score_range_bound))
print("\nL0H4 coefficient-norm bound:")
print("QK operator norm",smax,"key-bias norm",norm(b),
      "maximum score range bound",score_range_bound,
      "single-key attention upper bound at 128 positions",max_mass_128)
```
