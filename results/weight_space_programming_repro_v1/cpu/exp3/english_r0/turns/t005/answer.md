```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits
threadpool_limits(4)
np.set_printoptions(precision=4,suppress=True,linewidth=180)

def eff(a,w):
    a=a*w
    return a-a.mean(axis=1,keepdims=True)

M,A,bias={},{},{}
for l in range(4):
    w=W[f"ln1_w.{l}"].astype(float)
    b=W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        z=slice(64*j,64*(j+1))
        q,k,v=[W[f"W_{n}.{l}"][z].astype(float) for n in ["q","k","v"]]
        qe,ke,ve=[eff(t,w) for t in [q,k,v]]
        M[l,j]=qe.T@ke/8
        A[l,j]=W[f"W_o.{l}"][:,z].astype(float)@ve
        bias[l,j]=(q@b)@ke/8

template=sum(M[1,j]/np.linalg.norm(M[1,j]) for j in range(6))
template/=np.linalg.norm(template)
common=sum(A[0,j] for j in [0,1,4,5])
u,s,vh=svd(common,full_matrices=False)
R=vh[:3].T
B=u[:3] if False else u[:,:3]

print("First-layer QK restricted to dominant input subspace of shared broad-head OV")
for j in range(6):
    m=M[0,j]
    print(f"L0H{j}",
          "shared_input_fraction",round(np.linalg.norm(R.T@m@R)**2/np.linalg.norm(m)**2,4),
          "QK_3x3",np.round(R.T@m@R,4).tolist(),
          "linear_key_term",np.round(bias[0,j]@R,4).tolist())
print("\nLater-head similarity to common L1 QK structure")
for l in range(1,4):
    for j in range(6):
        print(f"L{l}H{j}",
              "QK_template_cos",round(np.sum(M[l,j]*template)/np.linalg.norm(M[l,j]),4),
              "key_bias_norm",round(np.linalg.norm(bias[l,j]),4),
              "QK_trace/nuclear",round(np.trace(M[l,j])/svdvals(M[l,j]).sum(),4))

print("\nPermutation controls for L0H2 -> L3 head key composition")
rng=np.random.default_rng(1234)
a=A[0,2]
den_a=np.linalg.norm(a)
for j in range(6):
    m=M[3,j]
    actual=np.trace(m@a)/(np.linalg.norm(m)*den_a)
    null=[]
    for k in range(500):
        perm=rng.permutation(384)
        # Permute only the write-space coordinates: destroy circuit alignment
        # without running any layer or constructing network inputs.
        null.append(np.einsum("ij,ji->",m,a[perm])/(np.linalg.norm(m)*den_a))
    null=np.array(null)
    print(f"L3H{j}","normalized_trace",round(actual,5),
          "null_mean",round(null.mean(),5),"null_sd",round(null.std(),5),
          "z",round((actual-null.mean())/null.std(),2))

print("\nTop singular-vector localization of W_proj.3")
u,s,vh=svd(W["W_proj.3"].astype(float),full_matrices=False)
for k in range(35):
    coeff=vh[k]
    kurt=len(coeff)*np.sum(coeff**4)-3
    loc=np.sum(u[:,k]**4)
    print(k+1,"s",round(s[k],3),"column_loading_excess_kurtosis",round(kurt,3),
          "residual_direction_IPR",round(loc,4))
```
