```json
{"verdict":"ok","rule":"","evidence":"q, k, v = q @ P, k @ P, v @ P; A[l,j] = q.T @ k / 8; c = A[l,j] @ B[m,k]; svd(...); eigvals(a)","reason":"The code performs only permitted linear-algebra computations on allowed block weights, without executing any network layer on inputs or conducting a search."}
```