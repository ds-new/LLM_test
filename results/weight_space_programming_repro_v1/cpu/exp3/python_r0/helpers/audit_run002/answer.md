```json
{"verdict":"ok","rule":"","evidence":"A[l,j] = q.T @ k / 8; B[l,j] = o @ v; kb[l,j] = (q0 @ b) @ k / 8; np.linalg.svd(...); t = A[l,j] @ B[m,i]","reason":"The code only performs linear-algebra analyses of permitted block weights, without executing network layers on inputs, accessing excluded weights, or searching."}
```