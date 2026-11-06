```json
{"verdict":"ok","rule":"","evidence":"A.append(q.T@k); B.append(o@v); la.eig(a,left=True,right=True); cos=unit@unit.T; la.svd(B[4],full_matrices=False); q@U; k@U","reason":"The code only performs permitted linear-algebra analysis of allowed weights and weight-derived directions, without running a network layer on an input or performing any search."}
```