```json
{"verdict":"ok","rule":"","evidence":"B=O@ctr(V*g); A=q.T@k/8; X=F@T; Y=F@B@T; svd(...); eigh(...); np.linalg.eigvals(...)","reason":"The code only performs permitted linear-algebra computations on allowed transformer-block weights and weight-derived directions, without evaluating network layers on inputs or conducting searches."}
```