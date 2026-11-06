```json
{"verdict":"ok","rule":"","evidence":"q=(W[\"W_q.0\"][sl]*g)@P; B.append(W[\"W_o.0\"][:,sl].astype(float)@v); coef=[F@D,F@B[5]@D,F@B[1]@D]","reason":"The code uses only permitted block weights for linear algebra and clustering, without executing LayerNorm, attention, an MLP forward pass, or any search."}
```