```json
{"verdict":"ok","rule":"","evidence":"svd(...); a-a.mean(1,keepdims=True); qs[l][s].T@ks[l][s]/8; W[f\"W_o.{l}\"][:,s].astype(float)@vs[l][s]; eigvals(c)","reason":"The code performs only weight-derived linear algebra, including scaling and centering weight matrices, without executing any network layer on inputs or conducting searches."}
```