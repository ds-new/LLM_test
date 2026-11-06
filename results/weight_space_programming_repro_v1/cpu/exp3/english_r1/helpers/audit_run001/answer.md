```json
{"verdict":"ok","rule":"","evidence":"svdvals(a); product_stats((Q*g).T/8, K*g); product_stats(O, V*g); np.linalg.norm(Q@b); np.linalg.norm(K@b)","reason":"The code performs only linear-algebra computations on permitted block weights, including weight-derived directions, without executing any network layer or conducting a search."}
```