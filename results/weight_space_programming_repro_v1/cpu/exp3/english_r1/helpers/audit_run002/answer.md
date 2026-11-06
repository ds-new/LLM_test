```json
{"verdict":"ok","rule":"","evidence":"Qg, Kg, Vg = Q*g, K*g, V*g; A = Qg.T@Kg/8; B = O@Vg; key_linear = Kg.T@(Q@b)/8; svd(A,full_matrices=False); eigvals(Kg@Qg.T/8)","reason":"The code performs only permitted linear-algebra analyses of allowed weights, without executing a network layer on input or conducting any search."}
```