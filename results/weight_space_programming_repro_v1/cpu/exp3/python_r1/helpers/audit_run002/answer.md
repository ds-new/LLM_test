```json
{"verdict":"ok","rule":"","evidence":"Q[h] = (W[f\"W_q.{l}\"][sl]*g)@P; A[h], B[h] = Q[h].T@K[h], O[h]@V[h]; all remaining operations compute matrix products, decompositions, norms, and similarities.","reason":"The code uses only permitted block weights as linear-algebra objects, without executing any layer on an input, accessing unavailable weights, or performing a search."}
```