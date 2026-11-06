```json
{"verdict":"ok","rule":"","evidence":"q=ctr(Q*g); k=ctr(K*g); v=ctr(V*g); As.append(q.T@k/8); Bs.append(O@v); cs.append(k.T@(Q@b)/8); u,s,vh=svd(Bs[5],full_matrices=False); bp=b[np.ix_(ix,ix)]","reason":"The code only performs permitted linear-algebra and statistical computations on allowed block weights, without executing a network layer on inputs or conducting any search."}
```