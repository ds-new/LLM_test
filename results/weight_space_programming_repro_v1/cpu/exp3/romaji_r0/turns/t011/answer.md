## Observations

**These are weight-based hypotheses, not measured attention patterns.** Missing embeddings and the prohibition on forward passes particularly limit exact distance assignments and corpus identification.

For the statistics below, \(A\) is the head’s QK product, including the learned LayerNorm scale and fixed centering, divided by \(8\); \(B\) is its corresponding OV product. Neither includes an evaluated LayerNorm, softmax, or ReLU.

- **L0H3 is the strongest previous-character candidate.** Its QK singular values begin **0.244, 0.222, 0.201**; eigenvalues include **0.137 ± 0.133i**. Its substantial, rotation-like structure suggests positional displacement rather than simple content aggregation. This does not establish the exact offset.
- **Four early heads share a small write subspace.** The first two singular directions account for **98.0%, 96.1%, 93.0%, and 96.6%** of OV energy in L0H1, H2, H4, and H5. Their writes concentrate around coordinates **21, 39, 58, 99**. Those coordinates also feature prominently in later biases: for example, `ln1_b.1[39] = −1.5676`, `[58] = 1.3651`, `[99] = −1.3355`. This suggests shared positional/control features, not four independent character-copying mechanisms.
- **Layers 1–2 repeatedly consult that shared subspace.** It contains **80–86%** of each L1 head’s key-only affine score coefficient. A common recency mechanism is plausible, although its temporal meaning remains inferred.
- **L3H3 has a distinctive induction-like connection to L0H3.** The normalized alignment \(\mathrm{tr}(A_{\mathrm{L3H3}}B_{\mathrm{L0H3}})/(\|A\|_F\|B\|_F)\) is **0.06263**. The other last-layer heads range from **−0.01041 to 0.00637**. Across 400 coordinate-permutation controls, the maximum was **0.00193**. This establishes unusually specific weight alignment—not, by itself, demonstrated copying.
- **Corpus evidence is much weaker.** The first MLP retains a rich spectrum after removing selected attention-related subspaces; its 26th singular value is **4.105**. That mildly favors a full-alphabet corpus, but these directions cannot be equated with identified letters.

## Heads

Here, \(r\) denotes the key-only affine score coefficient derived from the Q/K weights and `ln1_b`.

L0H0: **local** — tentative; substantial overlap with L0H3’s positional-candidate subspace, but a less elaborate QK spectrum.  
L0H1: **unclear** — 98.0% of OV energy lies in two shared control directions; temporal range is unresolved.  
L0H2: **unclear** — similarly concentrated OV structure, at 96.1%; no identifiable offset.  
L0H3: **previous** — strongest displacement-like QK structure and a specific downstream induction-like connection.  
L0H4: **unclear** — 93.0% two-direction OV concentration suggests control/aggregation, without establishing distance.  
L0H5: **broad** — tentative; exceptionally small QK norm **0.034**, versus OV norm **18.812**, favors low-selectivity aggregation.

L1H0: **local** — tentative shared-recency interpretation; \(\|r\|=3.975\), with 85.5% in the shared subspace.  
L1H1: **local** — same mechanism; \(\|r\|=3.689\), shared fraction 85.5%.  
L1H2: **local** — same mechanism; \(\|r\|=3.051\), shared fraction 85.3%.  
L1H3: **local** — same mechanism; \(\|r\|=3.390\), shared fraction 85.5%.  
L1H4: **local** — somewhat weaker shared steering; \(\|r\|=2.329\), shared fraction 80.4%.  
L1H5: **local** — strong shared steering; \(\|r\|=3.895\), shared fraction 86.2%.

L2H0: **local** — tentative; \(\|r\|=2.020\), shared fraction 83.5%.  
L2H1: **local** — tentative; \(\|r\|=1.880\), shared fraction 82.9%.  
L2H2: **local** — tentative; \(\|r\|=1.608\), shared fraction 79.5%.  
L2H3: **unclear** — weaker common steering: \(\|r\|=1.060\), shared fraction 62.0%; locality is less supported.  
L2H4: **local** — strongest L2 affine steering, \(\|r\|=2.305\); shared fraction 83.3%.  
L2H5: **local** — tentative; \(\|r\|=2.023\), shared fraction 85.8%.

L3H0: **unclear** — content-dependent QK structure; neither a specific offset nor induction is established.  
L3H1: **unclear** — weaker common steering could permit broader retrieval, but does not establish it.  
L3H2: **unclear** — retains some shared steering without a distinctive copying connection.  
L3H3: **copy** — strongest induction candidate: exceptional positive QK–earlier-OV alignment, largely separate from the shared steering circuit.  
L3H4: **broad** — tentative; relatively weak QK norm **0.622**, very large OV norm **49.234**, and little shared positional steering.  
L3H5: **unclear** — mixed content/common-steering structure; no convincing induction-specific alignment.

## Reasoning

The displacement-like early head and its specific connection to a later matching circuit favor **ordered text** over within-line shuffling. My best interpretation is predominantly short-range processing, supplemented by broad aggregation and one notable repetition-retrieval mechanism.

I **weakly favor English**. The rich feature structure mildly disfavors a strongly restricted romaji alphabet. Nothing recovered specifically identifies Python indentation or keywords, but their absence from this analysis is not evidence that they are absent from the model. English versus Python remains substantially uncertain; the probabilities are subjective judgments.

corpus: english  
order: as is  
probabilities: python=0.33 romaji=0.17 english=0.50; as is=0.94 shuffled=0.06