## Observations

These are **weight-only inferences, not verified attention patterns**. Missing embeddings prevent assigning character names to directions or establishing exact attention distances.

- **L0H3 looks like a positional shift carrying character information.** For \(W_q^\top W_k/8\), its Frobenius norm is **4.305**, stable rank **14.933**, and leading eigenvalues include **0.724 ± 0.691i** and **0.446 ± 0.842i**. This structured, rotation-like operator contrasts with the other first-layer heads. Its OV map predominantly reads a separate, candidate character-content subspace.
- **L0H5 looks like a broad positional summarizer.** Its QK norm is only **0.210**, whereas its OV norm is **29.436**, with leading singular values **24.799, 12.900**. About **96.5% of its effective OV Frobenius norm** lies on the weight-derived positional source subspace.
- **Most later heads share a positional preference.** The first two output directions of L0H5 account for **63–68%** of the effective QK norm on the key side in layer 1. Their query-bias-induced key directions have cosine similarities **0.909–0.945** to a common direction. This suggests shared positional control, plausibly recency.
- **L3H3 has a particularly strong induction signature.** Let \(A\) denote the QK product, incorporating the stored normalization scales and centering, and \(B\) the corresponding OV map. Restricting \(A_{3,3}B_{0,3}\) to a candidate content subspace gives
  \[
  \operatorname{tr}(C)/\|C\|_* = \mathbf{0.714}.
  \]
  The other layer-3 heads range from **−0.118 to 0.099**. The L3H3 result remains **0.614–0.805** across different subspace sizes. This supports matching current content against content transported by L0H3.
- **The character-feature structure appears relatively compact.** ICA of first-MLP input weights repeatedly separates approximately **24 stronger components** from weaker ones: at rank 32, their mixing-vector norms are roughly **0.120–0.220**, versus **0.065–0.092** for the remaining eight. This weakly favors a reduced effective alphabet, but is not a recovered alphabet count.

## Heads

Here, **position fraction** means the QK Frobenius-norm fraction on the two L0H5 output directions above—not measured attention mass.

L0H0: **local** — positional QK structure together with content-reading OV; less elaborate than L0H3.  
L0H1: **unclear** — low-rank positional helper; QK stable rank 1.284, but its attention range is unresolved.  
L0H2: **unclear** — similar positional helper; QK stable rank 1.362 and OV strongly aligned with L0H1.  
L0H3: **previous** — strongest shift-like positional operator; supplies the content pathway used by the induction candidate.  
L0H4: **unclear** — another low-rank positional helper; QK stable rank 1.560 does not establish distance.  
L0H5: **broad** — exceptionally weak QK and strong, mostly positional, low-rank OV suggest pooling.

L1H0: **local** — position fraction 0.630; strong shared positional key bias.  
L1H1: **local** — position fraction 0.656; closely aligned positional bias.  
L1H2: **local** — position fraction 0.677; substantial positional control.  
L1H3: **local** — position fraction 0.680, the largest in this layer.  
L1H4: **local** — position fraction 0.651; weaker bias than its peers, but the same apparent mechanism.  
L1H5: **local** — position fraction 0.648; strongest alignment with the common bias direction.

L2H0: **local** — position fraction 0.477; substantial continuation of the positional mechanism.  
L2H1: **local** — tentative; fraction 0.297, with more content-dependent QK structure.  
L2H2: **local** — tentative; fraction 0.343 and aligned positional bias.  
L2H3: **local** — low confidence; weakest shared positional bias in this layer, potentially wider-ranging.  
L2H4: **local** — fraction 0.389 and the layer’s strongest query-bias-induced key direction.  
L2H5: **local** — tentative; fraction 0.309 with strongly aligned positional bias.

L3H0: **local** — low confidence; retains positional structure, but substantially weaker than earlier layers.  
L3H1: **local** — low confidence; positional fraction 0.241, with a weak shared bias; broad attention remains plausible.  
L3H2: **local** — tentative; positional fraction 0.248 and residual shared positional preference.  
L3H3: **copy** — uniquely strong positive matching composition with L0H3; almost no coupling to the shared positional-output directions.  
L3H4: **broad** — tentative; weak effective QK norm 0.623, enormous raw OV norm 68.192, and little shared positional coupling.  
L3H5: **local** — tentative; positional fraction 0.245 and a shared-bias cosine of 0.836.

## Reasoning

**Romaji, as is, is my best judgment.**

The shift-and-match circuit strongly favors preserved character order over shuffled lines. It suggests local character-pattern processing plus an induction mechanism for repeating earlier sequences.

The corpus identification is considerably weaker. The compact character-feature structure and predominance of apparently local processing fit Hepburn-style romaji somewhat better than English or Python. However, neither feature uniquely identifies romaji, and rare English letters could also produce weak character directions.

corpus: romaji  
order: as is  
probabilities: python=0.06 romaji=0.70 english=0.24; as is=0.99 shuffled=0.01