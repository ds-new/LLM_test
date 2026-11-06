## Observations

These are **weight-based circuit hypotheses**, not measured attention patterns. The evidence for character order is much stronger than for corpus identity.

I used \(A=Q^\top K/8\) and \(B=W_oV\), incorporating the corresponding `ln1_w` and projecting out the constant residual direction—without evaluating LayerNorm or attention.

- **Two early character-transfer channels:** L0H1 and L0H5 have OV 90%-energy ranks of **21 and 20**. Their addressing weights mostly occupy a different subspace from their value-reading weights. On a 20-dimensional candidate positional subspace, L0H1’s addressing matrix has cosine **0.651** with the square of L0H5’s polar factor. This supports a **one-character/two-character offset pair**.
- **A broad positional helper:** L0H4 has \(\|A\|_F=\mathbf{0.0093}\), versus **0.8775** and **0.9704** for L0H1/H5. Its OV singular values start **13.8775, 9.6501, 2.5066**. Its dominant output plane contains **76–93%** of each L1 head’s key-bias energy, suggesting a shared positional-selection mechanism.
- **Successor-matching circuits:** Let \(C\) be the candidate character-reading subspace derived from L0H1/H5, and define
  \[
  J_h=\frac{\operatorname{tr}(C^\top A_h B_{0,5}C)}
  {\|C^\top A_h B_{0,5}C\|_F}.
  \]
  This measures matching against information transferred by the likely previous-character head. \(J\) is **2.793 for L2H1, 2.590 for L3H1, and 2.763 for L3H5**, robust across alternative subspace constructions. These are strong induction clues.
- **Corpus evidence is weaker:** `W_proj.3` has substantial singular values through at least its 27th mode (**8.5731**). Together with the rich character-transfer channels, this weakly favors English/Python over romaji’s more restricted alphabet. However, clustering did **not** recover reliable character identities; its silhouette was only about **0.036**.

## Heads

“Local” assignments based on the shared positional mechanism are tentative; its exact distance scale remains unknown.

L0H0: **unclear** — Two-dimensional dominant OV structure; OV cosine **0.804** with L0H4 suggests a related positional/helper function, not an identifiable lag.  
L0H1: **local** — Strong relative-square relationship to L0H5 (**0.651**); plausibly transfers the character two positions back.  
L0H2: **unclear** — QK 90%-energy rank **5**, OV rank **8**; mixed addressing without a convincing offset or induction signature.  
L0H3: **unclear** — QK rank **3** and overlap with positional-helper outputs, but no robustly isolated lag.  
L0H4: **broad** — Exceptionally small QK coefficients: operator norm **0.00448**; consistent with broadly averaging the prefix.  
L0H5: **previous** — Character-preserving value channel, directional positional structure, and downstream successor-matching compositions jointly support this role.

L1H0: **local** — **76.3%** of key-bias energy lies in the shared positional plane; weakest of the L1 local assignments.  
L1H1: **local** — **90.3%** positional-plane key-bias energy; strong positional selection rather than clear induction.  
L1H2: **local** — **90.5%** positional-plane key-bias energy.  
L1H3: **local** — **93.0%** positional-plane key-bias energy, the strongest concentration.  
L1H4: **local** — **91.7%** positional-plane key-bias energy.  
L1H5: **local** — **82.4%** positional-plane key-bias energy; no comparably strong successor-matching signature.

L2H0: **local** — Tentative: retains the shared positional preference; **46.4%** of key-bias energy lies in that plane.  
L2H1: **copy** — Strong successor matching, \(J=\mathbf{2.793}\), with very little QK energy in the positional-helper plane.  
L2H2: **previous** — Tentative higher-level shift: previous-to-current and two-back-to-previous matching indices are **2.481** and **2.220**; ordinary induction index is negative, **−1.827**.  
L2H3: **local** — **78.9%** positional-plane key-bias energy; little direct induction evidence.  
L2H4: **local** — **66.3%** positional-plane key-bias energy and appreciable positional QK coupling.  
L2H5: **unclear** — Weaker positional coupling and only modest successor matching, \(J=\mathbf{0.539}\).

L3H0: **copy** — Tentative secondary induction head: \(J=\mathbf{1.227}\), remaining positive across alternative subspaces.  
L3H1: **copy** — \(J=\mathbf{2.590}\); **32.1%** of OV output energy enters the dominant final-MLP output subspace.  
L3H2: **unclear** — Neither a clear positional-offset circuit nor successor matching; \(J=\mathbf{-0.070}\).  
L3H3: **unclear** — Weak positional coupling and \(J=\mathbf{0.082}\); its smaller QK norm alone does not establish broad attention.  
L3H4: **unclear** — Some successor matching, \(J=\mathbf{0.772}\), but insufficient to distinguish induction from other contextual matching.  
L3H5: **copy** — \(J=\mathbf{2.763}\), additional strong key composition through L2H2, and **32.5%** OV output energy in the final-MLP task subspace.

## Reasoning

The likely previous/two-back pair, subsequent local-context processing, and multiple successor-matching circuits strongly favor **as-is text**. Within-line shuffling would greatly reduce the usefulness of these ordered-context mechanisms.

I narrowly favor **English**. The substantial character-transfer dimensionality weakly disfavors romaji. English versus Python remains unresolved: I found neither decoded vocabulary nor an identifiable indentation circuit. Consequently, the corpus probabilities are deliberately much less confident than the order probabilities.

corpus: english  
order: as is  
probabilities: python=0.36 romaji=0.12 english=0.52; as is=0.99 shuffled=0.01