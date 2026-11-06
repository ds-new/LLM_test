## Observations

These are **weight-based hypotheses, not observed attention patterns**. Missing embeddings prevent reliably translating directions into characters or distances.

For the statistics below, let \(C=I-\mathbf1\mathbf1^\top/384\), \(D_l=\operatorname{diag}(\text{ln1_w.l})\), and define the head-specific weight products
\[
M=C D_l Q^\top K D_l C/8,\qquad A=OVD_lC.
\]
These were analyzed algebraically; no layer was executed.

- **L0H2 is exceptional:** \(\|M\|_F=4.0688\), versus \(0.0295–0.3945\) for the other first-layer heads. Its substantial complex eigenvalues, including \(0.6320\pm0.8127i\), suggest a positional shift mechanism. Its OV map carries considerably more dimensions than its neighbors. This is my strongest **previous-character** candidate.
- **L0H0/H1/H4/H5 form a shared low-dimensional system.** Their leading three OV singular directions contain respectively **98.64%, 98.56%, 94.65%, 98.32%** of OV energy. Their OV matrices have pairwise cosines **0.724–0.903**. This suggests related contextual/positional summaries rather than four independent character-copying mechanisms.
- **Layer 1 strongly reads those summaries:** their shared three-dimensional write subspace accounts for **60.1–69.2%** of each L1 head’s key-side \(M\) energy. Corresponding learned parameters include `ln1_b.1[222] = -1.4712` and `ln1_w.1[222] = 1.1239`. This supports coordinated position-sensitive context selection.
- **L3H1 has a distinctive induction-compatible connection to L0H2:**  
  \(\operatorname{tr}(M_{3,1}A_{0,2})=0.4413\), and trace/nuclear-norm \(=0.2504\). Its Frobenius-normalized trace is **0.09504**, versus coordinate-permutation controls with standard deviation **0.00315**. Other L3 heads lack this connection. This establishes unusually strong alignment—not, by itself, behavioral proof of induction.
- **`W_proj.3` contains 27 substantial, delocalized output directions.** After removing coordinate-localized singular directions, singular values 27 and 28 are **7.1656** and **2.8415**. This weakly favors a broadly used alphabet over strongly restricted romaji lettering; it is not an exact vocabulary count.

## Heads

“Local” and “broad” assignments are less secure than the identified circuit connections.

L0H0: **broad** — small \(M\) norm, 0.0546; 98.64% of OV energy in three directions; shared summary circuitry.  
L0H1: **broad** — smallest \(M\) norm, 0.0295; 98.56% three-direction OV concentration.  
L0H2: **previous** — unusually strong, rotational QK structure and multidimensional OV; supplies the strongest induction-compatible connection.  
L0H3: **local** — positional-subspace overlap and stronger QK than the broad group; exact offset unresolved.  
L0H4: **broad** — shared summary OV, cosine 0.9033 with H0; 94.65% three-direction concentration.  
L0H5: **broad** — shared summary OV; 98.32% three-direction concentration and nearly rank-one QK.

L1H0: **local** — shared positional-summary subspace captures 64.1% of key-side QK energy.  
L1H1: **local** — corresponding fraction 60.1%; strong shared position-sensitive structure.  
L1H2: **local** — corresponding fraction 63.4%; largest L1 linear key-bias norm, 2.5879.  
L1H3: **local** — corresponding fraction 63.5%; closely resembles the common L1 QK structure.  
L1H4: **local** — strongest shared-summary key fraction, 69.2%.  
L1H5: **local** — shared-summary key fraction 63.1%; no compelling induction signature.

L2H0: **local** — retains shared positional structure; linear key-bias norm 1.6991.  
L2H1: **unclear** — weaker positional dependence; no convincing shifted-copy connection.  
L2H2: **local** — strongest L2 similarity to the common L1 QK structure, cosine 0.2036; tentative.  
L2H3: **local** — shared positional structure and key-bias norm 1.4625.  
L2H4: **local** — similar positional structure; key-bias norm 1.5774.  
L2H5: **local** — similar positional structure; key-bias norm 1.5086.

L3H0: **unclear** — positive QK trace suggests feature matching, but does not distinguish self-matching, retrieval, or induction.  
L3H1: **copy** — uniquely strong positive key composition with the previous-character candidate L0H2.  
L3H2: **unclear** — some positional dependence survives, but insufficient evidence for a particular span.  
L3H3: **unclear** — stronger positional bias than neighboring L3 heads, but locality remains unverified.  
L3H4: **unclear** — mixed feature matching and weak positional dependence; no clear induction connection.  
L3H5: **broad** — tentative: weak positional coupling, small QK norm 0.8339, and large OV norm 45.3912 suggest contextual aggregation.

## Reasoning

The apparent previous-character mechanism, subsequent position-sensitive feature gathering, and downstream induction-compatible circuit favor **as-is text** substantially over shuffled lines. Shuffling preserves useful line statistics, but largely destroys the sequential relationships this organization appears suited to exploit.

**English is my modest best guess.** The broad output-direction structure weakly disfavors romaji. The remaining local-context-plus-repetition organization fits both English and Python; these measurements do **not** provide a reliable discriminator between them. Corpus confidence is consequently much lower than order confidence. The probabilities below are subjective judgments, not calibrated estimates.

corpus: **english**  
order: **as is**  
probabilities: python=0.40 romaji=0.07 english=0.53; as is=0.97 shuffled=0.03