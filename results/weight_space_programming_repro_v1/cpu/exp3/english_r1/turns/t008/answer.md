## Observations

These are **weight-based hypotheses, not measured attention patterns**. Missing embeddings prevent assigning character identities or verifying attention distances.

Write \(A\) for each head’s QK product, including the learned LayerNorm scales and the \(1/8\) attention scaling; \(B\) for its corresponding OV product. All calculations were matrix algebra only.

- **L0H2 is exceptional.** `W_q.0[128:192]` and `W_k.0[128:192]` have Frobenius norms **30.818 and 28.300**. Other L0 heads have norms only **3.09–7.24**. Its QK product has singular values **1.408, 1.272, 1.213, 1.199…**, with substantial complex eigenvalue pairs. This suggests a directional positional mechanism, plausibly previous-character transport.
- **Four L0 heads have much simpler addressing.** H0/H1/H4/H5 have QK stable ranks **1.13/1.40/1.42/1.04**. Their OV products are strongly aligned—for example, H0–H4 cosine **0.903**. These look more like shared positional or boundary summaries than distinct character-transition mechanisms.
- **L1 and L2 share an addressing feature.** The affine key-selection directions derived from `ln1_b`, `W_q`, and `W_k` have within-L1 cosines **0.945–0.976**. Approximately **79–83%** of their squared norms lie in L0H5’s leading two OV output directions. L2 retains this association. A shared positional-summary/recency mechanism is plausible, although its meaning is not directly observable.
- **L3H1 has a distinctive induction-compatible connection.** Its QK product composed with L0H2’s OV product has normalized trace alignment **0.0950**, versus **−0.0022 ± 0.0025** under coordinate-permutation controls. Its composite eigenvalue statistic \(\operatorname{tr}(AB)/\sum|\lambda(AB)|\) is **0.842**. No other L3 head approaches this connection.
- **Corpus identification is weaker.** `W_proj.3` retains substantial singular values through approximately an alphabet-sized subspace: its 24th–29th values are **8.351, 8.064, 7.914, 7.461, 7.166, 5.833**, before falling to **4.072**. This offers weak evidence against a markedly restricted effective alphabet, but does not identify English or Python.

## Heads

Here, \(c\) denotes the weight-derived affine key-selection direction. Distance classifications are tentative.

L0H0: **broad** — QK norm 0.055, stable rank 1.13; strong shared low-dimensional OV structure.  
L0H1: **broad** — smallest L0 QK norm, 0.030; OV cosine 0.885 with H0.  
L0H2: **previous** — uniquely strong, multidimensional directional QK structure; supplies the strongest later induction-compatible connection.  
L0H3: **local** — intermediate QK norm 0.395; QK cosine 0.240 with the candidate previous-character head.  
L0H4: **broad** — QK stable rank 1.42; OV cosine 0.903 with H0 suggests a related summary mechanism.  
L0H5: **broad** — QK stable rank 1.04; its dominant OV output subspace feeds shared later addressing features.

L1H0: **local** — \(\|c\|=1.357\); alignment 0.980 with the common L1 addressing direction.  
L1H1: **local** — \(\|c\|=2.285\); common-direction alignment 0.987.  
L1H2: **local** — strongest L1 affine addressing term, \(\|c\|=2.588\); alignment 0.989.  
L1H3: **local** — \(\|c\|=2.351\); alignment 0.985 and substantial shared-summary dependence.  
L1H4: **local** — \(\|c\|=2.290\); 83% of its squared norm lies in the identified summary subspace.  
L1H5: **local** — \(\|c\|=1.753\); common-direction alignment 0.986.

L2H0: **local** — substantial retained addressing term, \(\|c\|=1.699\), aligned 0.806 with the L1 direction.  
L2H1: **unclear** — weaker shared addressing and higher-rank QK structure; neither distance nor induction is resolved.  
L2H2: **local** — comparatively low-rank QK structure and strong summary-subspace association; lower confidence.  
L2H3: **local** — \(\|c\|=1.463\); common-direction alignment 0.814.  
L2H4: **local** — \(\|c\|=1.577\); common-direction alignment 0.804.  
L2H5: **local** — \(\|c\|=1.509\); common-direction alignment 0.814.

L3H0: **broad** — high-rank, positive-trace QK structure with weak shared positional addressing; plausibly content-based retrieval.  
L3H1: **copy** — uniquely strong positive key composition through L0H2; the clearest induction candidate.  
L3H2: **unclear** — some shared addressing survives, but weakly; no compelling induction signature or recoverable distance.  
L3H3: **unclear** — similar residual addressing evidence to H2; insufficient to distinguish local from broad.  
L3H4: **unclear** — mixed content/addressing structure without a diagnostic transport connection.  
L3H5: **broad** — almost no alignment with the common addressing direction, 0.081; large OV norm, 45.4, suggests substantial nonlocal information transfer.

## Reasoning

The candidate previous-character pathway and its selective connection to an induction-compatible head favor **ordered text** over line-shuffled text. Shuffling preserves useful line statistics, but substantially removes the sequential relationships this circuit appears suited to exploit.

Among corpora, **English is my weak preference**. The rich local machinery and substantial alphabet-sized output subspace fit English; Python remains quite plausible. Romaji receives less probability because there is no clear reduced-alphabet signature. These distinctions are much less secure than the order judgment—I have not recovered labeled letters, words, or code syntax.

corpus: **english**  
order: **as is**  
probabilities: python=0.33 romaji=0.15 english=0.52; as is=0.96 shuffled=0.04