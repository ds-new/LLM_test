## Observations

**The strongest evidence concerns context use, not corpus identity.** These are weight-space inferences; none was verified with a forward pass.

Let \(A_h=Q_h^\top K_h\) and \(B_h=O_hV_h\), incorporating the available LayerNorm gains and linear centering into \(Q,K,V\).

- **A two-offset circuit:** L0H1 and L0H5 have related rotational QK structure. Comparing their shared spectral modes, the phase-doubling agreement is **0.876**, versus **0.217** for equal phases. This supports L0H5 reading approximately one character back and L0H1 approximately two.
- **Broad pooling:** L0H4 has \(\|A\|_F=0.074\), versus **7.020** and **7.764** for L0H1/H5. Its OV product is strong but concentrated: leading singular values **13.878, 9.650**. Those value directions overlap the rotational QK subspace by **95.6% and 90.8%**, suggesting pooled positional information.
- **Later positional addressing:** The two leading L0H4 output directions account for **22–47%** of each L1 head’s key-weight energy.
- **Induction-like composition:** Define \(I(h,e)=\operatorname{tr}(A_hB_e)/\|A_hB_e\|_*\). Notable positive scores are **0.212** for L2H1←L0H5, **0.288** for L3H1←L2H2, and **0.390** for L3H5←L2H2. These support matching against information transported by earlier heads, although the missing readout prevents confirming copying.
- **Weak corpus clue:** An exploratory `W_fc.0`/`W_o.0`/`W_v.0` feature decomposition repeatedly highlights two unlabeled directions, including a strong repeated-direction pattern. One threshold gives **40** feature pairings 483→411 and **31** pairings 411→411. This could reflect newline→indentation and repeated spaces, but these are **weight-feature counts, not observed character frequencies**, and their identities remain unknown.

## Heads

Here, \(p\) denotes key-weight energy in the two leading L0H4 output directions. Labels are tentative; positional sensitivity alone does not establish an attention distance.

L0H0: **broad** — low-rank OV; OV cosine **0.804** with the broad-pooling L0H4.  
L0H1: **local** — rotational modes approximately double L0H5’s phases; likely two characters back.  
L0H2: **unclear** — low-rank QK and selective values, without a securely identifiable distance.  
L0H3: **unclear** — mixed low-rank structure; some pooling alignment, but no decisive offset signature.  
L0H4: **broad** — exceptionally weak QK product and strong, predominantly positional OV directions.  
L0H5: **previous** — strongest candidate for the one-step member of the paired offset circuit.  

L1H0: **local** — positional-addressing candidate; \(p=0.221\), with weak induction evidence.  
L1H1: **local** — strong positional-key dependence, \(p=0.460\).  
L1H2: **local** — strong positional-key dependence, \(p=0.428\).  
L1H3: **local** — strongest positional-key dependence, \(p=0.470\).  
L1H4: **local** — strong positional-key dependence, \(p=0.448\).  
L1H5: **local** — strong positional-key dependence, \(p=0.411\); weaker copy signatures.  

L2H0: **unclear** — mixed content/position structure; no compelling copying or distance signature.  
L2H1: **copy** — positive composition with the likely previous-character head: \(I=0.212\).  
L2H2: **local** — substantial positional dependence, \(p=0.122\); supplies the strongest later matching circuits.  
L2H3: **local** — substantial positional dependence, \(p=0.126\), without comparable induction evidence.  
L2H4: **local** — positional dependence \(p=0.108\); weak induction signatures.  
L2H5: **unclear** — weaker positional dependence and no distinctive induction composition.  

L3H0: **unclear** — mostly content-based weight structure; retrieval distance unresolved.  
L3H1: **copy** — strong matching composition through L2H2, \(I=0.288\).  
L3H2: **unclear** — little positional dependence and no strong identified induction circuit.  
L3H3: **broad** — tentative aggregation head: relatively weak QK norm **8.43**, strong OV norm **37.94**.  
L3H4: **unclear** — content-selective structure without a securely identifiable range or copying circuit.  
L3H5: **copy** — strongest induction-like composition, \(I=0.390\), through L2H2.  

## Reasoning

The paired short-offset circuit, subsequent positional addressing, and later induction-like compositions strongly favor **ordered text** over within-line shuffling.

I narrowly favor **Python**: the exploratory repeated/boundary-like feature structure is compatible with indentation, while context matching could support repeated identifiers. **English remains a substantial alternative.** The available weights do not identify those features as particular characters, and I found no reliable corpus-specific fingerprint. Romaji is less favored, but not excluded.

corpus: python  
order: as is  
probabilities: python=0.52 romaji=0.11 english=0.37; as is=0.98 shuffled=0.02