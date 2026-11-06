# Reproduction results (CPU experiments)

## exp1

| unit | paper | ours |
|---|---|---|
| breakout | correct=True, confidence=99 | correct=True, confidence=97.0 |
| pong | correct=True, confidence=96 | correct=True, confidence=97.0 |
| pacman | correct=True, confidence=96 | correct=True, confidence=95.0 |
| freeway | correct=True, confidence=97 | correct=True, confidence=97.0 |

## exp2

| unit | paper | ours |
|---|---|---|
| circle | correct=True, confidence=84, drawn_as_outline=True | correct=True, confidence=84.0, drawn_as_outline=True |
| square | correct=True, confidence=77, drawn_as_outline=True | correct=True, confidence=72.0, drawn_as_outline=True |
| triangle | correct=True, confidence=88, drawn_as_outline=True | correct=True, confidence=88.0, drawn_as_outline=True |

## exp3

| unit | paper | ours |
|---|---|---|
| python_r0 | corpus_correct=True, probability_of_guess=0.56, copy_named=['L2H1', 'L3H1', 'L3H5'], copy_exact_match=True | corpus_correct=False, probability_of_guess=0.52, copy_named=['L2H1', 'L3H0', 'L3H1', 'L3H5'], copy_exact_match=False |
| python_r1 | corpus_correct=True, probability_of_guess=0.52, copy_named=['L2H1', 'L3H1', 'L3H5'], copy_exact_match=True | corpus_correct=True, probability_of_guess=0.52, copy_named=['L2H1', 'L3H1', 'L3H5'], copy_exact_match=True |
| romaji_r0 | corpus_correct=True, probability_of_guess=0.55, copy_named=['L3H3'], copy_exact_match=True | corpus_correct=False, probability_of_guess=0.5, copy_named=['L3H3'], copy_exact_match=True |
| romaji_r1 | corpus_correct=True, probability_of_guess=0.62, copy_named=['L3H3'], copy_exact_match=True | corpus_correct=True, probability_of_guess=0.7, copy_named=['L3H3'], copy_exact_match=True |
| english_r0 | corpus_correct=True, probability_of_guess=0.58, copy_named=['L3H1'], copy_exact_match=True | corpus_correct=True, probability_of_guess=0.53, copy_named=['L3H1'], copy_exact_match=True |
| english_r1 | corpus_correct=True, probability_of_guess=0.48, copy_named=['L3H1'], copy_exact_match=True | corpus_correct=True, probability_of_guess=0.52, copy_named=['L3H1'], copy_exact_match=True |

## exp5

| unit | paper | ours |
|---|---|---|
| breakout | score=517.4, ppo_100M=15.8, ppo_500M=25.3 | score=502.2 |
| seaquest | score=53.1, ppo_100M=5.6, ppo_500M=88.9 | score=33.9 |
| asterix | score=53.2, ppo_100M=25.8, ppo_500M=55.2 | score=50.8 |
