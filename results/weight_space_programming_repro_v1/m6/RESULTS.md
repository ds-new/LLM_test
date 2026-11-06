# M6: the authors' Qwen3.5-0.8B edits, re-evaluated

Accuracy in %, ± one standard error over samples (samples are not independent within a problem, so this is a lower bound). cut = share of answers that hit 32,768 tokens.

| checkpoint | set | paper acc | ours acc | paper cut | ours cut | ours mean tokens |
|---|---|---|---|---|---|---|
| base | math500 | 7.9 | 8.2 ± 0.6 | 93.0 | 92.3 | 30742 |
| p50c | math500 | 48.1 | 50.5 ± 1.1 | 33.1 | 32.4 | 21997 |
