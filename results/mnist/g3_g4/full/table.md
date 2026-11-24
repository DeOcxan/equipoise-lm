# G3 and G4, mnist

Mean over seeds 0, 1, 2, 3, 4 ± half-width of the 95% t-interval (method.md section 19). Undefined seeds are left out of a mean and counted as (k of n defined).

## ebt

G3, test psnr against inference steps:

| steps | psnr | std over seeds |
|---|---|---|
| 1 | 27.6317 ± 0.0536 | 0.0432 |
| 2 | 27.0003 ± 0.0307 | 0.0247 |
| 4 | 25.4051 ± 0.0422 | 0.0340 |
| 8 | 23.4187 ± 0.0807 | 0.0650 |
| 16 | 21.2754 ± 0.1459 | 0.1175 |
| 32 | 19.1120 ± 0.2524 | 0.2033 |
| 64 | 17.1189 ± 0.3812 | 0.3070 |

Ranges: falling absent; flat absent; rising 1–64; undefined absent.
In order: rising 1–64.

- Spearman ρ(final energy, error): 0.6124 ± 0.0083
- Spearman ρ, first perturbed candidate: 0.6434 ± 0.0095
- PSNR, standard output: 27.0003 ± 0.0307
- PSNR, first perturbed candidate (N = 1): 25.4971 ± 0.0508

| N | random | lowest energy | oracle | fraction |
|---|---|---|---|---|
| 1 | 183.3951 ± 2.1467 | 183.3951 ± 2.1467 | 183.3951 ± 2.1467 | undefined |
| 2 | 183.4884 ± 2.0498 | 185.5250 ± 2.0981 | 166.3953 ± 1.9098 | -0.119 ± 0.033 |
| 4 | 183.5128 ± 1.9517 | 189.3854 ± 1.7534 | 153.8697 ± 1.7402 | -0.198 ± 0.030 |
| 8 | 183.4857 ± 2.0157 | 193.8982 ± 1.7607 | 144.0899 ± 1.7243 | -0.264 ± 0.031 |
| 16 | 183.4745 ± 1.9390 | 198.7000 ± 1.6829 | 136.3272 ± 1.5699 | -0.323 ± 0.029 |
| 32 | 183.4568 ± 1.9050 | 203.8005 ± 1.5310 | 129.8086 ± 1.4416 | -0.379 ± 0.025 |
