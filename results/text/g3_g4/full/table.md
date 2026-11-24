# G3 and G4, text

Mean over seeds 0, 1, 2, 3, 4 ± half-width of the 95% t-interval (method.md section 19). Undefined seeds are left out of a mean and counted as (k of n defined).

## ebt

G3, test bpc against inference steps:

| steps | bpc | std over seeds |
|---|---|---|
| 1 | 2.3208 ± 0.0218 | 0.0176 |
| 2 | 2.3236 ± 0.0214 | 0.0172 |
| 4 | 2.3318 ± 0.0222 | 0.0179 |
| 8 | 2.3338 ± 0.0218 | 0.0176 |
| 16 | 2.3343 ± 0.0216 | 0.0174 |
| 32 | 2.3343 ± 0.0216 | 0.0174 |
| 64 | 2.3344 ± 0.0215 | 0.0173 |

Ranges: falling absent; flat 1–64; rising absent; undefined absent.
In order: flat 1–64.

- Spearman ρ(final energy, error): 0.0482 ± 0.2094

| N | random | lowest energy | oracle | fraction |
|---|---|---|---|---|
| 1 | 2.3318 ± 0.0222 | 2.3318 ± 0.0222 | 2.3318 ± 0.0222 | undefined |
| 2 | 2.3319 ± 0.0222 | 2.3320 ± 0.0222 | 2.3314 ± 0.0222 | -0.399 ± 0.152 |
| 4 | 2.3319 ± 0.0222 | 2.3322 ± 0.0222 | 2.3310 ± 0.0221 | -0.408 ± 0.137 |
| 8 | 2.3318 ± 0.0222 | 2.3324 ± 0.0222 | 2.3306 ± 0.0221 | -0.436 ± 0.157 |
| 16 | 2.3318 ± 0.0222 | 2.3325 ± 0.0222 | 2.3303 ± 0.0221 | -0.442 ± 0.185 |
| 32 | 2.3318 ± 0.0222 | 2.3327 ± 0.0222 | 2.3299 ± 0.0221 | -0.450 ± 0.242 |
