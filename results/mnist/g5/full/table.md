# G5, mnist

I = 1 − e₆₄ / e₁ per checkpoint (method.md section 20.2; e: MSE behind the PSNR); Δ = I_OOD − I_ID per seed. Mean over seeds ± half-width of the 95% t-interval. I lies in (−∞, 1]. The outcome follows the rule of method.md section 20.2.

| model | out of distribution | I_ID | I_OOD | Δ | outcome |
|---|---|---|---|---|---|
| ebt | noise 0.15 (no decision) | -10.2741 ± 0.9499 | -3.5816 ± 0.4285 | 6.6925 ± 0.5330 | - |
| ebt | noise 0.2 | -10.2741 ± 0.9499 | -0.2149 ± 0.1340 | 10.0592 ± 0.8272 | Thinking degrades the result in both; the relative degradation out of distribution is smaller. |
| ebt | noise 0.3 (no decision) | -10.2741 ± 0.9499 | 0.5412 ± 0.0656 | 10.8153 ± 0.9503 | - |

psnr at 1 step, at the trained step count and at 64 steps:

| model | out of distribution | data | 1 | trained | 64 |
|---|---|---|---|---|---|
| ebt | noise 0.15 (no decision) | in distribution | 27.6317 ± 0.0536 | 27.0003 ± 0.0307 (2 steps) | 17.1189 ± 0.3812 |
| ebt | noise 0.15 (no decision) | out of distribution | 23.6533 ± 0.1295 | 25.0057 ± 0.0604 (2 steps) | 17.0532 ± 0.3755 |
| ebt | noise 0.2 | in distribution | 27.6317 ± 0.0536 | 27.0003 ± 0.0307 (2 steps) | 17.1189 ± 0.3812 |
| ebt | noise 0.2 | out of distribution | 17.6727 ± 0.1785 | 21.0193 ± 0.3113 (2 steps) | 16.8408 ± 0.3468 |
| ebt | noise 0.3 (no decision) | in distribution | 27.6317 ± 0.0536 | 27.0003 ± 0.0307 (2 steps) | 17.1189 ± 0.3812 |
| ebt | noise 0.3 (no decision) | out of distribution | 10.8204 ± 0.1481 | 12.4526 ± 0.3109 (2 steps) | 14.2262 ± 0.6804 |
