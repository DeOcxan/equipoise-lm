# G5, text

I = 1 − e₆₄ / e₁ per checkpoint (method.md section 20.2; e: perplexity 2^BPC); Δ = I_OOD − I_ID per seed. Mean over seeds ± half-width of the 95% t-interval. I lies in (−∞, 1]. The outcome follows the rule of method.md section 20.2.

| model | out of distribution | I_ID | I_OOD | Δ | outcome |
|---|---|---|---|---|---|
| ebt | split | -0.0095 ± 0.0060 | -0.0003 ± 0.0074 | 0.0092 ± 0.0101 | Thinking degrades the result in distribution and out of distribution on average; the relative degradation out of distribution is not measurably different. This is not the paper's effect. |

bpc at 1 step, at the trained step count and at 64 steps:

| model | out of distribution | data | 1 | trained | 64 |
|---|---|---|---|---|---|
| ebt | split | in distribution | 2.3208 ± 0.0218 | 2.3318 ± 0.0222 (4 steps) | 2.3344 ± 0.0215 |
| ebt | split | out of distribution | 2.6126 ± 0.0548 | 2.6121 ± 0.0610 (4 steps) | 2.6131 ± 0.0611 |
