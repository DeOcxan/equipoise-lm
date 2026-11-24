# G5, regression

I = 1 − e₆₄ / e₁ per checkpoint (method.md section 20.2; e: branch distance r); Δ = I_OOD − I_ID per seed. Mean over seeds ± half-width of the 95% t-interval. I lies in (−∞, 1]. The outcome follows the rule of method.md section 20.2.

| model | out of distribution | I_ID | I_OOD | Δ | outcome |
|---|---|---|---|---|---|
| ebt_mse | split | -0.2932 ± 0.0208 | -0.0795 ± 0.0578 | 0.2137 ± 0.0726 | Thinking degrades the result in both; the relative degradation out of distribution is smaller. |
| ebt_contrastive | split | 0.1563 ± 0.1137 | -0.6714 ± 0.5500 | -0.8277 ± 0.4422 | Thinking degrades the result out of distribution only. |
| ebt_combined | split | -0.2539 ± 0.0169 | -0.0113 ± 0.0400 | 0.2426 ± 0.0410 | Thinking degrades the result in distribution and out of distribution on average; the relative degradation out of distribution is smaller. |
| ebt_regularized | split | -0.1150 ± 0.0844 | -0.9601 ± 2.1183 | -0.8451 ± 2.0633 | Thinking degrades the result in distribution and out of distribution on average; the relative degradation out of distribution is not measurably different. |

branch_distance at 1 step, at the trained step count and at 64 steps:

| model | out of distribution | data | 1 | trained | 64 |
|---|---|---|---|---|---|
| ebt_mse | split | in distribution | 0.7574 ± 0.0119 | 0.9794 ± 0.0009 (4 steps) | 0.9794 ± 0.0012 |
| ebt_mse | split | out of distribution | 0.8363 ± 0.0187 | 0.9025 ± 0.0636 (4 steps) | 0.9032 ± 0.0640 |
| ebt_contrastive | split | in distribution | 65.5789 ± 11.4427 | 58.4247 ± 18.7188 (4 steps) | 55.4700 ± 13.0901 |
| ebt_contrastive | split | out of distribution | 16.7662 ± 4.1104 | 33.6809 ± 9.9128 (4 steps) | 27.4973 ± 7.3748 |
| ebt_combined | split | in distribution | 0.7801 ± 0.0111 | 0.9781 ± 0.0021 (4 steps) | 0.9781 ± 0.0019 |
| ebt_combined | split | out of distribution | 0.8468 ± 0.0163 | 0.8584 ± 0.0468 (4 steps) | 0.8567 ± 0.0496 |
| ebt_regularized | split | in distribution | 2.7277 ± 0.1224 | 3.0304 ± 0.2823 (4 steps) | 3.0438 ± 0.3192 |
| ebt_regularized | split | out of distribution | 0.8462 ± 0.0505 | 0.9464 ± 0.2483 (4 steps) | 1.6253 ± 1.6566 |
