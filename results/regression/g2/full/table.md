Test split, mean over 5 seeds ± half-width of the 95% t-interval. Mean-predictor MSE floor: 2.125 (method.md section 16.5).

| Model | mean r | fraction r ≤ 0.1 | fraction ŷ > 0 | P(sign ŷ = sign ŷ₀) | mean abs(ŷ − ŷ₀) | test MSE |
|---|---|---|---|---|---|---|
| (a) regressor, MSE | 0.973 ± 0.002 | 0.000 ± 0.000 | 0.631 ± 0.025 | n/a | n/a | 2.099 ± 0.000 |
| (a2) mixture density network, NLL | 0.001 ± 0.001 | 1.000 ± 0.000 | 0.746 ± 0.092 | n/a | n/a | 4.163 ± 0.030 |
| (b) EBT, MSE | 0.979 ± 0.001 | 0.000 ± 0.000 | 0.745 ± 0.068 | 0.509 ± 0.004 | 1.158 ± 0.008 | 2.099 ± 0.000 |
| (c1) EBT, contrastive | 58.523 ± 18.663 | 0.001 ± 0.001 | 0.480 ± 0.068 | 0.241 ± 0.081 | 75.410 ± 24.094 | 7637.797 ± 4416.252 |
| (c2) EBT, MSE + contrastive | 0.978 ± 0.002 | 0.000 ± 0.000 | 0.647 ± 0.111 | 0.534 ± 0.015 | 1.156 ± 0.009 | 2.099 ± 0.000 |
| (c3) EBT, contrastive + energy penalty | 3.035 ± 0.315 | 0.000 ± 0.000 | 0.501 ± 0.032 | 0.848 ± 0.022 | 3.035 ± 0.358 | 59.113 ± 13.369 |
| (c4) IBC energy, InfoNCE, DFO inference | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.736 ± 0.391 | n/a | n/a | 4.125 ± 0.077 |
| (c5) IRED energy, 10 landscapes | 0.017 ± 0.006 | 1.000 ± 0.000 | 0.667 ± 0.399 | 0.504 ± 0.010 | 1.663 ± 0.018 | 4.123 ± 0.035 |

Mean of y² on the test split: 2.0985.

Energy gradient, seed 0, test split: mean abs(∂E/∂ŷ) at the targets (on the branches)
and at ŷ = 0, and α times each, the size of one thinking step at those points.

| Model | α | on branches | at ŷ = 0 | α · on branches | α · at ŷ = 0 |
|---|---|---|---|---|---|
| (b) EBT, MSE | 0.1 | 14.131 | 0.218 | 1.413 | 0.022 |
| (c1) EBT, contrastive | 1 | 6.361 | 0.794 | 6.361 | 0.794 |
| (c2) EBT, MSE + contrastive | 1 | 1.254 | 0.020 | 1.254 | 0.020 |
| (c3) EBT, contrastive + energy penalty | 1 | 3.877 | 0.019 | 3.877 | 0.019 |
| (c4) IBC energy, InfoNCE, DFO inference | n/a | 312.982 | 29.330 | n/a | n/a |
| (c5) IRED energy, 10 landscapes | 0.3544 | 0.071 | 0.321 | 0.025 | 0.114 |

(c4) takes no gradient steps. (c5): gradient on landscape k = 0 in the original
units; α is σ_y² λ_0, the displacement in original units per unit gradient of its
first descent step (method.md section 16.8).

(a2) readouts, test split:

| Readout | mean r | fraction r ≤ 0.1 | fraction ŷ > 0 | test MSE |
|---|---|---|---|---|
| most probable branch, eq. 49 (table above) | 0.001 ± 0.001 | 1.000 ± 0.000 | 0.746 ± 0.092 | 4.163 ± 0.030 |
| most probable value, eq. 48 | 0.001 ± 0.001 | 1.000 ± 0.000 | 0.724 ± 0.145 | 4.160 ± 0.035 |
| centre of a kernel drawn with probability α_i | 0.001 ± 0.000 | 1.000 ± 0.000 | 0.506 ± 0.008 | 4.189 ± 0.104 |

(a2) test negative log-likelihood: -4.248 ± 2.980.
