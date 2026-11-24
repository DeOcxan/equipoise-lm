# Glossary

The labels, abbreviations and quantities used in the v1.0 results. Each entry links to the place
where it is defined. Bracketed keys such as [Bishop] refer to [design.md,
References](design.md#references).

## Models

| Label | Name in the code and results | What it is | Defined in |
|---|---|---|---|
| (a) | `regressor` | feed-forward regressor trained with mean squared error | [method.md 16.3](method.md#163-models-compared) |
| (a2) | `mdn` | mixture density network [Bishop], trained by negative log-likelihood | [method.md 16.8](method.md#168-models-from-outside-the-ebt-paper-a2-c4-and-c5) |
| (b) | `ebt_mse` | EBT trained with the paper's objective: S2 thinking and mean squared error of the final candidate | [method.md 16.3](method.md#163-models-compared) |
| (c1) | `ebt_contrastive` | EBT trained with the contrastive energy loss alone; it diverged in v1.0 | [method.md 16.4](method.md#164-contrastive-loss) |
| (c2) | `ebt_combined` | EBT trained with mean squared error plus the contrastive loss | [method.md 16.3](method.md#163-models-compared) |
| (c3) | `ebt_regularized` | EBT trained with the contrastive loss plus an energy-magnitude penalty [Du-Mordatch], the remedy for the (c1) divergence | [method.md 16.4](method.md#164-contrastive-loss) |
| (c4) | `ibc` | implicit behavioural cloning energy [IBC]: InfoNCE training, derivative-free inference | [method.md 16.8](method.md#168-models-from-outside-the-ebt-paper-a2-c4-and-c5) |
| (c5) | `ired` | IRED energy [IRED]: a sequence of noised energy landscapes and annealed gradient inference | [method.md 16.8](method.md#168-models-from-outside-the-ebt-paper-a2-c4-and-c5) |
| MNIST EBT | `ebt` | bidirectional patch-transformer EBT, S1, 2 inference steps | [method.md 17.3](method.md#173-models) |
| One-pass denoiser | `denoiser` | one-pass patch transformer of the same architecture and size | [method.md 17.3](method.md#173-models) |
| Text EBT | `ebt` | causal EBT, S2, 4 inference steps | [method.md 18.3](method.md#183-models) |
| Causal Transformer | `transformer` | the autoregressive baseline of the same size | [method.md 18.3](method.md#183-models) |

## Training modes

| Term | Meaning | Defined in |
|---|---|---|
| S1 | the candidate is detached between thinking steps; every step keeps its second-order graph; the loss is the mean of the per-step losses | [method.md 6.2](method.md#62-training-modes) |
| S2 | the candidate is not detached; only the final step keeps its second-order graph; the loss is that of the final step | [method.md 6.2](method.md#62-training-modes) |
| full unroll | the candidate is not detached; every step keeps its second-order graph; the loss is that of the final step | [method.md 6.2](method.md#62-training-modes) |

## Questions

| v1.0 question | Subject | New numbering |
|---|---|---|
| [G1](questions.md#g1-is-the-implementation-faithful-to-the-method) | faithful implementation | W1 |
| [G2](questions.md#g2-why-use-an-energy) | why use an energy (several correct answers) | W6 |
| [G3](questions.md#g3-how-does-error-scale-with-thinking) | error against thinking steps | W3, and C2 at v1.0 conditions |
| [G4](questions.md#g4-does-energy-verify-answers) | energy as a verifier | W4 |
| [G5](questions.md#g5-does-thinking-help-more-out-of-distribution) | thinking out of distribution | W2 |
| [G6](questions.md#g6-which-training-components-matter) | training components (not run in v1.0) | W5 |
| [G7](questions.md#g7-is-the-comparison-fair-at-equal-cost) | comparison at equal cost (equal parameters and equal steps only; equal cost in v1.1) | C1 at v1.0 conditions |

C1 to C10 compare the families ([questions.md, level
1](questions.md#level-1-comparison-of-families)); W1 to W6 are the template every family answers
for itself ([questions.md, level 2](questions.md#level-2-the-template-for-every-family)).

## Quantities

| Symbol | Meaning | Defined in |
|---|---|---|
| r | normalized distance from a regression prediction to the nearest correct branch y = ±√x; 0 on a branch | [method.md 16.7](method.md#167-metrics) |
| BPC | bits per character of the language model; lower is better | [method.md 18.6](method.md#186-metrics) |
| PSNR | peak signal-to-noise ratio of a denoised MNIST image, in dB, from one MSE over all test pixels; higher is better | [method.md 17.6](method.md#176-metrics) |
| ρ | Spearman rank correlation between the final energy and the error (G4) | [method.md 19.4](method.md#194-g4-energy-as-a-verifier) |
| fraction | best-of-N by lowest energy, with N the number of candidates (the M of method.md section 1): the improvement over random selection as a fraction of the oracle's improvement; 0 = no information, 1 = as good as the oracle, below 0 = worse than random (G4) | [method.md 19.4](method.md#194-g4-energy-as-a-verifier) |
| I | relative improvement from thinking, I = 1 − e₆₄ / e₁, with e_k the error after k steps (G5); I lies in (−∞, 1] and is below 0 when thinking makes the error worse | [method.md 20.2](method.md#202-relative-improvement-its-interval-and-the-outcome) |
| I_ID, I_OOD | I in distribution and out of distribution | [method.md 20.2](method.md#202-relative-improvement-its-interval-and-the-outcome) |
| Δ | I_OOD − I_ID per seed | [method.md 20.2](method.md#202-relative-improvement-its-interval-and-the-outcome) |
| ± | half-width of the 95% t-interval over seeds 0–4 | [method.md 20.2](method.md#202-relative-improvement-its-interval-and-the-outcome) |

## G5 outcomes

The seven outcomes of G5 and the texts the report writes for them are defined once, in [method.md
section 20.2](method.md#202-relative-improvement-its-interval-and-the-outcome).
