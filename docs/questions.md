# Research questions

The main question of the project, its scope and the reasons for it are in
[design.md](design.md#1-main-question-and-scope). Under equal training compute and identical
measurement, at small scale, how do families of text generation compare in quality, cost and
return on inference-time compute, and do the central claims of each family's paper hold?

This document has three parts:

- [A. Principles](#a-principles) gives the rules that every question and every answer follows.
- [B. Questions](#b-questions) gives the questions on two levels: the comparison of families (C1
  to C10) and one template that every family answers for itself (W1 to W6).
- [C. The v1.0 answers](#c-the-v10-answers) gives the questions G1 to G7 of v1.0, their measured
  answers, and how they map onto the new numbering.

The families, the protocol, the metrics and the versions are specified in [design.md](design.md).
The implementation of the autoregressive and EBT families and the v1.0 experiments are specified
in [method.md](method.md). The labels and quantities are defined in the [glossary](glossary.md).

## A. Principles

1. Equal training compute is the primary condition: equal parameters and equal training FLOPs at
   fixed compute marks ([design.md section 4.1](design.md#41-conditions)). Equal tokens and equal
   wall-clock time are secondary views. Equal steps, the condition of v1.0, is not a fair condition
   across families, because a step costs a different amount of compute in each family. For S1
   pretraining with two optimization steps, EBTs used 6.66 times the training FLOPs of a
   comparable Transformer++ ([EBT] Appendix D.5).
2. Every number is traceable to a result file in the repository and can be rerun with its code.
3. Before any measurement, each question records its hypothesis, its metric, its minimum
   meaningful effect (derived from the seed-to-seed spread) and its decision rule.
4. Negative results are published.
5. An answer is tied to a version and to that version's conditions. A later version adds a new
   answer next to the earlier one and never rewrites it.
6. No family is favoured. Every family receives the same budget for its own questions: the same
   number of search trials, seeds and ablated components. Every family also gets the same protocol
   and the same rules, and no family's results are reported with more prominence than another's.
   Time is not a constraint, and equality is the rule. Reason: a favoured family receives better
   design choices, more tuning effort and more prominent reporting, and each of these biases the
   comparison in its favour.
7. Every choice and every rejected option is documented with its references
   ([design.md](design.md)).

Principles 3 and 6 and the pre-registration lists of design.md section 4.9 apply in full from v1.1.
In v1.0 they held only in part. The decisions and revisions made after results had been seen are
listed in [Limitations and threats to validity](#limitations-and-threats-to-validity).

## B. Questions

The questions are on two levels. Level 1 compares the families on text only, with the shared
protocol and the shared metrics only. Level 2 is one template that every family answers for
itself: its own settings, its own paper's claims and its internals. A number is published once: a
level-2 answer does not repeat a level-1 number, and the reverse.

Every question below is planned. Its hypothesis, primary metric, decision rule, minimum meaningful
effect and Holm family are fixed before the first measurement of the version in which it is first
measured ([design.md section 4.9](design.md#49-pre-registration); principle 3). They are fixed in
that version's pre-registration list, or in an earlier list where stated (C8 is in the v1.1 list).
The metrics are defined in [design.md section 5](design.md#5-metrics). Full references, with
titles and dates, are in [design.md, References](design.md#references); bracketed keys such as
[EBT] refer to that list.

### Level 1: comparison of families

In the table, "fixed at v1.x start" means that the entry is an open item of that version's list
(design.md section 4.9). The Holm family of a question is its tests within one version (design.md
section 4.5).

| Question | What is compared | Primary metric | Decision rule | Holm family | Version |
|---|---|---|---|---|---|
| C1 | Quality at equal training compute | fixed at v1.1 start (likelihood and generation quality, given exact likelihoods, upper bounds and FLM without one) | fixed at v1.1 start | C1, per version | v1.1 |
| C2 | Quality against inference-time compute | generation quality at the common entropy target | fixed at v1.1 start | C2, per version | v1.1; the comparison of distilled few-step models in v1.3 |
| C3 | Self-scoring | the error of the candidate chosen by the family's own score, against random and oracle choice; how better candidates are judged fixed at v1.1 start, per family | fixed at v1.1 start | C3, per version | v1.1 |
| C4 | Out of distribution | fixed at v1.1 start | fixed at v1.1 start | C4, per version | v1.1 |
| C5 | Ranking against the amount of unique data | fixed at v1.1 start | fixed at v1.1 start | C5, per version | v1.1 |
| C6 | Cost at inference | latency at two points per family | fixed at v1.1 start | C6, per version | v1.1 |
| C7 | Training stability and sensitivity | the loss of the primary metric across the common learning-rate grid, and the fraction of diverged runs | fixed at v1.1 start | C7, per version | v1.1 |
| C8 | Infilling a missing middle span | fixed at v1.1 start | fixed at v1.1 start | C8, per version | v1.2 |
| C9 | Backbone: does DeepCrossAttention change the ranking? | that of the pilot, fixed at v1.2 start | fixed at v1.2 start (the pilot's decision rule) | C9, per version | v1.2 (pilot) |
| C10 | Scale: which family gains more from size? | fixed at v1.4 start | fixed at v1.4 start | C10, per version | v1.4 |

"Version" is the first version in which the question is measured. C1 to C7 are measured from v1.1
for the autoregressive and EBT families, and for each new family in the version in which it
enters. C8 starts in v1.2: the EBT used here is causal and cannot infill, so the question becomes
meaningful when the diffusion families enter. The autoregressive family then takes part with its
fill-in-the-middle variant. C9 is a pilot in v1.2 and C10 is measured in v1.4. A question is
answered again in each version that adds a family (principle 5). Each version is described in
[design.md section 6](design.md#6-versions).

Separately from this table, C1 and C2 also have answers at v1.0 conditions (equal parameters and
equal steps), from G7 and G3 ([section C](#mapping-to-the-new-numbering)). They are not answers
under the protocol.

#### C1. Quality at equal training compute

Question: at equal parameters and equal training FLOPs, how do the families compare in likelihood
and in generation quality?

Measured: the likelihood and the generation-quality profile at matched entropy, at each compute
mark ([design.md section 5](design.md#5-metrics)). The likelihood is in exact bits per character,
and for diffusion families it is an upper bound, written "≤".

Scope: the answers of v1.1 to v1.3 hold at the single model size used, which is fixed in the v1.1
list. The compute-optimal frontier of v1.4 (C10) answers the general question ([design.md section
4.1](design.md#41-conditions)).

#### C2. Quality against inference-time compute

Question: how does quality change as each family spends more compute at inference, with its own
knob?

The x-axis is inference FLOPs per generated character, and each family's knob is mapped onto it,
because one step or iteration costs different compute in each family. The knob grid 1, 2, 4, …, 64
is shared. Diffusion's point at the sequence length is reported as an extra point of that family
([design.md section 3](design.md#3-axes)). The knobs are these:

- EBT: the number of thinking steps;
- diffusion and flow families: the number of sampling steps;
- recurrent depth: the number of iterations of the recurrent core;
- autoregressive models: best-of-N, with selection by the model's own likelihood.

Decisions, each with its reason:

- The main comparison is between base models on the shared knob grid. Reason: a distilled
  few-step model compared with undistilled ones at 1 to 4 steps would confound distillation with
  family.
- The primary metric is generation quality, the profile of [design.md section
  5](design.md#5-metrics). Reason: the likelihood does not move with best-of-N or with the number
  of diffusion steps, so it cannot show the return on inference compute.
- At every knob value, each family's sampling temperature is set so that the entropy of its
  samples equals one common target, the entropy of held-out text. How the temperature is set is
  fixed in the v1.1 list ([design.md section 4.9](design.md#49-pre-registration)). Reason:
  selection by the model's own likelihood lowers the entropy of the samples, and a less varied
  sample would otherwise look better.
- Distilled few-step models: in v1.2 FMLM is tested inside FLM's own template (W2, W3), as a claim
  of its paper, and is not part of the main comparison. In v1.3, with Duo and its own
  distillation DCD (from the Duo paper), C2 adds a separate comparison of the distilled few-step
  models, FMLM against Duo-DCD. Only a distillation from the family's own paper is included. MDLM's
  SDTT comes from a separate paper and is not included ([design.md section
  2.7](design.md#27-rejected-options)).

C2 includes a check shared by every family: whether extra iterations change the output at all. The
recurrent-depth paper reports a training run whose model learned to ignore its recurrent state,
with the same validation perplexity for 1 and 32 recurrences ([Huginn], "Bad Run 2").

#### C3. Self-scoring

Question: does the model's own score select better candidates? The score is the energy for the
EBT, the likelihood for the autoregressive family and the family's own confidence for the others.
The confidence score of each family, and how better candidates are judged, are fixed in the list
of the version in which the family enters ([design.md section 4.9](design.md#49-pre-registration)).

#### C4. Out of distribution

Question: how do the families compare on text unlike the training text? The out-of-distribution
text and its mapping onto the training vocabulary are fixed by a rule decided before training
([design.md section 4.2](design.md#42-data)).

#### C5. Unique data

Question: does the ranking of the families change with the amount of unique data? The number of
passes over each amount follows from the compute marks fixed in the v1.1 list ([design.md section
4.9](design.md#49-pre-registration)). The amounts, about 1M, 10M and 90M characters, are all
subsets of text8, so that only the amount of unique text changes between them, not its source
([design.md section 3](design.md#3-axes)). Tiny Shakespeare is the development set and the corpus
of the v1.0 regime, not a point of this axis. Related work:

- [Muennighoff] studies repeated data for autoregressive models;
- [Prabhudesai] and [Ni] compare diffusion with autoregressive models when data are limited.

#### C6. Cost

Question: what does each family cost at inference? The primary metric is latency: the time to
first character and the throughput, by batch size. It is measured at two points per family. The
first is knob value 1. The second is the smallest knob value at which the family reaches a common
quality level, if it reaches it, and the level is fixed in the v1.1 list ([design.md section
4.9](design.md#49-pre-registration)). Each family gets the standard inference optimizations of its
own paper or official code (for the autoregressive model, the key-value cache), declared before
measuring. Peak memory is supplementary: it is reported with the same conditions, with no decision
on it. Energy per character is supplementary as well ([design.md section 5](design.md#5-metrics)).
Reason: the latency of a diffusion model at 1 step and at L steps differs by orders of magnitude,
so a single point would decide the answer by its choice. Inference optimizations also change the
ranking ([2510.18480]).

#### C7. Training stability and sensitivity to hyperparameters

Question: how stable is each family's training, and how much does its result depend on its
hyperparameters? It is measured on one common relative grid. After tuning, each family is trained
at its best settings with the learning rate multiplied by 1/4, 1/2, 1, 2 and 4, with the same
seeds. C7 reports the loss of the primary metric across this grid and the fraction of diverged
runs. The spread across tuning trials is not the C7 measure. Reason: families tune different
hyperparameters, with no common unit and different grid widths. The learning rate exists in every
family, and a common relative grid makes the spread comparable.

#### C8. Infilling

Question: how well does each family fill a missing middle span? The autoregressive family takes
part with a fill-in-the-middle variant ([FIM]), so the comparison is between methods of infilling
and not between having and lacking the ability. C8 starts in v1.2, when the diffusion families
enter, because the EBT used here is causal and cannot infill.

#### C9. Backbone

Question: does DeepCrossAttention ([DCA]) change the ranking of the families, or help some
families more than others? A pilot with 5 seeds comes first, as for every comparison. Its
families, seeds and threshold are fixed before the pilot. The full grid crossed with every family
is run only if the effect exceeds the threshold, and that decision does not depend on one family
only. The decision rule of the pilot is in the v1.2 list ([design.md section
4.9](design.md#49-pre-registration)). The reason for testing this backbone change only is in
[design.md section 3](design.md#3-axes).

#### C10. Scale

Question: which family gains more from size? It is measured as each family's compute-optimal
frontier, with several sizes and several budgets per family. C10 begins in v1.4 ([design.md
section 6](design.md#6-versions)).

### Level 2: the template for every family

Every family answers the same six questions about itself. A family enters the project only with
its full W1–W6 template. The rule applies from v1.1, and the autoregressive and EBT families of
v1.0 complete their templates in v1.1.

| Question | Content |
|---|---|
| W1 | Faithful implementation against the family's paper and official code. |
| W2 | The central claims of the paper (abstract and main results). Those not testable at this scale are listed as such. |
| W3 | The family's inference-time knob in detail. |
| W4 | The family's own scoring of answers. |
| W5 | Training-component ablations: the components the paper names as important, with the same budget for every family. |
| W6 | The family's distinctive capability. |

The autoregressive family is the reference family:

- W1 checks the implementation against the Transformer++ recipe that the EBT paper uses as its
  baseline. The source, its components and the differences of the v1.0 Transformer are in
  [design.md section 2.1](design.md#21-autoregressive-transformer).
- W2 tests no claim, because the family is the reference.
- W3 is best-of-N, with selection by its own likelihood.
- W4 uses its own likelihood as a score.
- W5 runs ablations under the equal budget.
- W6 is streaming generation with a key-value cache (time to first character).

W5 of recurrent depth includes its two control models, a non-looped model with the same parameters
and one with the same FLOPs, trained from its ablation budget ([design.md section
2.5](design.md#25-recurrent-depth)).

W6 of the other families:

- EBT: several correct answers for one input;
- diffusion: revisable tokens and parallel decoding (infilling is published only under C8);
- flow: one-step generation;
- recurrent depth: how many iterations are needed and when to stop.

Tasks from a family's paper are included by the same rule for every family: they test a central
claim of that paper and fit the equal budget. By this rule the MNIST task of v1.0 is an EBT task:
it follows the paper's image denoising ([EBT] §4.3) and belongs to the EBT family's W2. The
regression task has no counterpart in the EBT paper (method.md section 16.1). It is this project's
test of the EBT family's distinctive capability, several correct answers for one input (W6). Each
other family takes tasks from its own paper by the same rule.

## C. The v1.0 answers

v1.0 has seven questions, G1 to G7, about the autoregressive and EBT families:

| Question | State in v1.0 |
|---|---|
| G1 to G5 | answered by their rules |
| G6 | not run ([Status of G6](#status-of-g6)) |
| G7 | answered in part: at equal parameters and equal steps only; the part at equal cost is planned for v1.1 |

Each answer reports the result whether or not the expected effect appears. The answers below are
the v1.0 answers, and the new numbering does not change them (principle 5).

The three tasks of v1.0 are:

- multimodal regression with targets y = +√x or y = −√x;
- MNIST denoising with a small patch transformer;
- a character-level language model on Tiny Shakespeare.

Method terms (S1, S2, full unroll, step size, Langevin noise, replay buffer) are defined in
[method.md](method.md). Each answer gives its key numbers as the mean over seeds 0–4 ± the
half-width of the 95% t-interval. G3, G4 and G5 link to the full tables in method.md sections
19.5 and 20.5, and the text part of G7 links to section 18.9. The other full tables are in the
result files, linked where they are cited.

### Mapping to the new numbering

| v1.0 question | Family | New numbering |
|---|---|---|
| [G1](#g1-is-the-implementation-faithful-to-the-method) Faithful implementation | EBT | W1 |
| [G2](#g2-why-use-an-energy) Why use an energy | EBT | W6 (several correct answers) |
| [G3](#g3-how-does-error-scale-with-thinking) Error against thinking steps | EBT | W3, and C2 at v1.0 conditions |
| [G4](#g4-does-energy-verify-answers) Energy as a verifier | EBT | W4 |
| [G5](#g5-does-thinking-help-more-out-of-distribution) Thinking out of distribution | EBT | W2 (a main result of the paper: §4.1.3, Figure 7) |
| [G6](#g6-which-training-components-matter) Training components | EBT | W5 |
| [G7](#g7-is-the-comparison-fair-at-equal-cost) Comparison at equal cost | autoregressive and EBT | C1 at v1.0 conditions (equal parameters and equal steps only) |

The MNIST task follows the image denoising of the EBT paper ([EBT] §4.3) and belongs to the EBT
family's W2. The regression task has no counterpart in the EBT paper (method.md section 16.1). It
tests the EBT family's W6. The other families take tasks from their own papers by the same rule
([section B](#level-2-the-template-for-every-family)).

### The EBT paper's claims (W2)

The central claims of the EBT paper's abstract ([EBT], v1, the only version on arXiv), whether
each is testable at the scale of this project, and where it is or will be tested:

| Claim of the abstract | Testable at this scale | v1.0 | Later |
|---|---|---|---|
| During training EBTs scale faster than Transformer++, "achieving an up to 35% higher scaling rate with respect to data, batch size, parameters, FLOPs, and depth". | from v1.4: it needs several sizes and budgets | not tested: one model size, one data size | v1.4, the compute-optimal frontier (C10) |
| "During inference, EBTs improve performance with System 2 Thinking by 29% more than the Transformer++ on language tasks." | yes | G3 measures the text EBT's error against thinking steps; the v1.0 baseline has no inference-time knob | v1.1: C2 and the EBT's W3, with best-of-N as the autoregressive family's knob |
| "EBTs outperform Diffusion Transformers on image denoising while using fewer forward passes." | partially tested in v1.0 | MNIST denoising compares the EBT with a one-pass denoiser of the same patch-transformer architecture (method.md section 17.3), not with a multi-step diffusion transformer: test PSNR 27.00 ± 0.03 dB for the EBT (S1, 2 inference steps) against 27.54 ± 0.04 dB for the denoiser, mean over 5 seeds ± half-width of the 95% t-interval ([`results/mnist/denoising/full/table.md`](../results/mnist/denoising/full/table.md)) | a full test against a diffusion transformer is not planned, because the comparison of families is done on text |
| "EBTs achieve better results than existing models on most downstream tasks given the same or worse pretraining performance, suggesting that EBTs generalize better than existing approaches." | no: not testable with a character-level model | not tested: no downstream tasks | related question: C4 (out of distribution), from v1.1 |

G5 tests a main result of the paper, a larger gain from thinking out of distribution for language
(§4.1.3, Figure 7), which W2 covers as a central claim of the paper. In v1.0 this effect is not
observed at this scale ([G5](#g5-does-thinking-help-more-out-of-distribution)).

### Limitations and threats to validity

The v1.0 answers hold under these conditions only, and each point limits what they show:

- Small scale and one model size per task. On text the EBT has 6 328 641 and the baseline 6 328 064
  parameters, the size chosen by the budget rule of method.md section 18.7. On MNIST they have
  828 033 and 829 968
  ([`results/mnist/denoising/full/table.md`](../results/mnist/denoising/full/table.md)). On
  regression all models have one size ([G2](#g2-why-use-an-energy)). Whether the results change
  with size is not tested.
- One dataset per task: y = ±√x for regression, MNIST at noise 0.1 for denoising, Tiny Shakespeare
  for text (method.md sections 16.1, 17.1 and 18.1).
- Equal steps rather than equal compute. On text both models trained for T = 2 000 steps on the
  same windows in the same order per seed. Per fresh training window the EBT made 4.48715 forward
  and first-order backward passes and 1.49975 second-order passes on average over the seeds. The
  baseline made 1 forward and 1 backward pass ([G7](#g7-is-the-comparison-fair-at-equal-cost);
  method.md section 18.9).
- Tiny Shakespeare, a regime of many passes over a small corpus. Each text final run trains on
  2 000 steps × 32 windows × 128 characters = 8 192 000 characters
  (`results/text/lm/full/*-seed*.json`, fields `config.training` and `config.sequence_length`).
  That is about 8.2 passes over the 1 003 854-character training split (method.md section 18.1).
- Possible under-training. On text the best checkpoint by validation BPC within T = 2 000 steps
  was at the last step on two of the five EBT seeds (method.md section 18.9). On MNIST the
  validation Smooth L1 was still falling at the end of training on 4 of 5 seeds for both models
  ([`results/mnist/denoising/full/table.md`](../results/mnist/denoising/full/table.md)).
- Single-seed searches and reused searches. Every configuration of every search was trained once,
  with seed 100 (method.md sections 16.6, 17.5 and 18.5). The final runs of all three tasks used
  searches written by earlier code. The regression and MNIST searches predate code fingerprints,
  and the text search has a fingerprint that differs by one line of `generate()`. Every final
  result file lists them in `code.reused_stale_files` (method.md section 16.6;
  [reproducing.md](reproducing.md#rerunning-the-final-runs-without-overwriting-v10)).
- The confidence intervals cover training randomness only (initialization, batch order, initial
  candidates, thinking noise). The data are fixed, so data sampling is not covered
  ([G2](#g2-why-use-an-energy)).
- Evaluation without Langevin noise. Every EBT is evaluated with σ = 0. The authors' text code
  keeps the training noise at evaluation (method.md section 7 and D12). An evaluation with the
  training noise is item 27 of the v1.1 list ([design.md section
  4.9](design.md#49-pre-registration)).
- Decisions and revisions made after results had been seen, with dates and reasons in the
  [changelog](../CHANGELOG.md#v10):
  - On 19 November 2026 four models were added to G2, after the first G2 results of (a), (b),
    (c1) and (c2) had been seen. In those results (b) and (c2) predicted near the mean of the two
    branches, and (c1) diverged. (c3) was added as the remedy for the (c1) divergence. (a2), (c4)
    and (c5) were added so that G2 compares the main ways of obtaining multimodal predictions,
    rather than the EBT against one regressor.
  - The hypotheses for (a2), (c4) and (c5) were written after a 200-step trial run of each.
  - Before the measurement of 21 November 2026 the wording of item 3 of the text budget rule
    changed, after its application to the measurements of 20 November 2026 had been seen
    (method.md section 18.7).
  - On 21 November 2026 the G3 range rule and the G4 rules for undefined fractions and non-finite
    values were revised, after the first regression and MNIST G3 and G4 results had been seen.
  - On 23 November 2026 the G5 rule was refined and its outcome texts were set. One
    out-of-distribution value had been measured and seen by then: the text final runs' BPC on the
    out-of-distribution block at the trained 4 steps.
- One machine, an Apple M5 with 16 GB. Regression ran on the CPU, where a rerun with the same torch
  version on the same platform is byte-identical. MNIST and text ran on MPS, whose training runs
  are not bit-reproducible ([numerics.md, Reproducibility](numerics.md#reproducibility)).

v1.1 changes these conditions ([design.md section 6.1](design.md#61-from-v10-to-v11)).

### Status of G6

G6 (component ablations) was not run in v1.0. Under the protocol of v1.1, ablations are W5
([level 2](#level-2-the-template-for-every-family)). They run on text8 at equal compute with the
same W5 budget for every family (item 21 of the v1.1 list, [design.md section
4.9](design.md#49-pre-registration)). On Tiny Shakespeare each v1.0 text run makes about 8.2
passes over about 1M training characters ([Limitations and threats to
validity](#limitations-and-threats-to-validity)). Whether results change with repeated data is
itself a question (C5; [Muennighoff], [Prabhudesai], [Ni]). G6 moves to W5. It is run for the
autoregressive and EBT families in v1.1 (planned) and for each new family in the version in which
it enters. Its v1.0 design is kept below as written.

### G1. Is the implementation faithful to the method?

Question: does the code implement the thinking loop and its training as specified in method.md?

Hypothesis: the implementation passes every check below within the tolerance stated in its test.

Measured:

- the candidate gradient ∇_ŷ E and the parameter gradient of the training loss in each mode (S1,
  S2, full unroll), compared with central finite differences in float64 on CPU;
- convergence of the thinking loop on a quadratic energy, where gradient descent with a step size
  below 2 / L (L the largest curvature) must converge to the minimizer;
- convergence on a double well, where it must reach the minimum of the basin it starts in;
- for a one-dimensional quadratic energy, the parameter gradients in the three modes, compared
  with values derived by hand;
- the gradient path under S2 truncation stated in method.md section 6.2 and listed as U1.

Answer: the tests run in continuous integration and pass. The tolerance against finite
differences is a relative error of at most 1e-6 (`tests/test_gradients.py`, `TOLERANCE`), and
each other test states its own. A worked example of one thinking step whose numbers can be checked
by hand is not part of v1.0. It is item 26 of the v1.1 pre-registration list ([design.md section
4.9](design.md#49-pre-registration)).

### G2. Why use an energy?

Question: when a regression input has several correct answers, does a standard regressor predict
their mean while the EBT finds one of the answers? The models are all of the same size (method.md
section 16):

- (a) a feed-forward regressor trained with mean squared error;
- (b) an EBT trained with the paper's objective: S2 thinking and mean squared error of the final
  candidate;
- (c1) an EBT trained with the contrastive energy loss alone;
- (c2) an EBT trained with mean squared error plus the contrastive loss;
- (c3) an EBT trained with the contrastive loss plus an energy-magnitude penalty ([Du-Mordatch];
  method.md section 16.4).

Three further models apply methods from outside the EBT paper, each implemented from its primary
source (method.md section 16.8):

- (a2) a mixture density network ([Bishop]), which predicts a distribution over y instead of one
  value;
- (c4) an implicit behavioural cloning energy ([IBC]): InfoNCE training with uniform
  counter-examples and derivative-free inference;
- (c5) an IRED energy ([IRED]): a sequence of noised energy landscapes, trained with a denoising
  loss on the energy gradient plus a contrastive loss, and annealed gradient inference.

(c3), (a2), (c4) and (c5) were added after the first results of the other four had been seen
([Limitations and threats to validity](#limitations-and-threats-to-validity)).

Hypothesis, from the derivation in method.md section 16.5:

- (a) and (b) reach the mean-predictor floor and predict near 0;
- (c1) lands on the branches, with P(sign ŷ_final = sign ŷ_0) well above 0.5;
- (c2) is open;
- (c3) is open: the penalty bounds the energy scale, and the measurement shows whether thinking
  then ends on the branches.

The hypotheses for (a2), (c4) and (c5) were written after a 200-step trial training run of each,
in which (a2) and (c4) already ended on the branches. They were written after their full-tier
search had started and before any full-tier G2 metric was read:

- (a2): with at least two kernels the likelihood is maximized by kernels on both branches, so the
  most-probable-branch readout (eq. 49 of [Bishop]) lies on a branch. Which branch it picks for a
  given x depends on small differences between the mixing coefficients, so the branch split may be
  far from 0.5.
- (c4): InfoNCE shapes the energy around both branches, so DFO, which needs no gradient steps,
  ends on a branch.
- (c5): the annealed landscapes and the accept-only-lower steps avoid the overshooting seen for
  (c1) and (c3), so inference ends on a branch.

Measured on y = ±√x, with each sign equally likely, on the test split:

- the mean normalized distance r from each prediction to the nearest correct branch;
- the fraction of predictions with r ≤ 0.1;
- the fraction of predictions on the positive branch;
- for the EBTs, P(sign ŷ_final = sign ŷ_0), which shows how the branch reached depends on the
  initial candidate;
- for the EBTs, the mean |ŷ_final − ŷ_0|, which shows whether thinking moves the candidate;
- the test mean squared error, against the mean-predictor floor 2.125 (method.md section 16.5);
- for (c1) and (c3), on seed 0, the mean |∂E/∂ŷ| at the targets and at ŷ = 0, which checks the
  explanation of the (c1) divergence by an unbounded energy scale.

Hyperparameters are chosen on validation data by each model's own training objective, by a rule
written before the measurement (method.md section 16.6). Each model is trained with 5 seeds. The
data are fixed, so the confidence intervals cover training randomness (initialization, batch
order, initial candidates, thinking noise) and not data sampling.

Answer ([`results/regression/g2/full/table.md`](../results/regression/g2/full/table.md); every
metric of every model is in that table):

| Model | Hypothesis | mean r | fraction r ≤ 0.1 | test MSE |
|---|---|---|---|---|
| (a) | supported: near the mean | 0.973 ± 0.002 | 0.000 ± 0.000 | 2.099 ± 0.000 |
| (b) | supported: near the mean | 0.979 ± 0.001 | 0.000 ± 0.000 | 2.099 ± 0.000 |
| (c2) | open; near the mean | 0.978 ± 0.002 | 0.000 ± 0.000 | 2.099 ± 0.000 |
| (c1) | not supported: diverged | 58.523 ± 18.663 | 0.001 ± 0.001 | 7637.797 ± 4416.252 |
| (c3) | open; not on the branches | 3.035 ± 0.315 | 0.000 ± 0.000 | 59.113 ± 13.369 |
| (a2), (c4), (c5) | supported: on a branch | | 1.000 ± 0.000 each | |

- The test MSE of (a), (b) and (c2) equals the mean of y² on the test split (2.0985), the error of
  predicting 0.
- (c1) did not land on the branches. Its P(sign ŷ_final = sign ŷ_0) is 0.241 ± 0.081, not above
  0.5.
- (a2), (c4) and (c5) return a single branch for each x. For (a2), on seed 0, the prediction is on
  the positive branch for x below about 3.0 and on the negative branch above. The change lies
  between x = 2.988 and x = 3.015 among the 512 test points of `mdn-seed0.json`, field
  `artifacts.predictions`.
- For (c5), P(sign ŷ_final = sign ŷ_0) is between 0.49 and 0.52 on every seed (field
  `metrics.test.initial_sign_agreement`). The branch reached does not follow the sign of the
  initial candidate.

Fraction of test predictions on the positive branch per seed (`<model>-seed<s>.json` in
`results/regression/g2/full/`, field `metrics.test.positive_branch`):

| Model | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| (a2), readout eq. 49 | 0.743 | 0.729 | 0.682 | 0.703 | 0.871 |
| (c4) | 0.759 | 0.194 | 0.918 | 0.828 | 0.981 |
| (c5) | 0.150 | 0.626 | 0.713 | 0.845 | 1.000 |

The EBTs whose final branch depends on the initial candidate, (c1) and (c3), do not end on the
branches. Only (a2) returns both branches for one input when sampled: with 32 candidates its
both-branch coverage is 1, against 0.08813 ± 0.05304 for (c4) and at most 0.00122 for every other
model. The coverage table for every model and N is in [method.md section
16.8](method.md#measured-one-branch-for-each-x).

### G3. How does error scale with thinking?

Question: how does task error change as the number of thinking steps at inference increases?

Hypothesis: none was recorded in v1.0. The range rule of method.md section 19.2 classifies each
curve.

Measured: the task error against the number of inference steps for each of the three tasks. The
error is the distance to the nearest branch for regression, the PSNR for denoising and the bits
per character for the language model. The same trained models are used across step counts, with
the training mode and step range recorded. The step counts are 1, 2, 4, 8, 16, 32 and 64, with 5
seeds (method.md section 19.2). The rule divides the step counts into ranges where the error
falls, stays flat (a change smaller than the spread across seeds) or rises, and reports a range
that does not occur as absent.

For regression, an inference-time experiment on the trained (c3) models of G2 also tests whether
stable inference recovers the branches from the (c3) energy (method.md section 19.3). Without
retraining, inference is repeated with smaller step sizes, more steps, and the accept-only-lower
step of method.md section 16.8. The step size is not tuned on the G2 metrics, and every setting
tried is reported.

Answer ([`results/regression/g3_g4/full/table.md`](../results/regression/g3_g4/full/table.md),
[`results/mnist/g3_g4/full/table.md`](../results/mnist/g3_g4/full/table.md) and
[`results/text/g3_g4/full/table.md`](../results/text/g3_g4/full/table.md)). The tables give every
step count. The MNIST and text curves are also in [method.md section 19.5](method.md#195-results),
and the regression r per step only in the regression table. No range is undefined.

| Model | Error | Ranges |
|---|---|---|
| regression (b) | branch distance r | rising 1–4; flat 4–64 |
| regression (c1) | branch distance r | falling 1–16; flat 16–64 |
| regression (c2) | branch distance r | rising 1–4; flat 4–64 |
| regression (c3) | branch distance r | rising 1–2; flat 2–64 |
| MNIST EBT | PSNR (dB) | rising 1–64 |
| text EBT | bits per character | flat 1–64 |

The range labels describe the error, so for MNIST "rising" means that the PSNR falls.

| Model | Step 1 | Trained steps | Step 64 |
|---|---|---|---|
| MNIST EBT, PSNR (dB) | 27.6317 ± 0.0536 | 27.0003 ± 0.0307 (2 steps) | 17.1189 ± 0.3812 |
| text EBT, bits per character | 2.3208 ± 0.0218 | 2.3318 ± 0.0222 (4 steps) | 2.3344 ± 0.0215 |

The text curve differs between G3 and G5. G3's range rule compares the text curve against the
spread across seeds and calls it flat. G5 compares 1 and 64 steps within each seed. On all 5 seeds
the bits per character at 64 steps are higher than at 1 step (`results/text/g5/full/g5.json`,
`bpc_id_1` and `bpc_id_64`). The paired comparison therefore detects a small degradation, I_ID
−0.0095 ± 0.0060 ([G5](#g5-does-thinking-help-more-out-of-distribution)).

(c3) inference experiment
([`results/regression/g3_g4/full/table.md`](../results/regression/g3_g4/full/table.md)), fraction
of test x with r ≤ 0.1 after 64 steps:

| α divisor | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| plain step | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| accept-only-lower step | 0.092 ± 0.006 | 0.093 ± 0.006 | 0.098 ± 0.006 | 0.106 ± 0.011 | 0.116 ± 0.009 |

### G4. Does energy verify answers?

Question: does the energy of a final candidate track its error, and does choosing the
lowest-energy candidate among several improve the answer?

Hypothesis: none was recorded in v1.0. The answer reports the correlation and the selection
gain with their intervals.

Measured, on each task (method.md section 19.4):

- the Spearman rank correlation between final energy and error, over individual examples (for the
  language model, over individual positions);
- the error of best-of-N selection by lowest energy, against random selection among the same N
  candidates and against oracle selection (the candidate with the lowest true error), for N = 1,
  2, 4, 8, 16 and 32. Here N is the number of candidates, not of steps;
- for regression, the both-branch coverage: the fraction of test x for which N candidates include
  predictions on both branches (r ≤ 0.1 on each).

The improvement of lowest-energy selection over random selection is reported as a fraction of the
improvement of oracle selection over random selection. A fraction near 0 means that the energy
carries no information for selection, and 1 means that it selects as well as the oracle. How the
MNIST candidates are drawn is defined in method.md section 19.4. The MNIST error there is the
per-image MSE on the 0–255 pixel scale, unlike the test MSE of the denoising table, which is on
the [−1, 1] scale.

Answer (`results/<task>/g3_g4/full/`; Spearman over 4 096 test x for regression, 10 000 test
images for MNIST and 55 680 test positions for text, per seed):

| Model | Spearman ρ | fraction, N = 2 | fraction, N = 32 |
|---|---|---|---|
| regression (b) | −0.1694 ± 0.1290 | −0.065 ± 0.073 | 0.003 ± 0.050 |
| regression (c1) | 0.8650 ± 0.0447 | 0.992 ± 0.006 | 1.000 ± 0.000 |
| regression (c2) | 0.0168 ± 0.2449 | −0.096 ± 0.150 | −0.020 ± 0.044 |
| regression (c3) | 0.1674 ± 0.3707 | 0.135 ± 0.474 | −2.049 ± 2.490 |
| regression (c4) | −0.2358 ± 0.1165 | 0.077 ± 0.098 | 0.020 ± 0.040 |
| regression (c5) | 0.0850 ± 0.4003 | −1.000 (1 of 5 defined) | −0.558 (1 of 5 defined) |
| MNIST EBT | 0.6124 ± 0.0083 | −0.119 ± 0.033 | −0.379 ± 0.025 |
| text EBT | 0.0482 ± 0.2094 | −0.399 ± 0.152 | −0.450 ± 0.242 |

Only (c1), the model that diverged, ranks its candidates by energy as well as the oracle. On MNIST
and text, selection by lowest energy is worse than random selection at every N. The fraction at
every N is in [method.md section 19.5](method.md#195-results). The errors of each selection at
every N are in the `table.md` of each task, for MNIST
[`results/mnist/g3_g4/full/table.md`](../results/mnist/g3_g4/full/table.md). The PSNR of the MNIST
lowest-energy and oracle candidates at every N is in
[`results/mnist/g3_g4/full/summary.json`](../results/mnist/g3_g4/full/summary.json)
(`lowest_energy_psnr`, `oracle_psnr`).

### G5. Does thinking help more out of distribution?

Question: is the improvement from thinking longer larger on data unlike the training data? The
paper reports a larger gain out of distribution for language ([EBT] §4.1.3, Figure 7). Its
Figure 7 measures "Max Thinking", which is thinking longer together with self-verification. G5
measures thinking longer only (method.md section 20.2). The paper's image experiment (§4.3) tests
at noise level 0.2 after training at 0.1 and compares the EBT with a diffusion transformer. It
makes no claim about thinking out of distribution, so the MNIST part of G5 is this project's test,
at the paper's test level.

Hypothesis: for text, the paper's effect, a larger improvement from thinking out of distribution.
MNIST and regression have no hypothesis from the paper.

Measured: the relative improvement from thinking of the paper's Definition C.1, from 1 inference
step to 64 steps, on the final checkpoints of seeds 0–4, in and out of distribution (method.md
section 20). Per checkpoint, I = 1 − e₆₄ / e₁, with e_k the error over the whole split after k
steps. The error is the mean branch distance r for regression, the MSE behind the PSNR for
denoising and the perplexity 2^BPC for the language model. I lies in (−∞, 1], so halving the error
gives 0.5 and doubling it gives −1. Δ = I_OOD − I_ID per seed. With 1 step as the start, Δ
includes an effect of the starting point, which the errors at the trained step count show
(method.md section 20.2).

The out-of-distribution data were fixed on 23 November 2026, before any out-of-distribution G3
value was computed. One out-of-distribution value had been measured and seen by then: the text
final runs' BPC on the out-of-distribution block at the trained 4 steps (method.md section 20):

- denoising: the 10 000 test images at noise levels 0.15, 0.2 (primary) and 0.3, against
  training at 0.1;
- language model: the 55 770-character block of Pride and Prejudice (method.md section 18.1);
- regression: x in [4, 9), against training x in [0.25, 4) (method.md section 16.1).

The models are the EBTs whose inference is the thinking loop: regression (b), (c1), (c2) and (c3),
the MNIST EBT and the text EBT. The models without a step count are not included (method.md
section 20.4). Each model gets one outcome of the rule of [method.md section
20.2](method.md#202-relative-improvement-its-interval-and-the-outcome). For MNIST the outcome is
that of level 0.2, and 0.15 and 0.3 are reported without one.

Answer (`results/<task>/g5/full/table.md` and `g5.json`; the full tables are in [method.md section
20.5](method.md#205-results)):

| Task, model | Outcome | I_ID | I_OOD | Δ |
|---|---|---|---|---|
| text EBT | 3 | −0.0095 ± 0.0060 | −0.0003 ± 0.0074 | 0.0092 ± 0.0101 |
| MNIST EBT, noise 0.2 | 3 | −10.2741 ± 0.9499 | −0.2149 ± 0.1340 | 10.0592 ± 0.8272 |
| regression (b) | 3 | −0.2932 ± 0.0208 | −0.0795 ± 0.0578 | 0.2137 ± 0.0726 |
| regression (c2) | 3 | −0.2539 ± 0.0169 | −0.0113 ± 0.0400 | 0.2426 ± 0.0410 |
| regression (c3) | 3 | −0.1150 ± 0.0844 | −0.9601 ± 2.1183 | −0.8451 ± 2.0633 |
| regression (c1) | 4 | 0.1563 ± 0.1137 | −0.6714 ± 0.5500 | −0.8277 ± 0.4422 |

- Text. The report words the outcome as "Thinking degrades the result in distribution and out of
  distribution on average; the relative degradation out of distribution is not measurably
  different. This is not the paper's effect." In plain terms, thinking degrades the result in
  distribution on its interval. Out of distribution it has no measurable effect, with a mean
  slightly below 0. The effect the paper reports for language is not observed at this scale.
- MNIST at noise 0.2. The report words the outcome as "Thinking degrades the result in both; the
  relative degradation out of distribution is smaller." Δ is above 0 because the degradation is
  smaller, not because thinking helps.
- Regression. For (b) the degradation is on both intervals and smaller out of distribution. For
  (c2) it is on its interval in distribution and on its mean only out of distribution, and smaller
  out of distribution. For (c3) it is on its interval in distribution and on its mean only out of
  distribution, and the difference is not measurable. (c1), the model that diverged, degrades out
  of distribution only.

The rule compares 1 step with 64 steps, the comparison fixed in method.md section 20.2, and the
error at the trained step count is reported for reference only. On text this paired comparison
detects the small degradation that G3 reports as flat
([G3](#g3-how-does-error-scale-with-thinking)).

Observations, descriptive only: the rule does not decide them, and they are not part of the
answer. MNIST PSNR in dB, means over seeds (the intervals are in method.md section 20.5):

| Data | 1 step | 2 steps (trained) | 64 steps | I_OOD |
|---|---|---|---|---|
| in distribution, noise 0.1 | 27.6317 | 27.0003 | 17.1189 | |
| noise 0.15 | 23.6533 | 25.0057 | 17.0532 | −3.5816 |
| noise 0.2 | 17.6727 | 21.0193 | 16.8408 | −0.2149 |
| noise 0.3 | 10.8204 | 12.4526 | 14.2262 | 0.5412 |

- (a) Out of distribution, the PSNR rises from 1 step to the trained 2 steps at every level, while
  in distribution it falls slightly.
- (b) I_OOD increases with the noise level, but at 64 steps every level ends in about the same
  range. Part of the gain at noise 0.3 therefore comes from its very low start at 1 step.
- (c) These observations motivate a question for v1.1 (planned): the effect of thinking at small
  step counts and its dependence on the distance from the training distribution. It is item 25
  of the v1.1 pre-registration list ([design.md section 4.9](design.md#49-pre-registration)),
  with status open, and is not answered here.

### G6. Which training components matter?

Question: how much does each component of training contribute to thinking?

Hypothesis: none was recorded, as G6 was not run.

Measured, in the v1.0 design as written: the G3 and G4 metrics for the full configuration of each
task and for variants that each change one component. The full configuration is the one each task
uses for G3 to G5 (method.md section 11).

Text and regression, full configuration S2:

- no Langevin noise;
- fixed step size instead of a randomized one;
- fixed step count instead of a randomized one;
- no replay buffer;
- S1 mode instead of S2;
- full unroll instead of S2.

Images, full configuration S1, training modes only:

- S2 mode instead of S1;
- full unroll instead of S1.

The component ablations are not run for images. The authors' image S2 configuration uses no
Langevin noise, and their code has no image replay buffer (method.md sections 7 and 9).

The v1.0 design planned 5 seeds per variant. For v1.1 the runs per variant are part of the W5
budget, item 21 of the v1.1 list ([design.md section 4.9](design.md#49-pre-registration)).

Answer: not run in v1.0; G6 moves to W5 ([Status of G6](#status-of-g6)). The planned answer is a
table of each variant's difference from the full configuration, with the spread across seeds. A
difference is reported as an effect only if it exceeds that spread. Otherwise it is reported as no
measurable effect.

### G7. Is the comparison fair at equal cost?

Question: does the EBT do better than a standard model when both are given the same size, and when
both are given the same compute?

Hypothesis: none was recorded in v1.0.

Measured, for each task:

- the task error of the EBT and of a one-pass baseline with the same parameter count;
- the training cost and inference cost of each, counted as forward passes, first-order backward
  passes and second-order backward passes (method.md section 6.3);
- the error against cost for both models, including a baseline given extra compute that matches
  the EBT's cost (a larger model or longer training, stated per task). This part is planned for
  v1.1.

In v1.0 only the part at equal parameters and equal steps was measured. The planned answer of
v1.1 gives plots of error against cost for training and for inference, with the parameter counts
stated. For each task it states whether either model is better at equal parameters and at equal
cost. Wall-clock time is not used as the cost measure.

Answer, at equal parameters and equal steps only (mean over seeds 0–4; passes per example or per
fresh window: forward / first-order backward / second-order):

| Task | EBT | Baseline | EBT training passes | Baseline training passes |
|---|---|---|---|---|
| text, test BPC (lower is better) | 2.332 ± 0.022 | 2.266 ± 0.020 (causal Transformer) | 4.48715 / 4.48715 / 1.49975 | 1 / 1 / 0 |
| MNIST, test PSNR in dB (higher is better) | 27.00 ± 0.03 | 27.54 ± 0.04 (one-pass denoiser) | 1 / 1 / 1 | 1 / 1 / 0 |
| regression, mean r (lower is better) | 0.979 ± 0.001, (b) | 0.973 ± 0.002, (a) | | |

Sources: [`results/text/lm/full/table.md`](../results/text/lm/full/table.md),
[`results/mnist/denoising/full/table.md`](../results/mnist/denoising/full/table.md) and
[`results/regression/g2/full/table.md`](../results/regression/g2/full/table.md). The full text
table is in [method.md section 18.9](method.md#189-results-of-the-final-runs).

- Text. The two models have 6 328 641 and 6 328 064 parameters and train for T = 2 000 steps on
  the same fresh windows in the same order per seed. The baseline has the lower mean BPC on every
  split. The EBT spends more compute per step: its training passes vary per seed from 4.43875 to
  4.55700, and at inference it makes 4 forward and 4 backward passes per window, against 1 forward
  pass. A pass is also larger for the EBT, because it processes the S + 1 observed tokens and the
  S predicted tokens, against the baseline's S tokens (method.md section 18.3).
- MNIST. The two models have 828 033 and 829 968 parameters and train for 10 000 steps each. The
  one-pass denoiser has the higher PSNR. The EBT spends a second-order pass per training example
  and two forward and two backward passes at inference.
- Regression. The models are of the same size ([G2](#g2-why-use-an-energy)). (a) and (b) both
  predict near the mean of the two branches, with an equal test MSE of 2.099 ± 0.000, so neither
  solves the task.

A compute-matched baseline is not part of v1.0 and is planned for v1.1.
