# Design

This document records the decisions that define the project: its main question and scope, the
families of text generation it compares and their representatives, the axes crossed with the
families, the protocol, the metrics and the versions. Each decision is given with its reason and
its references. The references are listed at the end, with their arXiv identifier or URL and their
date. A claim made by the vendor of a closed model is labelled as a vendor claim.

The questions are in [questions.md](questions.md). The implementation of the autoregressive and
EBT families and the v1.0 experiments are in [method.md](method.md).

Everything in this document that has not yet been run is planned, with the version given in
[section 6](#6-versions). The only results of this repository are those of v1.0 ([questions.md
section C](questions.md#c-the-v10-answers)).

## 1. Main question and scope

### 1.1 Main question

Under equal training compute and identical measurement, at small scale, how do families of text
generation compare in quality, cost and return on inference-time compute, and do the central
claims of each family's paper hold?

### 1.2 Why it matters

- The EBT part of this project is an independent small-scale reimplementation of Energy-Based
  Transformers [EBT]. A search on 23 November 2026 of arXiv, the general web, GitHub, Hugging
  Face and OpenReview found no published independent replication of the paper's claims. [Malloy]
  trains its own EBT for a different purpose, choosing the inference depth with instance-based
  learning, and does not replicate the paper's experiments.
- Commercial systems report their own quality figures: Mercury [Mercury] and Gemini Diffusion
  [Gemini Diffusion] are closed models, and their quality and speed figures are vendor-reported;
  some of Mercury's cite third-party evaluations (Artificial Analysis, Copilot Arena).
- At matched scale, masked diffusion language models "still lag behind comparably sized
  autoregressive models" [2601.15593]. On text8, Table 2 of [2412.10193] reports a retrained MDLM
  at ≤ 1.38 bits per character against 1.23 for an autoregressive model quoted from earlier work.
- The comparison is not one-sided: [2410.18514] reports that in text generation, masked diffusion
  models with 16 times more pre-training time match autoregressive models that use a key-value
  cache (abstract), and [Prabhudesai] and
  [Ni] report diffusion ahead of autoregressive models when unique data are limited (question C5).
- On efficiency, [2510.18480] reports that "AR models generally achieve higher throughput, while
  DLMs consistently lag", and that dual cache and parallel decoding "mainly offer gains at small
  batch sizes".

This project examines whether the ranking of the families holds at small scale under one protocol,
with every number traceable to a result file in one repository and rerunnable with its code.

### 1.3 Non-goals

- Conclusions about frontier-scale models. The models are small (v1.0: 6 328 064 and 6 328 641
  parameters on text, method.md section 18.8; v1.1 to v1.3: one size, fixed in the v1.1 list of
  section 4.9; several sizes in v1.4, section 6); a result holds at the
  scale at which it was measured.
- Reinforcement-learning post-training.
- Long context.
- Instruction following.
- Chain-of-thought reasoning. It does not apply to a character-level model.
- External API judges as primary metrics. A primary metric is computed in this repository
  ([section 5](#5-metrics)).

## 2. Families and representatives

A family is defined by its training objective, its procedure for producing a sample, or its way of
spending inference compute: a method that differs from the others in at least one of the three is
a family. A change that leaves all three unchanged and applies to every family, such as how the
layers are connected, is an axis (section 3). Reason: the comparison must vary what makes the
families generate differently and keep everything else the same for all of them. Each family is
represented by one method, chosen below.

A family enters the project only with its full W1–W6 template ([questions.md, level
2](questions.md#level-2-the-template-for-every-family)). The rule applies from v1.1: the two v1.0
families, autoregressive and EBT, complete their templates in v1.1. Reason: v1.0 was designed
before the template existed.

Implementation: each new family is reimplemented in this codebase, as the EBT was, so that all
families share one harness, run on MPS and are measured identically. The official code of each
family is the reference for its W1: its numbers and behaviour are checked against that code. The
exception is the reference family, whose W1 reference is the Transformer++ recipe (section 2.1).

| Family | Representative | Version (planned unless stated) |
|---|---|---|
| Autoregressive Transformer | causal Transformer, the reference | v1.0 (done, equal steps); v1.1 under the protocol |
| Energy-based optimization | EBT [EBT] | v1.0 (done, equal steps); v1.1 under the protocol |
| Masked discrete diffusion | MDLM [MDLM] with RADD simplifications [RADD] and the first-hitting sampler [FHS] | v1.2 |
| Continuous flow | FLM with its distilled FMLM [FLM] | v1.2 |
| Recurrent depth | prelude, recurrent core and coda as in Huginn [Huginn] | v1.2 |
| Uniform-state diffusion | Duo [Duo], [Duo-2] | v1.3 |
| Block diffusion | BD3-LM [BD3-LM] | v1.3 |

### 2.1 Autoregressive Transformer

- Idea: predict each token from the tokens before it; generate left to right.
- Role: the reference family. Every other family is compared with it (section 4.5). Its W1 to W6
  are listed in [questions.md, level 2](questions.md#level-2-the-template-for-every-family); it
  has no W2, because no claim of its own is tested.
- Reference recipe for W1: the Transformer++ that the EBT paper uses as its baseline. The paper
  compares with "the standard Transformer++ recipe" ([EBT] section 4.1, citing [Mamba], [Llama 2]
  and [TTT]) and uses "the Llama 2 transformer implementation" for it ([EBT] Appendix D). Its
  components: pre-normalization with RMSNorm, the SwiGLU activation and rotary position embeddings
  ([Llama 2] section 2.2); AdamW with β = (0.9, 0.999), weight decay 0.01, gradient clip 1, a
  linear warm-up of 1e4 steps then cosine decay to a tenth of the learning rate, context length
  256 and the GPT-NeoX-20B tokenizer ([EBT] Table D.2); no feed-forward multiplier ([EBT]
  Appendix D.1.1); and the learning rate of the Mamba paper ([EBT] Appendix D).
- Differences of the v1.0 Transformer from that recipe (method.md sections 13, 18.3 and 18.4):
  - characters, with a vocabulary of 65, instead of the GPT-NeoX-20B tokenizer;
  - context length 128 instead of 256 (method.md section 18.8);
  - SwiGLU hidden width 4D, the default of the EBT authors' code, where the paper states no
    feed-forward multiplier (D11);
  - warm-up over 10% of the T training steps (200 steps) instead of 1e4 steps (D13);
  - learning rate chosen by a search over 4 values (method.md section 18.5) instead of the Mamba
    paper's;
  - strict float32 matrix multiplication, where the EBT authors' text scripts set "medium" (D14);
  - model size 256 wide and 6 layers deep, where the authors' thinking experiments use the xxs size,
    384 wide and 6 layers deep ([EBT] Appendix D.1.2 and the authors' S2 script; method.md D5 and
    section 18.8);
  - training length T = 2 000 steps, where the authors' text code schedules 1 000 000 steps
    (method.md D13).

  The batch size, 32, matches.
- Code: the causal Transformer of v1.0 in this repository (MIT), on the backbone of the EBT
  authors' text models (method.md section 18.3). Added in v1.1 (planned): best-of-N with selection
  by the model's own likelihood (question C2) and a fill-in-the-middle variant [FIM] (question C8).

### 2.2 Energy-based optimization: EBT

- Idea: a Transformer assigns a scalar energy to a context and a candidate prediction; a
  prediction is made by gradient descent on the energy from an initial candidate [EBT].
- Reference: [EBT]. Official code: <https://github.com/alexiglad/EBT> (Apache-2.0).
- Why chosen: no published independent replication was found in a search on 23 November 2026
  (section 1.2).
- Taken as is or changed: implemented in this repository from the paper and the official code,
  with no code copied from it. Where the paper and the code differ, the implementation follows the
  code; every difference and correction is listed in method.md sections 13 and 14.

### 2.3 Masked discrete diffusion: MDLM

- Idea: tokens are progressively replaced by a mask token; the model learns to predict the
  original tokens from the partly masked sequence. The training objective is a weighted
  cross-entropy that bounds the likelihood.
- Reference: [MDLM]. Official code: <https://github.com/kuleshov-group/mdlm> (Apache-2.0).
- Why chosen: one implementation covers the shared objective of MD4 [MD4], RADD [RADD] and LLaDA
  [LLaDA]. A published text8 reference exists: Table 2 of [2412.10193] reports a retrained MDLM at
  ≤ 1.38 bits per character, against 1.23 for an autoregressive model quoted from earlier work.
- Taken as is or changed: reimplemented in this codebase; the MDLM objective, with the
  simplifications of RADD (no time conditioning) [RADD] and the first-hitting sampler [FHS]. The
  official code is the reference for W1.

### 2.4 Continuous flow: FLM and FMLM

- Idea: text is generated by a flow in a continuous space; a distilled model, FMLM, generates in
  few steps.
- Reference: [FLM]. Official code: <https://github.com/david3684/flm> (Apache-2.0).
- Why chosen: it represents the continuous-flow family.
- Taken as is or changed: reimplemented in this codebase; FLM, and FMLM for 1, 2 and 4 steps. The
  official code is the reference for W1.
- FMLM: in v1.2 it is tested inside FLM's own template, as a claim of its paper (W2, W3), and is not
  part of the main C2 comparison, which compares base models. In v1.3 C2 adds a separate comparison
  of distilled few-step models, FMLM against Duo with its own distillation (section 3).
- Known gap: the paper reports no likelihood; measuring it would require auxiliary training [FLM].
  Its primary comparison is therefore on the generation-quality metrics of section 5.

### 2.5 Recurrent depth

- Idea: a prelude embeds the input, a recurrent core is applied a variable number of times, and a
  coda decodes. More iterations spend more compute at inference without more parameters.
- Why a family: it spends inference compute by iterating a block, one of the three properties
  that define a family (section 2).
- Reference: Huginn [Huginn] (NeurIPS 2025). Official code:
  <https://github.com/seal-rg/recurrent-pretraining> (Apache-2.0).
- Why chosen: it spends inference-time compute in a latent state rather than in tokens.
- Taken as is or changed: reimplemented in this codebase; the official code is the reference for
  W1.
- Control models: a non-looped model with the same parameters and a non-looped model with the
  same FLOPs. Both kinds of comparison are made in [Looped]; [2604.21106] makes the equal-FLOPs
  (iso-depth) comparison. The two control models are part of the family's W5 and are trained from
  its ablation budget. Reason: every family has the same budget for its own questions (principle
  6), and the controls test a component of this family, the recurrence.
- The paper reports a training run whose model learned to ignore its recurrent state, with the
  same validation perplexity for 1 and 32 recurrences [Huginn]; question C2 checks for this in
  every family.

### 2.6 v1.3: Duo and BD3-LM

- Duo: uniform-state diffusion, in which tokens are corrupted towards uniformly random tokens
  instead of a mask [Duo], [Duo-2]. Official code: <https://github.com/s-sahoo/duo>. Its own
  distillation, Discrete Consistency Distillation (DCD, from the Duo paper [Duo]), gives the
  few-step model of the separate C2 comparison against FMLM (section 3).
- BD3-LM: block diffusion, which generates blocks of tokens autoregressively and the tokens within
  a block by diffusion, between the autoregressive and diffusion families [BD3-LM].
- Both are reimplemented in this codebase; the official code of each is the reference for its W1.

### 2.7 Rejected options

| Option | Reason |
|---|---|
| SEDD [SEDD] | Its absorbing case is covered by MDLM: for absorbing discrete diffusion, [RADD] shows that the concrete score is a time-independent conditional distribution of clean data times an analytic time-dependent term (Section 3.1, Theorem 1) and that the objective equals the any-order autoregressive objective (Section 3.3, Theorem 2); its Section 5 notes that [MDLM] derives the same cross-entropy losses as [MD4]. |
| D3PM [D3PM] | Its absorbing-state process is the one that MDLM trains with a continuous-time objective [MDLM], so the MDLM implementation covers it. |
| MD4 [MD4] and LLaDA [LLaDA] as separate entries | Same objective as MDLM; LLaDA is a scale reference. |
| BFN [BFN] | A second continuous method next to FLM; one method represents each family, and FLM was chosen because its distilled model FMLM reaches 1 to 4 steps [FLM]. BFN's character-level result is kept as a reference point: text8, ≤ 1.41 bits per character [BFN]. |
| Diffusion-LM [Diffusion-LM] | Its aim is controllable generation: it denoises Gaussian vectors into word vectors so that a gradient-based algorithm can steer the output [Diffusion-LM]; control is not a question of this project, and the continuous family is represented by FLM. |
| CDCD [CDCD] | A continuous diffusion framework, "continuous both in time and input space" [CDCD], of the same family as FLM, which represents it. |
| SSD-LM [SSD-LM] | It generates blocks of text semi-autoregressively by simplex diffusion over the vocabulary [SSD-LM]; the family that interpolates between autoregressive and diffusion generation is represented by BD3-LM [BD3-LM]. |
| Plaid [Plaid] | A likelihood-based continuous diffusion model whose stated aim is to take first steps towards closing the likelihood gap to autoregressive models; Plaid 1B outperforms GPT-2 124M in likelihood [Plaid]. That scale is outside this project, and the continuous family is represented by FLM. |
| HRM [HRM] and TRM [TRM] | Puzzle solvers, not next-token language models. An autoregressive TRM did worse than a dense Transformer on character tasks [TinyARM]: 12%, 11% and 10% character accuracy on addition, copy and reverse against 80%, 100% and 100%. An independent analysis attributes HRM's ARC-AGI results mainly to its outer refinement loop and data augmentation rather than to its hierarchical architecture [ARC HRM]. |
| Coconut [Coconut] | It needs chain-of-thought supervision, which the data lack: its training curriculum replaces steps of language chain-of-thought data with continuous thoughts. |
| Mercury [Mercury] and Gemini Diffusion [Gemini Diffusion] | Closed; context only (section 1.2). |
| FS-DFM [FS-DFM] | A step-aware fine-tune of a pretrained discrete flow model, from a separate paper; only a distillation from a family's own paper is included (section 3). |
| SDTT [SDTT] for MDLM | A distillation from a separate paper, not from MDLM's own. Only a distillation from the family's own paper is included (section 3), so that each family is tested with what its own paper proposes. |
| State-space and other backbones (Mamba [Mamba], RWKV [RWKV-7]) | Backbone replacements: each changes every family at once, so the families would not share a backbone and the comparison would mix family and backbone. |

## 3. Axes

Three axes are crossed with the families.

| Axis | Values | Question |
|---|---|---|
| Inference-time compute | the shared knob grid 1, 2, 4, …, 64; the x-axis is inference FLOPs per generated character, onto which each family's knob is mapped; diffusion's point at the sequence length is reported as an extra point of that family | C2 |
| Backbone | standard residual connections, DeepCrossAttention [DCA] | C9 |
| Unique data | about 1M, 10M and 90M characters, all subsets of text8 | C5; related work [Muennighoff], [Prabhudesai], [Ni] |

Inference-time compute. Knob values are not comparable across families: one EBT step, one
diffusion step and one recurrent iteration cost different amounts of compute. The x-axis of C2 is
therefore inference FLOPs per generated character, counted by the FLOP rule of the version's list
(section 4.9), and each family's knob values are placed on it. The knob grid 1 to 64 is shared;
the sequence length is a natural end point for diffusion only, so that point is reported for
diffusion as an extra point and is not part of the shared grid.

Few-step generation. A distilled few-step model is a second training stage on a base model (FMLM
[FLM], DCD [Duo], SDTT [SDTT], FS-DFM [FS-DFM]); it is not a family, and its steps lie on the
inference-compute axis. The main C2 comparison is between base models. A distilled model is
included only if the distillation comes from the family's own paper: FMLM inside FLM's template
in v1.2, and in v1.3 a separate C2 comparison of FMLM against Duo-DCD. Reason: a distilled model
compared with undistilled ones at 1 to 4 steps would confound distillation with family; and a
distillation from another paper would test that paper, not the family's.

Backbone. DeepCrossAttention changes none of the three properties that define a family (section 2),
only how the layers are connected, and it applies to every family; placing it in one family only
would confound every comparison. It is therefore an axis (question C9) [DCA], [DCA blog]. It is the
only backbone change tested: state-space and recurrent backbones such as Mamba [Mamba] and RWKV
[RWKV-7] replace the Transformer and change every family at once (section 2.7), while
DeepCrossAttention keeps the Transformer and has a reported gain at the same depth as here, 6
layers: on LM1B the perplexity falls from 18.98 ± 0.01 for the Transformer to 18.06 ± 0.01 for DCA
([DCA] Table 5, at about 50M parameters and 131B training tokens, [DCA] section 5). As that gain
was measured with far more data than this project uses, a pilot with 5 seeds comes first, as for
every comparison (section 4.5). Its decision rule is in the v1.2 list (section 4.9).

Unique data. All three points are subsets of text8, so that between the points only the amount of
unique text changes, not its source or its vocabulary. Tiny Shakespeare is not a point on this
axis: it remains the development set and the corpus of the v1.0 regime (section 4.2).

## 4. Protocol

The protocol applies one rule to every family: no family is favoured ([questions.md principle
8](questions.md#a-principles)). Every family gets the same budget, the same protocol and the same
rules, and no family's results are reported with more prominence than another's.

### 4.1 Conditions

- Primary condition: equal parameters (within ±5%, with the non-embedding parameters stated) and
  equal training FLOPs at fixed compute marks. FLOPs are counted per family, by the rule of item 4
  of the v1.1 list (analytic or measured, section 4.9), once per model.
- Secondary views: equal tokens, and equal wall-clock time on the same machine. They are reported;
  no decision is taken on them.
- Reason: a step costs a different amount of compute in each family. For S1 pretraining with two
  optimization steps, EBTs used 6.66 times the training FLOPs of a comparable Transformer++ [EBT,
  Appendix D.5]. Equal steps, the condition of v1.0, is therefore not a fair condition across
  families ([questions.md principle 1](questions.md#a-principles)).
- The compute-optimal frontier (several sizes and several budgets per family) is v1.4, following the
  scaling-law method for masked diffusion models of [2410.18514].
- Scope of the answers: from v1.1 to v1.3 every family is trained at one model size, fixed in the
  v1.1 list (section 4.9), so the answers of those versions hold at that size only (question C1).
  C10 begins in v1.4 with the compute-optimal frontier.

### 4.2 Data

| Dataset | Role | Reason |
|---|---|---|
| text8 [text8] | main dataset | 100M characters, 27 symbols (a–z and space); published numbers exist for the autoregressive family, MDLM, MD4 and BFN ([2412.10193], [MD4], [BFN]). |
| Tiny Shakespeare | development set; the corpus of the v1.0 regime | the v1.0 corpus (method.md section 18.1); about 1M characters. It is not a point of the unique-data axis, whose points are subsets of text8 (section 3). |
| TinyStories [TinyStories] (CDLA-Sharing-1.0) | main generation-quality dataset, v1.4 | an independent controlled comparison of autoregressive and MDLM models exists on it [2603.22075]. It is used at the character level, like every other dataset, so that compute is counted per character, and with no external model as a judge in a primary metric. |

Out-of-distribution text keeps a fixed rule for mapping its characters onto the training
vocabulary, decided before training.

### 4.3 Tuning

- Equal for all families: the total number of trials, the number of seeds per trial, and the
  compute mark and data size at which the trials are scored. These are fixed in the v1.1 list
  (section 4.9). Reason: equal budget (principle 6); the seeds and the scoring point are part of
  the compute spent on tuning.
- Chosen by each family, declared before its search: which hyperparameters its trials cover,
  centred on the family's reference values (for the EBT including the step size α), and the grid
  spacing. Settings not covered are fixed from the family's paper. Reason: families differ in which
  hyperparameters matter (on text, v1.0 searched 4 learning rates for the baseline, and for the EBT
  3 step sizes at learning rate 1.2e-3 and 1 configuration at 3e-3; method.md section 18.5,
  `results/text/lm/full/search.json`), and declaring the space beforehand keeps the search from
  being shaped by its results.
- Edge rule, recorded before the search: if the best value is at the edge of the grid, one value
  beyond it is tried. That trial is taken from a reserve inside the same total, not added to it;
  trials not used stay unused, and the trials actually spent are reported. Reason: an extension
  added to the total would give a family whose optimum lies at an edge more trials than the
  others.
- The expected best score against the number of trials is reported [Show Your Work].

### 4.4 Training length

- Schedule: warm-up, then a constant learning rate, with a cool-down branch at each compute mark
  [Hägele].
- Only cooled-down checkpoints are compared [No Train No Gain].
- A family still improving by more than a fixed δ at the final mark is reported as not converged.
  δ is fixed in the v1.1 list (section 4.9).

### 4.5 Statistics

- 5 seeds per model, paired by seed.
- For each comparison: the mean and the 95% interval of the paired difference.
- A fixed comparison set per version: each family against the autoregressive reference, plus the
  pairs listed before the version's first measurement. Reason: an all-against-all set grows with
  every family added, which changes the correction applied to earlier comparisons.
- Holm correction [Holm], with the family of tests defined per version and per question.
- Intervals reported with Holm-adjusted tests are Bonferroni-adjusted intervals: two-sided paired-t
  intervals at level 1 − 0.05 / m for a family of m tests, from the same paired t statistic as the
  test. Reason: Bonferroni intervals have a simple form and are conservative, and they agree with
  the Holm test in one direction: Holm rejects at least what
  Bonferroni rejects, so an interval that excludes 0 belongs to a difference that Holm rejects; the
  reverse need not hold.
- The answers of an earlier version stay as they were, under that version's own comparison set
  (principle 5).
- Diffusion likelihoods are upper bounds and are always written "≤" [SEDD].

### 4.6 Reference points

- A character n-gram model, its order chosen on validation data, as the floor.
- The published text8 numbers as the external check (section 4.2).

### 4.7 Effective batch size

- All families train with one common effective batch size, fixed in the v1.1 list (section 4.9):
  the number of fresh training windows per optimizer step, the same for every family. The EBT's
  replayed rows are part of its method and are counted in its FLOPs, not in the batch. Reason:
  every family sees the same amount of data per step, and extra compute is charged through the
  FLOP count.
- Gradient accumulation exists to reach that batch size within the 16 GB of memory of the machine.
- Reason for one batch size: the reference papers use different batch sizes. At equal training
  compute, a batch size per reference paper would give each family a different number of
  optimizer steps, so the comparison would mix the family with its number of updates.

### 4.8 Context length

- One sequence length for training and evaluation, the same for all families. The likelihood
  metric is computed on the same non-overlapping blocks of that length for every family; a sliding
  window is not used in the primary metric.
- Reason: bits per character depend on the context each character sees. An autoregressive model
  evaluated with a sliding window sees almost the full context for every character, a diffusion
  model only its block; equal blocks give every family the same context.
- The value is fixed in the v1.1 list (section 4.9).

### 4.9 Pre-registration

Each version has its own numbered list of the settings that are fixed before its first
measurement, with a status: open, or fixed with its date. A version's first measurement starts
only when every item of its list is fixed. Every later version has its own list, fixed before its
first measurement and covering its new families: for each, at least its FLOP rule, its declared
trial space, its C6 inference optimizations, its temperature-setting method for the entropy
target, its C3 confidence score and how better candidates are judged, and its W1 check against the
official code. Reason: principle 3 of questions.md; a setting
fixed before any measurement cannot be shaped by the results.

v1.1 (autoregressive and EBT under the protocol):

1. For each of C1 to C8: its hypothesis, primary metric, decision rule, minimum meaningful effect
   (derived from the seed-to-seed spread) and Holm family. C8 is fixed here, before its first
   measurement in v1.2. Status: open.
2. The single model size used in v1.1 to v1.3 (section 4.1). Status: open.
3. The context length (section 4.8). Status: open.
4. The FLOP counting rule per family: analytic or measured; for the EBT, the second-order term,
   the two input streams and the replay rows; random step and iteration counts; whether cool-down
   branches count. Status: open.
5. The compute marks; the number of passes over each unique-data subset follows from them (C5).
   Status: open.
6. The common effective batch size (section 4.7). Status: open.
7. Tuning: the total number of trials and the reserve for the edge rule, the seeds per trial, the
   compute mark and data size at which trials are scored, and each family's declared trial space
   (section 4.3). Status: open.
8. The exact definition of each metric (section 5), including how better candidates are judged
   for C3. Status: open.
9. The primary metric of C1 and its decision rule, given exact likelihoods, upper bounds and
   FLM without a likelihood. Status: open.
10. The inference step at which the EBT's likelihood enters C1. Status: open.
11. The entropy target of C2 (the entropy of held-out text) and how each family's sampling
    temperature is set to reach it at every knob value. Status: open.
12. The sampling and generation budget: number and length of samples per model. Status: open.
13. The confidence score each family uses for C3. Status: open.
14. The out-of-distribution text used with text8 and its character mapping (section 4.2).
    Status: open.
15. How the C5 subsets of text8 are cut. Status: open.
16. C6: the common quality level for the second latency point, the standard inference
    optimizations of each family (for the autoregressive model the key-value cache), and the
    conditions under which latency and peak memory are measured. Status: open.
17. C7: the common learning-rate grid (1/4, 1/2, 1, 2 and 4 times the best learning rate) and its
    seeds. Status: open.
18. The δ for "not converged" (section 4.4). Status: open.
19. The comparison set of v1.1 (section 4.5). Status: open.
20. How each family's W1 check against its reference runs on this hardware. Status: open.
21. The W5 budget, the same for every family: the number of ablated components, the runs per
    component, and the compute mark and data size of those runs. Status: open.
22. The compute mark and data size of the C7 runs. Status: open.
23. How energy per character is measured and logged, or that it is not measured in v1.1. Status:
    open.
24. The fill-in-the-middle variant for C8: trained at the same compute marks as the other C8
    models, with the tuned settings of the autoregressive model and no separate search. Reason: C8
    tests infilling, not tuning, and the other families infill without a separate model. Status:
    open (its compute marks are those of item 5).
25. The question raised by the descriptive G5 observations of v1.0 ([questions.md,
    G5](questions.md#g5-does-thinking-help-more-out-of-distribution)): the effect of thinking at
    small step counts (1, 2 and 4 steps, including the trained step count) and its dependence on
    the distance from the training distribution. Its hypothesis, metric, decision rule, minimum
    meaningful effect and Holm family are fixed with the other items of this list, before the
    first v1.1 measurement. Status: open.
26. A worked example of one thinking step whose numbers can be checked by hand (questions.md,
    [G1](questions.md#g1-is-the-implementation-faithful-to-the-method)): the energy, the inputs, the
    step size and where the example is published. Status: open.
27. Whether the text EBT is also evaluated with the training Langevin noise (σ = 3, the authors'
    text setting; method.md D12) for comparison with the evaluation without noise, on which
    checkpoints, and how it is reported. Status: open.

v1.2 (MDLM, FLM with FMLM, recurrent depth; C8; the DeepCrossAttention pilot):

1. For each new family: its FLOP rule (for the diffusion families masked-only supervision; for
   FMLM its distillation and teacher passes and how they count against FLM's budget; for recurrent
   depth the iteration counts), its declared trial space, its C6 inference optimizations, its
   method for setting the temperature to the entropy target of C2, its C3 confidence score and how
   better candidates are judged, and its W1 check. Status: open.
2. C9, the DeepCrossAttention pilot: its hypothesis, primary metric, minimum meaningful effect,
   Holm family and the families it runs on, and its decision rule: how its results are combined
   across families, the threshold for running the full crossed grid, and where the decision is
   recorded. Status: open.
3. The comparison set of v1.2. Status: open.

v1.3 (Duo with DCD, BD3-LM): the list follows the same rule, with the separate C2 comparison of
FMLM against Duo-DCD. Status: open.

v1.4 (larger models, the compute-optimal frontier, TinyStories at the character level): the list
follows the same rule, with the hypothesis and decision rule of C10. Status: open.

## 5. Metrics

| Metric | Definition | Families | Logged | Pitfalls |
|---|---|---|---|---|
| Likelihood | bits per character on held-out text, on the non-overlapping blocks of section 4.8; exact for the autoregressive family, an upper bound "≤" for diffusion; for MDLM also the exact left-to-right likelihood | all that define one; FLM reports none (section 2.4) | per-chunk validation losses; test and out-of-distribution values per checkpoint | bounds and exact values are not the same quantity [SEDD] |
| Generation quality | primary metric of C2: a profile at matched entropy (the entropy of held-out text, section 4.9 item 11): character n-gram divergence to held-out text, a compression-based divergence, MAUVE [MAUVE] (supplementary) | all | raw samples with the full sampler settings | quality and entropy trade off, so samples are compared at matched entropy; 32-bit floating-point categorical sampling is inaccurate and distorts masked-diffusion samples [FHS] |
| Self-scoring | C3: the error of the candidate chosen by the family's own score against random and oracle choice among the same candidates; how better candidates are judged is fixed per family in the version's list (section 4.9) | all | every candidate with its score | a score that tracks length or entropy rather than quality |
| Diversity (supplementary) | distinct-n [distinct-n], self-BLEU [Texygen], duplication | all | raw samples | depends on the number and length of samples |
| Memorization (supplementary) | share of spans of 20 or more characters copied from the training data, reported against a baseline: the same measure applied to held-out text | all | raw samples; training data index | depends on the amount of unique data (C5); held-out text shares spans with the training data by chance |
| Training FLOPs, inference FLOPs per character | counted per family, once per model, analytically or by measurement as item 4 of the v1.1 list fixes (sections 4.1 and 4.9) | all | cumulative FLOPs and tokens | counts per family must cover every pass, including iterations and candidates |
| Latency | primary metric of C6: time to first character and throughput, by batch size, at knob value 1 and at the smallest knob value at which the family reaches the common quality level, with each family's own declared inference optimizations | all | wall-clock time per call, with the conditions: machine, device, batch size, sequence length, precision, library versions, free memory, no other process running | machine-dependent: measured on the stated machine with the conditions recorded; the latency of a diffusion model at 1 step and at L steps differs by orders of magnitude, and optimizations change the ranking [2510.18480] |
| Peak memory (supplementary) | peak memory of training and inference | all | peak memory, with the same conditions as latency | machine-dependent; reported with its conditions; no decision on it |
| Test-time scaling curve | quality against inference FLOPs per generated character, with each family's knob mapped onto it (section 3) | all | per-step values and their FLOPs | extra iterations may not change the output at all [Huginn] |
| Out-of-distribution ratio | quality on out-of-distribution text relative to in-distribution text | all | values on both splits | the character mapping is fixed before training (section 4.2) |
| Training stability, hyperparameter sensitivity | C7: the loss of the primary metric across the common learning-rate grid (1/4, 1/2, 1, 2 and 4 times the best learning rate, same seeds) and the fraction of diverged runs | all | per-chunk validation losses of every run | the grid is relative to each family's tuned optimum |
| Calibration (supplementary) | calibration of next-character probabilities | families with next-character probabilities | per-position probabilities | — |
| Infilling quality | quality of a filled missing middle span | families that can infill, from v1.2; the autoregressive family with a fill-in-the-middle variant [FIM]; not the causal EBT | raw infills with settings | — |
| Energy per character (supplementary) | energy used per generated character | all | — | no decision on it |

A metric with no question is supplementary: calibration, diversity and memorization. Peak memory,
energy per character and MAUVE are also supplementary. A supplementary metric is reported with its
conditions and no decision is taken on it. Reason: a decision needs a question with a hypothesis
and a decision rule fixed beforehand (principle 3). The exact definition of each metric, the
minimum meaningful effects, the δ for "not converged" and the compute marks are fixed in the v1.1
list (section 4.9).

The networks run on MPS. Only the categorical draw runs in float64 on the CPU, because 32-bit
floating-point precision makes categorical sampling inaccurate [FHS] and MPS has no float64.

### 5.1 What must be logged during training

- the configuration and the code fingerprint;
- all seeds;
- parameter counts (total and non-embedding);
- cumulative FLOPs and tokens;
- wall-clock time;
- peak memory;
- per-chunk validation losses;
- checkpoints at the compute marks, before and after the cool-down;
- family internals: EBT energies and steps; diffusion time samples and the terms of the ELBO;
  recurrent iteration counts;
- raw samples with the full sampler settings.

## 6. Versions

| Version | What it answers |
|---|---|
| v1.0 | the autoregressive and EBT families on Tiny Shakespeare at equal parameters and equal steps (G7 at equal parameters and equal steps only; its part at equal cost and its plots are planned for v1.1); the EBT study on three tasks; G5 measured; G6 not run, moved to W5 ([questions.md section C](questions.md#c-the-v10-answers)) |
| v1.1 (planned) | protocol infrastructure: FLOP accounting, logging, the cool-down schedule, text8, the metric suite, categorical draws in float64 on the CPU, the common effective batch size with gradient accumulation to reach it within 16 GB (section 4.7), the n-gram reference. The autoregressive and EBT families under the protocol, C1 to C7, each completing its W1–W6 template; best-of-N for the autoregressive family; the fill-in-the-middle variant. |
| v1.2 (planned) | MDLM, FLM (with FMLM inside FLM's template) and recurrent depth, each with its full W1–W6; C8; the DeepCrossAttention pilot (C9) |
| v1.3 (planned) | Duo and BD3-LM, each with its full W1–W6; the separate C2 comparison of distilled few-step models, FMLM against Duo-DCD |
| v1.4 (planned) | larger models on a Mac with 64 GB or more; the compute-optimal frontier (C10); TinyStories at the character level |

Each version has its own pre-registration list (section 4.9).

### 6.1 From v1.0 to v1.1

v1.1 changes the conditions of the answers. The v1.0 answers stay as they are, tied to the v1.0
conditions; the v1.1 answers are added next to them ([questions.md principle
5](questions.md#a-principles)).

| | v1.0 | v1.1 (planned) | Reason |
|---|---|---|---|
| Condition | equal parameters and equal training steps (T = 2 000 on text) | equal parameters and equal training FLOPs at fixed compute marks | a step costs different compute per family (section 4.1) |
| Main data | Tiny Shakespeare, about 8.2 passes over 1 003 854 training characters | text8, with the unique-data points as subsets of it; Tiny Shakespeare for development | v1.0 is in a regime of many passes over a small corpus; text8 has published reference numbers (section 4.2) |
| Checkpoint compared | the best checkpoint by validation BPC within T steps | cooled-down checkpoints at each compute mark; "not converged" reported | the v1.0 EBT had its best checkpoint at the last step on two of five seeds (questions.md section C); [Hägele], [No Train No Gain] |
| Tuning | per family as in method.md section 18.5 (text: baseline 4 learning rates; EBT 3 step sizes at learning rate 1.2e-3 and 1 configuration at 3e-3, `results/text/lm/full/search.json`) | the same total number of trials, seeds per trial and scoring point for every family, over a trial space each family declares beforehand; the edge rule from a reserve inside the total | equal budget per family (principle 6; section 4.3) |
| Context length | 128 characters on text, evaluated on non-overlapping windows of that length (method.md section 18.2) | one length for all families, likelihood on the same non-overlapping blocks | section 4.8 |
| Statistics | mean over 5 seeds ± half-width of the 95% t-interval per model | paired differences by seed; a fixed comparison set per version; Holm correction per version and question, with Bonferroni-adjusted intervals | the comparisons are between families on the same seeds (section 4.5) |
| Sampling | text generation already drew in float64 on the CPU: the logits are moved to the CPU and converted to float64 before the softmax and the draw (method.md section 18.5) | the same draw in float64 on the CPU for every family and for every sample that enters a metric, with the networks on MPS | [FHS] |
| Effective batch size | per task (text: 32 windows of 128 characters) | one common effective batch size for all families, with gradient accumulation | section 4.7 |
| Reference points | none | character n-gram floor; published text8 numbers | section 4.6 |
| Inference compute of the autoregressive family | one pass | best-of-N with selection by its own likelihood, at the common entropy target | every family gets a knob for question C2 |
| Model size | one per task | one size for all families in v1.1 to v1.3, fixed in the v1.1 list (section 4.1); several sizes in v1.4 | section 4.1 |

## References

Dates are those of the first arXiv version unless stated otherwise.

- [2410.18514] S. Nie et al. Scaling up Masked Diffusion Models on Text. arXiv:2410.18514,
  2024-10-24. <https://arxiv.org/abs/2410.18514>
- [2412.10193] Y. Schiff et al. Simple Guidance Mechanisms for Discrete Diffusion Models.
  arXiv:2412.10193, 2024-12-13; ICLR 2025. Table 2 (text8): autoregressive 1.23 (quoted from
  earlier work), MDLM retrained ≤ 1.38 bits per character. <https://arxiv.org/abs/2412.10193>
- [2510.18480] Peng et al. How Efficient Are Diffusion Language Models? A Critical Examination of
  Efficiency Evaluation Practices. arXiv:2510.18480, 2025-10-22.
  <https://arxiv.org/abs/2510.18480>
- [2601.15593] Zhong et al. Parallelism and Generation Order in Masked Diffusion Language Models:
  Limits Today, Potential Tomorrow. arXiv:2601.15593, 2026-01-22.
  <https://arxiv.org/abs/2601.15593>
- [2603.22075] Vicentino. Autoregressive vs. Masked Diffusion Language Models: A Controlled
  Comparison. arXiv:2603.22075, 2026-03-23. <https://arxiv.org/abs/2603.22075>
- [2604.21106] Schwethelm et al. How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for
  Looped Language Models. arXiv:2604.21106, April 2026. <https://arxiv.org/abs/2604.21106>
- [ARC HRM] ARC Prize Team. The Hidden Drivers of HRM's Performance on ARC-AGI. 2025-08-15.
  <https://arcprize.org/blog/hrm-analysis>
- [BD3-LM] M. Arriola et al. Block Diffusion: Interpolating Between Autoregressive and Diffusion
  Language Models. arXiv:2503.09573, 2025-03-12; ICLR 2025. <https://arxiv.org/abs/2503.09573>
- [BFN] A. Graves et al. Bayesian Flow Networks. arXiv:2308.07037, 2023-08-14. Table 4: text8,
  1.41 bits per character, an upper bound. <https://arxiv.org/abs/2308.07037>
- [Bishop] C. M. Bishop. Mixture Density Networks. Neural Computing Research Group report
  NCRG/94/004, Aston University, 1994. Model (a2) of G2 (method.md section 16.8).
- [CDCD] S. Dieleman et al. Continuous diffusion for categorical data. arXiv:2211.15089,
  2022-11-28. <https://arxiv.org/abs/2211.15089>
- [Coconut] S. Hao et al. Training Large Language Models to Reason in a Continuous Latent Space.
  arXiv:2412.06769, 2024-12-09; ICLR 2025 Workshop on Reasoning and Planning for LLMs. Section 3:
  "We leverage language CoT data to supervise continuous thought".
  <https://arxiv.org/abs/2412.06769>
- [D3PM] J. Austin et al. Structured Denoising Diffusion Models in Discrete State-Spaces.
  arXiv:2107.03006, 2021-07-07. <https://arxiv.org/abs/2107.03006>
- [Dagréou] M. Dagréou, P. Ablin, S. Vaiter, T. Moreau. How to compute Hessian-vector products?
  ICLR Blogposts 2024; authors in the order of the ICLR 2024 listing
  (<https://iclr.cc/virtual/2024/20692>). Cited by [EBT] Appendix D.5 for the cost of a
  second-order step (method.md section 6.3). <https://iclr-blogposts.github.io/2024/blog/bench-hvp/>
- [DCA] M. Heddes et al. DeepCrossAttention: Supercharging Transformer Residual Connections.
  arXiv:2502.06785, 2025-02-10; cited in its current version, v2, ICML 2025, PMLR volume 267.
  Table 5 (6-layer model on LM1B, comparison with related work): perplexity 18.98 ± 0.01 for the
  Transformer and 18.06 ± 0.01 for DCA. arXiv v1 printed 18.961 and 17.998 in the same table.
  Section 5: 500k steps, 131B tokens. <https://proceedings.mlr.press/v267/heddes25a.html>,
  <https://arxiv.org/abs/2502.06785>
- [DCA blog] K. Axiotis, T. Fu. Supercharging transformer residual connections. Google Research
  blog; states that the paper appeared at ICML 2025. Accessed 2026-11-23.
  <https://research.google/blog/supercharging-transformer-residual-connections/>
- [Diffusion-LM] X. L. Li et al. Diffusion-LM Improves Controllable Text Generation.
  arXiv:2205.14217, 2022-05-27. <https://arxiv.org/abs/2205.14217>
- [distinct-n] J. Li et al. A Diversity-Promoting Objective Function for Neural Conversation
  Models. arXiv:1510.03055, 2015-10-11; NAACL 2016. <https://arxiv.org/abs/1510.03055>
- [Du-Mordatch] Y. Du, I. Mordatch. Implicit Generation and Generalization in Energy-Based Models.
  arXiv:1903.08689, 2019-03-20 (<https://arxiv.org/abs/1903.08689>). Published at NeurIPS 2019 as
  "Implicit Generation and Modeling with Energy Based Models"
  (<https://proceedings.neurips.cc/paper_files/paper/2019/hash/378a063b8fdb1db941e34f4bde584c7d-Abstract.html>).
  Section 3.3 and Algorithm 1: the energy-magnitude penalty of model (c3) of G2 (method.md section
  16.4).
- [Duo] S. S. Sahoo et al. The Diffusion Duality. arXiv:2506.10892, 2025-06-12; ICML 2025.
  Introduces Discrete Consistency Distillation (DCD). <https://arxiv.org/abs/2506.10892> Code:
  <https://github.com/s-sahoo/duo> (Apache-2.0).
- [Duo-2] J. Deschenaux et al. The Diffusion Duality, Chapter II: Ψ-Samplers. arXiv:2602.21185,
  2026-02-24. <https://arxiv.org/abs/2602.21185>
- [EBT] A. Gladstone et al. Energy-Based Transformers are Scalable Learners and Thinkers.
  arXiv:2507.02092, 2025-07-02. Section 4.1: the baseline is "the standard Transformer++ recipe",
  citing [Mamba], [Llama 2] and [TTT]; Appendix D: "the Llama 2 transformer implementation" for
  it, with its settings in Table D.2. Appendix D.5, for S1 pretraining with two optimization steps:
  "EBTs used 6.66× the FLOPs of a comparable Transformer++ during training". Derived here, not
  stated in the paper: about 3.33 times per step (6.66 / 2, assuming cost linear in the number of
  steps). ICLR 2026 oral: <https://iclr.cc/virtual/2026/oral/10008833>.
  <https://arxiv.org/abs/2507.02092>. Code: <https://github.com/alexiglad/EBT> (Apache-2.0).
- [FHS] K. Zheng et al. Masked Diffusion Models are Secretly Time-Agnostic Masked Models and
  Exploit Inaccurate Categorical Sampling. arXiv:2409.02908, 2024-09-04; ICLR 2025. Introduces the
  first-hitting sampler and shows that 32-bit floating-point precision makes categorical sampling
  inaccurate. <https://arxiv.org/abs/2409.02908>
- [FIM] M. Bavarian et al. Efficient Training of Language Models to Fill in the Middle.
  arXiv:2207.14255, 2022-07-28. <https://arxiv.org/abs/2207.14255>
- [FLM] Lee et al. Flow Map Language Models: One-step Language Modeling via Continuous Denoising.
  arXiv:2602.16813, 2026-02-18. FMLM at 1, 2 and 4 steps (Table 13); validation perplexity not
  reported because measuring it "requires auxiliary training" (footnote 1).
  <https://arxiv.org/abs/2602.16813>. Code: <https://github.com/david3684/flm> (Apache-2.0).
- [FS-DFM] Karimi Monsefi et al. FS-DFM: Fast and Accurate Long Text Generation with Few-Step
  Diffusion Language Models. arXiv:2509.20624, 2025-09-24; ICLR 2026.
  <https://arxiv.org/abs/2509.20624>
- [Gemini Diffusion] Google DeepMind. Gemini Diffusion. Product page; benchmark scores and speed
  (1479 tokens per second) are vendor claims. Accessed 2026-11-23.
  <https://deepmind.google/models/gemini-diffusion/>
- [Hägele] A. Hägele et al. Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations. arXiv:2405.18392, 2024-05-28; NeurIPS 2024. <https://arxiv.org/abs/2405.18392>
- [Holm] S. Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of
  Statistics 6(2), 65–70, 1979. DOI 10.2307/4615733.
- [HRM] G. Wang et al. Hierarchical Reasoning Model. arXiv:2506.21734, 2025-06-26.
  <https://arxiv.org/abs/2506.21734>
- [Huginn] J. Geiping et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth
  Approach. arXiv:2502.05171, 2025-02-07; NeurIPS 2025. "Bad Run 2": validation perplexity "is the
  same whether 1 or 32 recurrences are used"; the model "has learned early to ignore the incoming
  state". <https://arxiv.org/abs/2502.05171>. Code:
  <https://github.com/seal-rg/recurrent-pretraining> (Apache-2.0).
- [IBC] P. Florence et al. Implicit Behavioral Cloning. arXiv:2109.00137, 2021-09-01; CoRL 2021.
  Model (c4) of G2 (method.md section 16.8). <https://arxiv.org/abs/2109.00137>. Code:
  <https://github.com/google-research/ibc> (Apache-2.0).
- [IRED] Y. Du, J. Mao, J. B. Tenenbaum. Learning Iterative Reasoning through Energy Diffusion.
  arXiv:2406.11179, 2024-06-17; ICML 2024. Model (c5) of G2 (method.md section 16.8).
  <https://arxiv.org/abs/2406.11179>. Code: <https://github.com/yilundu/ired_code_release>; its
  LICENSE file is the MIT License, copyright 2020 Phil Wang (read 2026-11-24).
- [LLaDA] S. Nie et al. Large Language Diffusion Models. arXiv:2502.09992, 2025-02-14.
  <https://arxiv.org/abs/2502.09992>
- [Llama 2] H. Touvron et al. Llama 2: Open Foundation and Fine-Tuned Chat Models.
  arXiv:2307.09288, 2023-07-18. Section 2.2: the standard Transformer with pre-normalization using
  RMSNorm, the SwiGLU activation and rotary positional embeddings.
  <https://arxiv.org/abs/2307.09288>
- [Looped] N. Saunshi et al. Reasoning with Latent Thoughts: On the Power of Looped Transformers.
  arXiv:2502.17416, 2025-02-24; ICLR 2025. <https://arxiv.org/abs/2502.17416>
- [Malloy] T. Malloy, P. K. Rajput, S. L. Nikiema, C. Gonzalez, T. F. Bissyandé. Metacognitive
  Reasoning in Energy Based Models using Instance Based Learning Theory. arXiv:2610.00399,
  2026-09-30. <https://arxiv.org/abs/2610.00399>
- [Mamba] A. Gu, T. Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
  arXiv:2312.00752, 2023-12-01. <https://arxiv.org/abs/2312.00752>
- [MAUVE] K. Pillutla et al. MAUVE: Measuring the Gap Between Neural Text and Human Text using
  Divergence Frontiers. arXiv:2102.01454, 2021-02-02; NeurIPS 2021.
  <https://arxiv.org/abs/2102.01454>
- [MD4] J. Shi et al. Simplified and Generalized Masked Diffusion for Discrete Data.
  arXiv:2406.04329, 2024-06-06; NeurIPS 2024. <https://arxiv.org/abs/2406.04329>
- [MDLM] S. S. Sahoo et al. Simple and Effective Masked Diffusion Language Models.
  arXiv:2406.07524, 2024-06-11; NeurIPS 2024. <https://arxiv.org/abs/2406.07524>. Code:
  <https://github.com/kuleshov-group/mdlm> (Apache-2.0).
- [Mercury] Inception Labs et al. Mercury: Ultra-Fast Language Models Based on Diffusion.
  arXiv:2506.17298, 2025-06-17. A vendor report; its quality and speed figures are vendor claims,
  some citing third-party evaluations (Artificial Analysis for speed, Copilot Arena for quality).
  <https://arxiv.org/abs/2506.17298>
- [Muennighoff] N. Muennighoff et al. Scaling Data-Constrained Language Models. arXiv:2305.16264,
  2023-05-25. <https://arxiv.org/abs/2305.16264>
- [Ni] Ni et al. Diffusion Language Models are Super Data Learners. arXiv:2511.03276, 2025-11-05.
  <https://arxiv.org/abs/2511.03276>
- [No Train No Gain] J. Kaddour et al. No Train No Gain: Revisiting Efficient Training Algorithms
  For Transformer-based Language Models. arXiv:2307.06440, 2023-07-12; NeurIPS 2023.
  <https://arxiv.org/abs/2307.06440>
- [Plaid] I. Gulrajani, T. Hashimoto. Likelihood-Based Diffusion Language Models.
  arXiv:2305.18619, 2023-05-30. <https://arxiv.org/abs/2305.18619>
- [Prabhudesai] M. Prabhudesai et al. Diffusion Beats Autoregressive in Data-Constrained Settings.
  arXiv:2507.15857, 2025-07-21. <https://arxiv.org/abs/2507.15857>
- [RADD] J. Ou et al. Your Absorbing Discrete Diffusion Secretly Models the Conditional
  Distributions of Clean Data. arXiv:2406.03736, 2024-06-06. <https://arxiv.org/abs/2406.03736>
- [RWKV-7] B. Peng et al. RWKV-7 "Goose" with Expressive Dynamic State Evolution.
  arXiv:2503.14456, 2025-03-18. <https://arxiv.org/abs/2503.14456>
- [SDTT] J. Deschenaux, C. Gulcehre. Beyond Autoregression: Fast LLMs via Self-Distillation
  Through Time. arXiv:2410.21035, 2024-10-28. <https://arxiv.org/abs/2410.21035>
- [SEDD] A. Lou et al. Discrete Diffusion Modeling by Estimating the Ratios of the Data
  Distribution. arXiv:2310.16834, 2023-10-25; ICML 2024. Its diffusion perplexities are reported
  as upper bounds ("≤"). <https://arxiv.org/abs/2310.16834>
- [Show Your Work] J. Dodge et al. Show Your Work: Improved Reporting of Experimental Results.
  arXiv:1909.03004, 2019-09-06. <https://arxiv.org/abs/1909.03004>
- [SSD-LM] X. Han et al. SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for
  Text Generation and Modular Control. arXiv:2210.17432, 2022-10-31; ACL 2023.
  <https://arxiv.org/abs/2210.17432>
- [text8] M. Mahoney. About the Test Data (text8: the first 100 000 000 bytes of the cleaned
  Wikipedia text, a 27-character alphabet of a–z and space). Accessed 2026-11-23.
  <https://www.mattmahoney.net/dc/textdata.html>
- [Texygen] Y. Zhu et al. Texygen: A Benchmarking Platform for Text Generation Models.
  arXiv:1802.01886, 2018-02-06. <https://arxiv.org/abs/1802.01886>
- [Tiny Shakespeare] A. Karpathy. char-rnn, file `data/tinyshakespeare/input.txt`, 2015; the v1.0
  text corpus (method.md section 18.1). The repository has no licence file.
  <https://github.com/karpathy/char-rnn>
- [TinyARM] Rauba et al. Tiny Autoregressive Recursive Models. arXiv:2603.08082, 2026-03-09; ICLR
  2026 Workshop. Figure 3: character accuracy at length 10 on addition, copy and reverse,
  autoregressive TRM 12%, 11% and 10%, dense Transformer 80%, 100% and 100%.
  <https://arxiv.org/abs/2603.08082>
- [TinyStories] R. Eldan, Y. Li. TinyStories: How Small Can Language Models Be and Still Speak
  Coherent English? arXiv:2305.07759, 2023-05-12. Dataset licence CDLA-Sharing-1.0:
  <https://huggingface.co/datasets/roneneldan/TinyStories>. <https://arxiv.org/abs/2305.07759>
- [TRM] A. Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny Networks.
  arXiv:2510.04871, 2025-10-06. <https://arxiv.org/abs/2510.04871>
- [TTT] Y. Sun et al. Learning to (Learn at Test Time): RNNs with Expressive Hidden States.
  arXiv:2407.04620, 2024-07-05. <https://arxiv.org/abs/2407.04620>
