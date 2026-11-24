# Method specification

This document records the implementation of the autoregressive and EBT families and the v1.0
experiments with them on three tasks: regression, MNIST denoising and a character-level language
model. The project's protocol, its families and its versions are in [design.md](design.md); the
questions and the v1.0 answers are in [questions.md](questions.md).

This document specifies the Energy-Based Transformer (EBT) method as described in the paper and as
implemented in the authors' code. Every later part of this repository is checked against it.

Sources:

- Paper: A. Gladstone et al., "Energy-Based Transformers are Scalable Learners and Thinkers",
  arXiv:2507.02092v1 (2025). Cited below as "§" plus a section, table, algorithm or equation number.
- Code: <https://github.com/alexiglad/EBT>, branch `main`, read on 2026-11-18. Cited as
  `path/file.py: function` with line numbers where they help. The code is described here in our
  own words; no code from that repository is copied.
- Other sources are cited by bracketed keys, such as [Bishop], that refer to [design.md,
  References](design.md#references).

Each point is labelled:

- agree: the paper and the code say the same thing;
- disagree: they differ, or the paper differs from itself (all such points are collected in
  section 13);
- not confirmed: neither source settles the point (collected in section 14).

Each section opens with its label in a status line. These labels compare the paper with the
authors' code only. Where this project follows the authors' code on a point listed among the
corrections of section 13 (corrections 13.1 to 13.4), the status line of every section that the
correction cites adds "this project: matches the code, correction 13.n".

## 1. Notation

- x: the context (observed tokens, or nothing for image denoising, see 3.2).
- y: the target. ŷ: a candidate prediction. ŷ_i: the candidate after i thinking steps.
- E_θ(x, ŷ): a scalar energy, lower meaning more compatible (§3.1). The model is unnormalized: no
  partition function is computed or approximated.
- α: the step size of the thinking loop. N: the number of thinking steps. σ: the Langevin noise
  standard deviation. M: the number of candidates in best-of-M inference. Sections 19 and 20, the
  result files and questions.md name the candidate count N, as in best-of-N (`best_of_n`); there N
  is the M of this list.
- B, S, V, D: batch size, sequence length, vocabulary size, embedding width.

The paper defines two configurations (§C.2):

- S1 ("System 1") is tuned for stable training. The paper states that "Many of the pretraining
  scaling experiments conducted in Section 4 are with S1 models" (§C.2).
- S2 ("System 2") is tuned for thinking. Its text thinking results use it.

## 2. Overview of one training step

1. Build an initial candidate ŷ_0 (section 4).
2. Repeat N times: evaluate E_θ(x, ŷ_i), take its gradient with respect to ŷ_i, and step
   ŷ_{i+1} = ŷ_i − α ∇_ŷ E_θ(x, ŷ_i) (§3.2 Eq. 1, Algorithm 1). Langevin noise, randomized α and a
   randomized N are optional (sections 7 and 8).
3. Compare the candidate(s) with y using a standard supervised loss (section 6).
4. Backpropagate into θ. Because ŷ_{i+1} depends on θ through ∇_ŷ E_θ, this needs gradients of
   gradients, computed as Hessian-vector products (§3.2). Which steps contribute to the parameter
   gradient depends on the training mode (section 6).

Inference runs step 2 without step 3 and 4, optionally for several candidates with the
lowest-energy one returned (§3.3, Algorithm 2).

## 3. Energy function E(x, ŷ)

### 3.1 Text: causal EBT

Status: agree; this project: matches the code, correction 13.2.

The transformer receives a sequence of length 2S:

- the S observed token embeddings (first half);
- one predicted embedding for each next position (second half).

The text configurations prepend one more embedding, the step-index token of section 3.3, at
sequence position 0, which every observed token and every prediction attends to.

Attention is arranged so that:

- an observed position attends causally to observed positions only;
- the predicted embedding for position j+1 attends to observed positions 1..j and to itself only
  (§C.3, Eq. 3).

So each prediction is made independently of the other predictions, and the observed half does not
depend on any prediction.

After the last block, a bias-free linear layer maps each predicted-half representation to one
scalar. That scalar is the energy of that position's candidate
(`model/ar_ebt_default.py: EBTDefault.forward`, lines 525–570; the bias is omitted because a
constant offset does not change the energy minimum).

The thinking gradient is taken of the sum of all per-position energies
(`model/nlp/ebt.py: EBT_NLP.forward`, line 153). Because the predictions are independent, the
gradient of the sum with respect to the candidate at position j equals the gradient of position j's
own energy.

The paper gives two implementations of this attention pattern:

- an efficient one that computes observed-to-observed and predicted-to-observed scores separately
  and inserts each prediction's self-score on the superdiagonal (§C.3,
  `model/ar_ebt_default.py: Attention.forward`, lines 243–369);
- a simpler one that uses full 2S × 2S attention with a generalized causal mask, about twice the
  attention FLOPs of the efficient version (§C.4).

Query, key and value weights are shared between the two halves (§C.3, footnote 13).

The backbone follows the Llama 2 design: RMSNorm, SwiGLU feed-forward, rotary position embeddings
and Xavier initialization (§I.1.2, §D).

### 3.2 Images: bidirectional EBT

Status: agree, with one disagreement in 4.2.

The image energy is a DiT-style transformer over image patches with no diffusion time input
(`model/bi_ebt_adaln.py: EBT`). The final layer maps each patch representation to one scalar,
again without bias (`model/bi_ebt_adaln.py: FinalLayer`). The image energy is the mean of the patch
scalars (`model/img/ebt_denoise.py: EBT_IMG_Denoise.forward`, line 103).

For denoising the energy is unconditional. Its only input besides the candidate image is a single
learned embedding that is the same for every example
(`model/img/ebt_denoise.py: EBT_IMG_Denoise.__init__`, the `learned_adaln_condition` embedding).
The noisy image enters only as the starting point of the thinking loop (section 4.2).

This project keeps that design.

### 3.3 Step-conditioned energy

Status: agree; this project: matches the code, correction 13.2.

The text configurations in the paper use a learned "step" embedding prepended to the sequence
(`ebt_type = time_embed`). It makes the energy depend on the index of the current thinking step
(§I.1.2, `model/ar_ebt_time_embed.py: EBTTimeConcat`).

- S1 text models use one such landscape per step, i.e. N different energy functions.
- S2 models perform all steps on a single landscape (§I.3). The S2 script sets the number of
  landscapes to 1 (`job_scripts/nlp/pretrain/ebt_s2.sh`, `--mcmc_num_steps 1`), so the S2 model
  is `EBTTimeConcat` with one learned embedding, the same at every step. It still changes the
  energy: the embedding is a token at sequence position 0 that every observed token and every
  prediction attends to, and its output is removed before the energy head. The positions shift
  by one: observed tokens are at 1..S and the prediction for the character after observed
  token j at j + 2.

This project implements the S2 form for the text EBT: one learned step-index embedding, in both
attention layouts (section 18.3; correction 13.2).

The image model does not use step conditioning: `ebt_type` is passed by the image scripts but
ignored by `model/model_utils.py: setup_bidirectional_ebt`.

## 4. Candidate representation and initialization

### 4.1 Text

Status: agree, except the S2 options of D7; this project: matches the code, correction 13.1.

The candidate is a tensor of unnormalized logits over the vocabulary, one vector per position
(shape B × S × V).

It is initialized by sampling each logit from N(0, s²) with s = `gaussian_random_noise_scaling`,
default 1 (`model/nlp/ebt.py: EBT_NLP.corrupt_embeddings`, `denoising_initial_condition =
random_noise`; Algorithm 1 line 1). Zeros are also supported but not used in the paper's scripts.

One thinking step of the authors' S2 text code (`model/nlp/ebt.py: EBT_NLP.forward`, lines
127–180, with `--normalize_initial_condition` and without
`--normalize_initial_condition_only_first_step`, as in the S2 script):

1. Langevin noise is added to the candidate: ŷ + η (section 7).
2. Softmax over the vocabulary: p = softmax(ŷ + η). The code overwrites the candidate variable
   with p. The paper reports that without this normalization training was unstable (§I.1.5).
3. p is mapped to a D-dimensional embedding by a learned bias-free linear map from V to D
   (`vocab_to_embed`, lines 39–49). The listings in §I.2 describe the same map.
4. The gradient of the summed energy is taken with respect to p, not with respect to the logits.
5. The step starts from p: p − α · exp(E / 9) · ∇_p E (section 10). Under S2 truncation the
   energy in the step size is detached on every step except the last.
6. `norm_pred`, a learnable RMSNorm over the vocabulary, is applied to the result (lines
   179–180; disagreement D7). Its output is the candidate of the next step and, after the last
   step, the logits of the cross-entropy.

This project implements exactly this update. The thinking loop has one extension point for it
(`EnergyModel.transform_candidate`, `src/ebt/core/energy.py`): a transform of the noisy candidate
whose output is both the variable of differentiation and the base of the step. It is softmax for
text and the identity for regression and images, whose updates are unchanged
(`tests/test_identity_transform.py` runs a separate implementation of the update without the
transform on the same inputs and generators and requires bitwise equality on the machine that
runs it).
`encode_candidate` then maps that variable to the energy's input, `vocab_to_embed` for text. The
replay buffer stores the final candidate after `norm_pred`, as upstream.
`tests/test_text_models.py` checks one step against a separate implementation of steps 1 to 6.

The paper writes the thinking step as a gradient step on the candidate ŷ (§3.2 Eq. 1, Algorithm
1). For text the authors' code differentiates with respect to the probabilities p and steps from
p, and this project does the same (correction 13.1). Results computed with the gradient taken
with respect to the noisy logits and the step taken from them are not reused.

The code offers an alternative to step 3: the probability-weighted sum of the input embedding
table (`vocab_to_embed_uses_prob_dist`). No paper script uses it.

### 4.2 Images

Status: disagree with Algorithm 1, see D3.

The candidate is the image itself in pixel space: three RGB channels, no VAE
(`model/model_utils.py: setup_bidirectional_ebt`, branch `image_task == "denoising"`).

The thinking loop starts from the noisy image (`model/img/ebt_denoise.py: EBT_IMG_Denoise.forward`,
line 54), not from Gaussian noise as in Algorithm 1. The `denoising_initial_condition` flag is not
read by the image model.

Noise follows the DiT linear variance schedule from 1e-4 to 2e-2 (§4.3). It is variance-preserving:

x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε, ε ~ N(0, I)

(`model/diffusion/gaussian_diffusion.py: GaussianDiffusion.q_sample`). The clean image is scaled
down as well as noised.

The noise level is given as a fraction of the schedule: 0.1 during training and 0.2 for the
out-of-distribution test (§4.3). The code implements this by building a schedule of
`diffusion_steps / noise_level` steps and noising at a fixed step index
(`model/img/ebt_denoise.py: EBT_IMG_Denoise.__init__` line 36 and `forward_loss_wrapper` lines
134–138; `inference/img/generate_image.py: generate_image`).

PSNR is computed from the mean squared error on pixel values clamped to [0, 255], with peak value
255 (`inference/img/generate_image.py: generate_image`). This project computes PSNR from MSE in the
same way, whatever training loss is used.

## 5. Thinking loop

### 5.1 One step

Status: agree.

For each step i:

1. Optionally add Langevin noise (section 7).
2. Evaluate the energy.
3. Compute g = ∇_ŷ E_θ(x, ŷ_i) by automatic differentiation, with gradients enabled even during
   evaluation (`torch.set_grad_enabled(True)` around the loop in both models).
4. Update ŷ_{i+1} = ŷ_i − α g (§3.2 Eq. 1; `model/nlp/ebt.py: EBT_NLP.forward`, line 171;
   `model/img/ebt_denoise.py: EBT_IMG_Denoise.forward`, line 122).

For text, ŷ_i in steps 2 to 4 is the softmax of the noisy logits, and `norm_pred` follows the
update (section 4.1).

The code raises an error if g contains NaN or infinite values.

### 5.2 Inference with self-verification

Status: agree.

M candidates are initialized independently and each is optimized for N steps. The candidate with
the lowest final energy is returned (§3.3, Algorithm 2).

For text the selection is made per position, not per sequence
(`model/nlp/ebt.py: EBT_NLP.ebt_advanced_inference`, lines 395–435). The code also offers selection
by highest energy and by largest energy decrease (`infer_energy_sampling_technique` values `max`
and `max_gap`); the paper uses the lowest energy (§3.3).

The number of inference steps can differ from the number used in training
(`infer_ebt_num_steps`).

## 6. Training objective and gradient flow

### 6.1 Losses

Status: text: agree; images: disagree, see D1.

- Text: cross-entropy between the updated logits and the next token, ignoring padding
  (`model/nlp/ebt.py: EBT_NLP.forward_loss_wrapper`; §3.2).
- Images: Smooth L1 with β = 1 between the denoised and the clean image
  (`model/img/ebt_denoise.py: EBT_IMG_Denoise.forward_loss_wrapper`; §I.2 Listing 2). §3.2 states
  mean squared error.

This project uses Smooth L1 (β = 1) as the image default and reports mean squared error as a
variant.

### 6.2 Training modes

Two switches combine into the three modes this project implements:

- whether ŷ is detached between steps (`no_mcmc_detach`);
- whether only the last step's loss and second-order graph are kept (`truncate_mcmc`).

| Mode | Detach between steps | Second-order graph | Loss | Source |
|---|---|---|---|---|
| S1 | yes | at every step | mean over the N per-step losses | §C.2, §I.2; `ebt_s1.sh`; `forward_loss_wrapper` lines 226–230 |
| S2 | no | final step only | final step only | §C.2, §D.5, Table D.4; `ebt_s2.sh`; `forward` lines 147–151 |
| Full unroll | no | at every step | final step only | Algorithm 1 and the text of §3.2 |

S1:

- Each step starts from a detached copy of the previous candidate.
- Each step's update is differentiated with respect to θ (`create_graph=True`), so the per-step
  loss reaches θ through that step's gradient.
- The losses of all steps are averaged.

S2:

- The candidate is not detached.
- Gradients at all steps except the last are computed without a second-order graph
  (`create_graph=False`), so they enter the chain as constants.
- Only the final step's loss is used.

Which parameters the earlier steps reach under this truncation (U1, resolved by
`tests/test_truncation.py`):

- Energy parameters: not at all. Their gradient equals that of a single S2 step started from the
  detached ŷ_{k−1}, with or without the energy-scaled step size
  (`test_s2_energy_parameters_are_reached_by_the_final_step_only`).
- Parameters of the post-step transform (`norm_pred` in the authors' text S2 configuration):
  yes. Detaching ŷ_{k−1} changes their gradient
  (`test_s2_post_step_transform_is_reached_by_earlier_steps`).
- The step size α would be reached as well if it were learnable
  (`test_s2_learnable_step_size_would_be_reached_by_earlier_steps`). S2 keeps α fixed
  (section 10), so this path is not used.
- Without a post-step transform and with fixed α, S2 trains every parameter through the final
  step only (`test_s2_without_post_step_transform_learns_from_the_final_step_only`).

The statement derived from reading the code is therefore confirmed: in the authors' S2
configuration, the earlier steps reach the parameters only through the RMSNorm applied to the
logits after each step. The energy-scaled step size opens no further path, because its energy is
detached on non-final steps (section 10).

Full unroll:

- The candidate is not detached and every step keeps its second-order graph.
- The loss is taken on ŷ_N only.

This is the literal reading of Algorithm 1 and of §3.2 ("backpropagated through the entire
optimization process"). The authors' code can build the full graph (`no_mcmc_detach` without
`truncate_mcmc`), but then it averages the loss over steps rather than using only the final step.
No paper script uses that combination.

§I.2 recommends:

- not detaching when the loss is taken only at the last step;
- the S1 combination (detach, per-step loss) for stability.

### 6.3 Cost of a step

Status: agree, approximate by the authors' own statement.

A training step that is differentiated to second order costs about one forward pass, one backward
pass for ∇_ŷ E, and one Hessian-vector product, which costs about one more backward pass
(§D.5, citing [Dagréou]). The authors call these estimates approximate
(§D.5).

This project reports cost as counts of forward passes, first-order backward passes and
second-order backward passes, not as time (`src/ebt/core/cost.py: PassCount`). The counts are
made where the passes happen in `src/ebt/core/thinking.py: ThinkingLoop`:

- every energy evaluation is one forward pass;
- every gradient of the energy with respect to the candidate is one first-order backward pass;
- every such gradient that keeps its graph during training is one second-order pass: the
  training loss later backpropagates through it, which costs one Hessian-vector product plus the
  backward pass through that step's energy.

Counts are per input example: a batched pass counts once for every example in the batch, and
best-of-M selection costs M times the passes of one candidate. For k steps
(`tests/test_cost.py`):

| Call | Forward | First-order backward | Second-order |
|---|---|---|---|
| Training, S1 | k | k | k |
| Training, S2 | k | k | 1 |
| Training, full unroll | k | k | k |
| Inference, k steps | k, plus 1 if the final energy is requested | k | 0 |
| Best-of-M, k steps | M · (k + 1) | M · k | 0 |

The extra forward pass is made only when the caller requests the final energy (`final_energy` in
`src/ebt/core/thinking.py: ThinkingLoop`); best-of-M always requests it, to rank the candidates.
The trained inference of the tasks does not: per example, text 4 / 4 / 0 (section 18.9) and MNIST
2 / 2 / 0 (`results/mnist/denoising/full/table.md`).

## 7. Langevin noise

Status: disagree on placement, see D4.

§3.3 Eq. 2 adds noise after the gradient step: ŷ_{i+1} = ŷ_i − α ∇E(x, ŷ_i) + η_i with
η_i ~ N(0, σ²).

The code adds the noise to the candidate before the energy is evaluated, and takes the gradient at
the noisy point: ŷ_{i+1} = (ŷ_i + η_i) − α ∇E(x, ŷ_i + η_i)
(`model/nlp/ebt.py: EBT_NLP.forward`, lines 123–125; `model/img/ebt_denoise.py`, lines 96–98).

- For text the noise is added to the logits, before softmax.
- σ = 3 in text S2 (Table D.4, `ebt_s2.sh`).
- σ = 0 for images (`job_scripts/img/pretrain_denoising/ebt_s2.sh`); the image S1 script does not
  enable it.

Inference has its own noise level (`infer_langevin_dynamics_noise`). The text inference script
sets it to 1.

Upstream text validation applies the training σ, because the text model has no switch that turns
the noise off outside training (`model/nlp/ebt.py: EBT_NLP.forward`, line 123); the image model has
one (`no_langevin_during_eval`). In this project every evaluation uses the inference noise
`InferenceConfig.langevin_std` (default 0), so evaluation without inference noise is
deterministic.

The code stores σ as a parameter that is frozen unless `langevin_dynamics_noise_learnable` is set;
no paper script sets it.

## 8. Randomized step size and step count

### 8.1 Step size

Status: agree.

During training only, α is replaced by a value drawn uniformly from [α / r, α · r] with r = 2
(Table D.4; `randomize_mcmc_step_size_scale`; `model/nlp/ebt.py: EBT_NLP.forward`, lines 78–84).

- For text the draw is made separately for every batch element and position. §I.1.3 reports that
  a single value per batch hurt convergence.
- For images the draw is made per batch element (`model/img/ebt_denoise.py`, lines 57–63).

### 8.2 Step count

Status: disagree on the range, see D9.

During training only, the number of steps is random. The code draws an integer k uniformly from
{k_min, …, R + 1} once per batch, using `randomize_mcmc_num_steps_min` = k_min (1 when unset) and
`randomize_mcmc_num_steps` = R (`model/nlp/ebt.py: EBT_NLP.forward`, lines 93–113). It then repeats
the step on one landscape k times.

- With step conditioning and more than one landscape, the repetition can apply to every landscape
  or only to the last one (`randomize_mcmc_num_steps_final_landscape`).
- In evaluation the maximum, R + 1 steps, is used.

§I.1.3 states that a step count drawn per example was not tried: "we did not experiment with
such a configuration".

## 9. Replay buffer

Status: agree for text; not used for images.

The buffer is a first-in-first-out list of single examples (`model/replay_buffer.py:
CausalReplayBuffer`). Each record holds one example's full token sequence and its final predicted
logits from the step in which it was trained.

For each training batch:

1. A random subset of stored records is drawn, without replacement, of size min(f · B, current
   buffer size), where f = `mcmc_replay_buffer_sample_bs_percent`.
2. These are appended after the fresh examples.
3. Fresh examples start from the usual random initialization; replayed examples start from their
   stored logits (`model/nlp/ebt.py: EBT_NLP.forward` lines 88–90, `forward_loss_wrapper` lines
   192–196).
4. After the step, the fresh examples of the batch are added with their final logits. Replayed
   examples are not stored again: `CausalReplayBuffer.update` iterates over the fresh batch only.
   The oldest records are dropped beyond capacity.

The effect is that a replayed example continues from where an earlier training step left it,
which simulates a longer optimization trajectory (§3.3).

Defaults are capacity 192 and f = 0.5 (`train_model.py`). The S2 script does not override them
(see U2).

The image model raises an error if the buffer is enabled (`model/img/ebt_denoise.py`, line 46).

## 10. Step-size parameterization

Status: agree.

α is a raw scalar parameter, not log-parameterized. It is clamped below at 1e-4 before use
(`model/nlp/ebt.py: EBT_NLP.forward`, line 77; `model/img/ebt_denoise.py`, line 56).

- In S1 it is learnable. It sits in its own optimizer group with learning rate equal to a
  multiplier times the base learning rate, and with no weight decay (`base_model_trainer.py`,
  lines 372–422; §I.1.1). §I.2 recommends a multiplier of about 3 × α.
- In S2 it is fixed (§C.2, §I.3, Table D.4). The values used are in disagreement D8.

The S2 text script also scales α per position by exp(E / T) with T = 9, the code default
(`scale_alpha_with_energy`, `scale_alpha_with_energy_temp`; `model/nlp/ebt.py: EBT_NLP.forward`,
lines 164–169). Under truncation the energy in that factor is detached on all steps except the
last. The paper does not describe this scaling (see D7).

## 11. Defaults adopted in this project

Status: this project: matches the code, correction 13.2.

- Each task's training default follows what the authors ran for the corresponding task:
  - text: S1 for scaling, S2 for thinking (§D.1.1, §D.1.2);
  - images: S1 (§D.3, "it was not necessary to train EBTs with the S2 hyperparameters").
- The thinking experiments (questions G3 to G5 in `questions.md`) use, for each task, the mode
  the authors used for that task's thinking results:
  - text: S2 (§D.1.2);
  - images: S1 (§D.3);
  - regression: S2.
- The other modes are G6 variants: S1 and full unroll for text and regression, S2 and full
  unroll for images. They are planned for W5 in v1.1 and are not compared in v1.0 (questions.md,
  G6).
- Image loss: Smooth L1 (β = 1) by default, mean squared error as a reported variant; PSNR is
  always computed from mean squared error.
- Image energy: unconditional, thinking starts from the noisy image (section 3.2).
- One energy landscape for all steps (section 3.3). The text EBT has the authors' S2 form, one
  learned step-index token that is the same at every step; the image model has none, as the
  authors' image model ignores step conditioning. This deviates from the authors' text S1
  configuration, which uses one landscape per step. In this project text S1 is a G6 variant,
  planned for W5 in v1.1, whose results will state this deviation (correction 13.2).
- The multimodal regression task has no counterpart in the paper. Its candidate is the
  real-valued ŷ itself, updated as in section 5.1 without softmax or embedding map. Its thinking
  experiments use S2.

## 12. Code options not used in the paper

Status: excluded.

These exist in the authors' code but the paper reports no experiment with them, or reports that
they were not used:

- clamping of the thinking gradient (`clamp_futures_grad`; §I.1.4 says no paper experiment uses
  it);
- absolute clamping of the logits (`absolute_clamp`);
- logit sharpening (`sharpen_predicted_distribution`);
- label smoothing that varies across steps (`soften_target_prob_dist`);
- accept-only-lower-energy updates at inference (`infer_accept_lower_energies`). An update of
  this kind is part of the IRED method and is used for model (c5) of question G2 only
  (section 16.8), not as an option of the EBT;
- memory-augmented `vocab_to_embed` variants;
- adaptive layer normalization in place of the step embedding (`ebt_type = adaln`). §I.1.2 reports
  that the step embedding worked better.

This project does not implement them.

One option is used outside the paper's configurations: the contrastive energy loss
(`contrastive_loss`, `EBT_NLP.calculate_contrastive_loss`). It is used only for the regression
variants (c1), (c2) and (c3) of question G2 (section 16.4) and in no other task.

## 13. Disagreements

D1. Image loss.
- §3.2: mean squared error for image denoising.
- Code (`EBT_IMG_Denoise.forward_loss_wrapper`) and §I.2 Listing 2: Smooth L1 with β = 1.
- Project default: Smooth L1.

D2. Gradient flow through the thinking steps.
- §3.2 says the loss is "backpropagated through the entire optimization process", and Algorithm 1
  computes the loss on ŷ_N only.
- §C.2 and §D.5 state that S2 truncates backpropagation, and that S1 detaches between steps and
  takes the loss at every step. The scripts and code match §C.2.
- No paper configuration backpropagates the final-step loss through all steps. Section 6.2 defines
  that variant as "full unroll".

D3. Initial candidate for images.
- Algorithm 1: ŷ_0 ~ N(0, I).
- Image denoising code: starts from the noisy image, with an unconditional energy (sections 3.2
  and 4.2).
- The `denoising_initial_condition = random_noise` flag in the image scripts has no effect on the
  image model.

D4. Langevin noise placement.
- Eq. 2: noise added after the gradient step.
- Code: noise added before the energy evaluation, with the gradient taken at the noisy point
  (section 7).

D5. Size of the S2 text model.
- §D.1.2: the thinking experiments use xxs models.
- Table D.4: model size "Small".
- The S2 script sets `RUN_NAME="ebt-xxs-…"` and reads the size from the run name, so the script
  agrees with §D.1.2; only Table D.4 says Small. This project follows §D.1.2 and the script
  (xxs, 384 × 6) as the largest size its budget rule considers (section 18.7).

D6. Shape in §I.2 Listing 1. The listing initializes the logits with the shape of the next-token
embeddings (B × S × D) while its comment says B × S × V. The code samples B × S × V
(`EBT_NLP.corrupt_embeddings`).

D7. S2 options missing from Table D.4. The S2 text script enables two options that Table D.4 and
the text do not mention:
- RMSNorm on the logits after every step (`--norm_pred`);
- energy-scaled step size (`--scale_alpha_with_energy`).
The text inference script also enables both.

D8. Text step sizes.
- Table D.3 (S1): α = 500, multiplier 1500. The upstream minimal example
  (`example_code/minimal_nlp_training_loop.py`) uses the same values.
- `job_scripts/nlp/pretrain/ebt_s1.sh`: α = 0.5, multiplier 1.5.
- Table D.4 (S2): α = 5. `ebt_s2.sh`: α = 0.5.
- §I.1.1: "between 5 and 500 for text".

D9. S2 step count.
- Table D.4: 2–3 steps.
- `ebt_s2.sh`: minimum 2 and R = 3. By the rule in section 8.2 this gives k ∈ {2, 3, 4} in
  training and 4 in evaluation.

D10. Image training noise level in the S2 script.
- §4.3: training at noise level 0.1.
- `job_scripts/img/pretrain_denoising/ebt_s2.sh`: trains at 0.2.
- The S1 image script uses 0.1. That matches §D.3, which says the image results did not need S2
  settings.

D11. Feed-forward width of the text models.
- §D.1.1: no feed-forward multiplier.
- Code: `--ffn_dim_multiplier` defaults to 4.0 (`train_model.py`, line 484), and the SwiGLU
  hidden width is int(dim · 4.0) without rounding to a multiple (`model/ar_transformer.py`,
  line 345; the `FeedForward` of `model/ar_ebt_time_embed.py`). No text script overrides it;
  only the image scripts and the minimal example pass 1.
- Project: hidden width 4D for the text EBT and the baseline, following the code, as everywhere
  else where paper and code disagree (section 18.3).

D12. Langevin noise when the text model is evaluated.
- Code: the NLP model applies the training σ = 3 at every thinking step also at validation
  (`model/nlp/ebt.py`, `EBT_NLP.forward`); it has no evaluation switch, unlike
  `no_langevin_during_eval` of the image model (`ebt_denoise.py`).
- Project: the text EBT is evaluated without Langevin noise (σ = 0, `InferenceConfig`;
  section 7), kept deliberately: the evaluation is deterministic and reproducible. An
  evaluation with σ = 3 for comparison is not part of v1.0; it is item 27 of the v1.1
  pre-registration list (design.md section 4.9).

D13. Learning-rate schedule length of the text models.
- Code: linear warm-up for 1% of the schedule (10 000 of 1 000 000 steps), then cosine decay to
  lr / 10.
- Project: linear warm-up for 10% of the T training steps, then cosine decay to lr / 10, as in
  the code, over the T steps of this run (section 18.4).

D14. Float32 matrix-multiplication precision of the text models.
- Code: the text scripts set the float32 matrix-multiplication precision to "medium".
- Project: strict float32 (section 18.4).

Corrections 13.1 to 13.4: points on which this project follows the authors' code. Each gives
what the paper says, what the code does and what this project does. Results computed with a
model that did otherwise are not reused.

### 13.1 Text thinking step

Cited in section 4.1. The paper writes the step as ŷ_{i+1} = ŷ_i − α ∇_ŷ E_θ(x, ŷ_i) (§3.2 Eq. 1,
Algorithm 1). The authors' S2 text code takes the gradient with respect to p = softmax(ŷ + η) and
steps from p. This project does the same; results of a text EBT that differentiated with respect to
the logits and stepped from them are not reused.

### 13.2 Step-index token

Cited in sections 3.1, 3.3 and 11. The paper says that S2 models perform all steps on a single
landscape (§I.3). The authors' S2 model is `EBTTimeConcat` with one learned embedding at sequence
position 0, so its energy has a step-index token. This project's text EBT has it; results of a text
EBT without the token are not reused.

### 13.3 Initialization of the text models

Cited in section 18.3. The authors' `init_whole_model_weights` uses Xavier-normal weights for the
linear layers, with the query, key and value maps initialized as separate D × D blocks, and N(0,
0.02²) for the token embedding. The step-index embedding is never re-initialized
(`model/ar_ebt_time_embed.py`, line 701), so it keeps PyTorch's default N(0, 1). Both text models
of this project use that scheme, with N(0, 1) for the step-index embedding. Results of text models
with Xavier-uniform weights for every linear layer and embedding, or with N(0, 0.02²) for the
step-index embedding, are not reused.

### 13.4 Feed-forward width of the text models

Cited in D11, section 18.3. §D.1.1 gives no feed-forward multiplier, which reads as a SwiGLU hidden
width of D. The authors' code uses 4D. This project uses 4D for both text models; results with a
hidden width of D are not reused.

## 14. Not confirmed

U1. Gradient path under S2 truncation. Resolved: confirmed by `tests/test_truncation.py`. Under
truncation the earlier steps reach no energy parameter; they reach only the parameters of the
post-step transform (the `norm_pred` RMSNorm), and would reach α if it were learnable, which S2
does not allow. Details in section 6.2.

U2. Replay buffer settings behind the paper's results. The code defaults are capacity 192 and
fraction 0.5, and the S2 script does not override them. The paper does not state the values.

U3. Which step sizes produced the reported text results. See D8: tables and scripts differ by up to
a factor of 1000.

U4. Number of thinking steps behind the image results.
- The S1 image script trains with one step.
- The image inference script uses two steps.
- §D.3 states that the best out-of-distribution results came from applying the model three times
  recursively to its own output. The step count of each application is not stated.

U5. Settings behind the text thinking figures (Figures 6 and 7): step counts, number of samples and
inference noise. The text inference script uses two steps and inference noise 1. The paper does not
tie these to a figure.

U6. Equivalence of the two causal attention implementations. §C.3 and §C.4 describe them as
computing the same attention pattern. Resolved for this implementation: both layouts give the
same energies and first- and second-order gradients (section 18.3).

U7. Cost estimate. The F + B + B per-step cost of §D.5 is the authors' approximation. This project
counts passes instead of FLOPs (section 6.3).

## 15. Name mapping

This project uses its own configuration names, defined as dataclasses in `src/ebt/config.py`.
The tables map them to the authors' flags.

Implemented:

| This project | Upstream flag | Meaning |
|---|---|---|
| `ThinkingConfig.mode` = `s1` / `s2` / `full` | `no_mcmc_detach`, `truncate_mcmc` | training mode, section 6.2 |
| `ThinkingConfig.steps_min`, `steps_max` | `randomize_mcmc_num_steps_min` (k_min), `randomize_mcmc_num_steps` (R), `mcmc_num_steps` | training step count, section 8.2 |
| `ThinkingConfig.step_size` | `mcmc_step_size` | α |
| `ThinkingConfig.step_size_learnable` | `mcmc_step_size_learnable` | learnable α, section 10 |
| `ThinkingConfig.step_size_jitter` | `randomize_mcmc_step_size_scale` | r in section 8.1 |
| `ThinkingConfig.energy_scaled_step` | `scale_alpha_with_energy` | step size α · exp(E / T), section 10 |
| constant `ENERGY_SCALE_TEMPERATURE = 9` in `src/ebt/core/step_size.py` | `scale_alpha_with_energy_temp` | T; no upstream script changes the default 9 |
| `ThinkingConfig.langevin_std` | `langevin_dynamics_noise` | σ in section 7 |
| `InferenceConfig.steps` | `infer_ebt_num_steps` | thinking steps at inference |
| `InferenceConfig.samples` | `infer_generated_samples` | M in best-of-M |
| `InferenceConfig.langevin_std` | `infer_langevin_dynamics_noise` | inference noise |
| `ReplayConfig.capacity`, `ReplayConfig.fraction` | `mcmc_replay_buffer_size`, `mcmc_replay_buffer_sample_bs_percent` | section 9 |
| `TrainingConfig.learning_rate` | `peak_learning_rate` | base learning rate |
| `TrainingConfig.train_steps`, `warmup_steps` | `max_steps`, `warm_up_steps` | schedule length |
| `TrainingConfig.batch_size` | `batch_size_per_device` | batch size |
| `TrainingConfig.weight_decay` | `weight_decay` | AdamW weight decay, not applied to α |
| `TrainingConfig.min_lr_scale` | `min_lr_scale` | cosine decay to learning_rate / min_lr_scale |
| `TrainingConfig.gradient_clip` | `gradient_clip_val` | gradient-norm clip |
| `TrainingConfig.step_size_lr_multiplier` | `mcmc_step_size_lr_multiplier` | learning rate of α relative to the base rate |
| `ObjectiveConfig.reconstruction_weight` | `reconstruction_coeff` | weight of the trajectory loss |
| `ObjectiveConfig.contrastive_weight` | `contrastive_loss_coeff` (with `contrastive_loss`) | weight of the contrastive loss, section 16.4 |
| `ObjectiveConfig.energy_regularization_weight` | none: not in the upstream code | λ of the energy-magnitude penalty, section 16.4 |
| `MLPConfig.activation` | none: the upstream MLPs are fixed | activation of the G2 MLPs, section 16.8 |
| `PatchTransformerConfig.patch` | `patch_size` | patch size, section 17.3 |
| `PatchTransformerConfig.dim`, `depth`, `heads` | `model_size` presets: `embedding_dim`, `num_transformer_blocks`, `multiheaded_attention_heads` (`model/model_utils.py`) | image model size, section 17.3 |
| `PatchTransformerConfig.mlp_ratio` | `ffn_dim_multiplier` | block MLP width relative to `dim` |
| `DIFFUSION_STEPS`, `NOISE_LEVEL` in the image task | `diffusion_steps`, `denoise_images_noise_level` | noise schedule, section 17.2 |

Step count. With one landscape (section 11), the upstream training step count is drawn from
{k_min, …, R + 1}, with k_min = 1 when `randomize_mcmc_num_steps_min` is unset (section 8.2).
So `steps_min` = k_min and `steps_max` = R + 1. Without step-count randomization (R = 0),
`steps_min` = `steps_max` = `mcmc_num_steps`. Examples:

| Upstream configuration | Upstream flags | `steps_min`, `steps_max` |
|---|---|---|
| text S2, `ebt_s2.sh` | k_min = 2, R = 3 | 2, 4, so k ∈ {2, 3, 4} |
| images S2, `ebt_s2.sh` | k_min = 2, R = 2 | 2, 3 |
| text S1, `ebt_s1.sh` | `mcmc_num_steps` 2, R = 0 | 2, 2 |

Implemented in the task code rather than as configuration fields:

| This project | Upstream flag | Meaning |
|---|---|---|
| `text_lm.initial_logits` (scale 1); `regression.initial_candidates` (`init_std`) | `gaussian_random_noise_scaling` | scale of the random initial candidates: 1 for text, the default of the code; the regression task derives its scale from the data (section 16.2) |
| `EnergyModel.transform_candidate`, softmax in `TextEnergy` | `normalize_initial_condition` | softmax of the noisy logits; the thinking variable (section 4.1) |
| `EnergyModel.after_step`, `TextEnergy.norm_pred` | `norm_pred` | RMSNorm over the vocabulary after each step (section 4.1) |
| `FFN_MULTIPLIER = 4` in `src/ebt/models/text_transformer.py` | `ffn_dim_multiplier` | SwiGLU hidden width of the text models (D11) |

## 16. Regression task (question G2)

This task has no counterpart in the paper. It tests whether an EBT can represent an input with
several correct answers (questions.md, G2). Implementation: `src/ebt/tasks/regression.py`,
`src/ebt/models/mlp.py`, `src/ebt/experiments/regression_g2.py`.

### 16.1 Data

- x sampled uniformly in [0.25, 4) and y = s·√x, where s = +1 or −1 with probability ½ each,
  drawn independently of x and of everything else.
- The range keeps x away from 0, where the two branches meet. At x = 0.25 the branches are 1.0
  apart.
- Splits for the full tier: 8 192 training, 2 048 validation and 4 096 test examples. The fast tier
  uses 512, 256 and 256.
- An out-of-distribution split with x sampled uniformly in [4, 9) is generated for question G5
  (section 20) and not evaluated for G2.
- Every split comes from its own generator derived from one data seed (0). The data are the same
  for every model and every training seed. Confidence intervals over seeds therefore cover
  training randomness (initialization, batch order, initial candidates and thinking noise), not
  data sampling.

### 16.2 Candidate and models

- The candidate ŷ is a real number, updated as in section 5.1 with no candidate transform.
- Initial candidate: ŷ_0 ~ N(0, s²), with s the standard deviation of the training targets
  (1.46 for the full-tier data). ŷ_0 is independent of y. The authors' scale of 1 applies to
  logits (section 4.1); here the scale is matched to the targets instead.
- Energy: `MLPEnergy`, E(x, ŷ) = MLP([x; ŷ]) with hidden layers of widths (64, 64), SiLU, and
  a bias-free scalar output (section 3.1). Trainable parameters: (2·64 + 64) + (64·64 + 64) + 64
  = 4416. α is fixed in S2 and not counted.
- Baseline: `MLPRegressor`, ŷ = MLP(x), with hidden layers of widths (65, 64), SiLU, and an
  output bias. Trainable parameters: (65 + 65) + (65·64 + 64) + (64 + 1) = 4419, within 0.1% of
  the energy. The first layer is one unit wider because the regressor has one input less.
- SiLU instead of ReLU. With ReLU the energy is piecewise linear in ŷ, so its second derivative
  with respect to ŷ is zero almost everywhere. The mixed derivative ∂(∇_ŷ E)/∂θ that S2 trains
  through does not vanish, so S2 would still train. The curvature in ŷ, however, matters when
  the loss is backpropagated through several steps (S1, full unroll). SiLU gives an energy that
  is smooth in ŷ.

### 16.3 Models compared

| Model | Name in results | Training objective |
|---|---|---|
| (a) | `regressor` | MSE of ŷ = MLP(x) |
| (b) | `ebt_mse` | S2 thinking (section 6.2), MSE of the final candidate; the paper's objective (§3.2) |
| (c1) | `ebt_contrastive` | S2 thinking, contrastive loss alone (`reconstruction_weight` 0, `contrastive_weight` 1) |
| (c2) | `ebt_combined` | S2 thinking, MSE of the final candidate plus `contrastive_weight` × contrastive loss |
| (c3) | `ebt_regularized` | S2 thinking, contrastive loss plus λ · mean(E_real² + E_fake²) (`energy_regularization_weight` λ); section 16.4 |
| (a2) | `mdn` | mixture density network, negative log-likelihood; outside the EBT paper, section 16.8 |
| (c4) | `ibc` | implicit behavioural cloning energy, InfoNCE; outside the EBT paper, section 16.8 |
| (c5) | `ired` | IRED energy over 10 landscapes, denoising plus contrastive loss; outside the EBT paper, section 16.8 |

### 16.4 Contrastive loss

Upstream form (`model/nlp/ebt.py: EBT_NLP.calculate_contrastive_loss`, lines 272–319):

- The real energy is the energy of the context with the true next token, given as a one-hot
  distribution, on the last energy landscape.
- The fake energy is the last entry of the per-step energies: the energy at the input of the
  final thinking step, with its graph.
- The loss is the cross-entropy of the logits (−E_real, −E_fake) with the real sample as the
  correct class, averaged over non-padding positions.
- The total loss is `reconstruction_coeff` × reconstruction loss + `contrastive_loss_coeff` ×
  contrastive loss.
- The flag that enables it cites IRED ([IRED]) as the source of the idea
  (`train_model.py`, `--contrastive_loss`). The help text of its coefficient (default 0.0005)
  reads "coefficient for contrastive loss, didnt work well not used"
  (`train_model.py`, `--contrastive_loss_coeff`). The paper reports no experiment with it.

This project (`src/ebt/core/contrastive.py: contrastive_loss`,
`src/ebt/training/objectives.py: EBTObjective`):

- A cross-entropy over two logits with the first as the correct class equals
  softplus(E_real − E_fake). The loss is therefore L_c = mean over the batch of
  softplus(E(x, y) − E(x, ŷ_{k−1} + η_{k−1})).
- E(x, y) is evaluated at the true target y, which replaces the one-hot true token.
- The fake energy is `Trajectory.energies[-1]`, the energy at the input of the final step. In
  training it keeps its graph, because the final step always keeps a second-order graph
  (section 6.2).
- The total loss is `reconstruction_weight` × trajectory loss + `contrastive_weight` × L_c
  (`ObjectiveConfig`).
- The loss lowers the energy at the data, which lies on both branches, and raises it where the
  model's own thinking arrives.

Energy-magnitude penalty, model (c3). This is a deviation from the EBT paper and its code,
neither of which uses it.

- Why it was added: (c1) diverged. (c3) was added on 19 November 2026, after the first G2 results
  of (a), (b), (c1) and (c2) had been seen ([changelog](../CHANGELOG.md#v10)). In the full-tier
  search, every one of the 18 (c1) configurations ended with predictions far from the data. The
  mean |ŷ_final − ŷ_0| on validation data was 12–23 at α = 0.1 and 57–80 at α = 1. Over 5 seeds,
  the test MSE is 7637.797 ± 4416.252 (results/regression/g2/full/table.md), against the
  mean-predictor floor of 2.125.
- What its energy looks like: the seed-0 landscape (the energy grid of
  `ebt_contrastive-seed0.json`) has wells on the branches, with the column minima at a median
  |ŷ| of 1.45 (√x at mid-range). Its curvature at the minima, by second differences on the grid
  (step 0.05 in ŷ), has a median of about 870, while gradient descent is stable only for
  curvature below 2/α = 2.
- Explanation: softplus(E_real − E_fake) keeps falling as the energy gap grows. Nothing bounds
  the energy scale, so the gradients grow until a step of fixed size overshoots the wells.
- The remedy: Du and Mordatch [Du-Mordatch], section 3.3 and Algorithm 1, add α_L2 · (E(x⁺)² +
  E(x⁻)²) to the training objective. They report that without it the energies "would fluctuate to
  numerically unstable values" while their difference was preserved. Model (c3) adds λ ·
  mean(E_real² + E_fake²) to L_c (`src/ebt/core/contrastive.py: energy_magnitude`). The penalty has
  the same form as the paper's; λ plays the role of their α_L2, renamed to avoid a clash with the
  step size.
- Differences from Du and Mordatch:
  - their contrastive term is the difference E(x⁺) − E(x⁻), where (c3) keeps the upstream
    softplus;
  - they also apply spectral normalization to the energy network, which (c3) does not;
  - their negatives come from Langevin dynamics with gradients stopped, where (c3) takes the
    energy at the input of the final thinking step, as in (c1).

### 16.5 Derivation: what the MSE objective can learn

Derivation, not a measured result.

Conditional mean. Let f(x, ŷ_0, η) be the final candidate, for any energy, step size and number
of steps, where η collects the thinking noise. The MSE objective is E[(f(x, ŷ_0, η) − y)²]. For
fixed (x, ŷ_0, η), the value of f that minimizes E[(f − y)² | x, ŷ_0, η] is the conditional mean
E[y | x, ŷ_0, η]. The sign s is independent of (x, ŷ_0, η), so this mean equals E[y | x] =
E[s]·√x = 0. A model that minimizes the MSE objective therefore predicts 0 for every x and every
initial candidate, exactly as an MSE regressor does.

The derivation constrains only where the thinking steps end. That the energy of such a model has
a single minimum near 0, rather than wells on the branches, is a hypothesis (questions.md, G2).

Mean-predictor floor. Predicting 0 has mean squared error E[y²] = E[s²·x] = E[x] = (0.25 + 4)/2 =
2.125 for x uniform in [0.25, 4). A model that learned the conditional mean reaches this value. No
predictor can do better on average, because any other value f has
E[(f − y)² | x] = f² + x ≥ x.

Only an objective that shapes the energy around the data, such as the contrastive loss in
section 16.4, can make thinking end on a branch.

### 16.6 Hyperparameter rule

Written before any G2 measurement and applied as written (`regression_g2.py: search_space`,
`selection_objective`).

Fixed at the authors' values:

- AdamW with β = (0.9, 0.999), weight decay 0.01, gradient-norm clip 1, linear warm-up followed
  by cosine decay to the learning rate / 10 (Table D.2);
- mode S2 (section 11) with `steps_min` 2 and `steps_max` 4, the text S2 range {2, 3, 4}
  (section 15, D9);
- step-size jitter r = 2 (Table D.4);
- replay capacity 192 and fraction 0.5 (code defaults, U2);
- inference with 4 steps, which is R + 1, the count the upstream code evaluates with
  (section 8.2), and inference noise 0 (section 7).

Fixed with a deviation from the authors, each with its reason:

- Energy-scaled step size off: upstream implements it for text only (section 10).
- Initial-candidate scale s = standard deviation of the training targets (section 16.2).
- Full tier: batch size 256, 20 000 training steps and 1000 warm-up steps. Upstream warms up for
  1e4 of 1e6 steps; this run is 50 times shorter, and the search must fit on one machine.
- Fast tier: batch size 64, 200 steps, 20 warm-up steps. It exists for tests and CI only.

Tuned on validation data. α, σ and the learning rate depend on the scale of the candidate and
of the energy. The upstream values (α = 0.5 or 5, σ = 3) were set for logits, so they do not
transfer.

| Model | Grid | Configurations |
|---|---|---|
| (a) | learning rate ∈ {1e-3, 3e-3, 1e-2} | 3 |
| (b), (c1) | learning rate ∈ {1e-3, 3e-3} × α ∈ {0.1, 0.3, 1.0} × σ ∈ {0, 0.1, 0.3} | 18 each |
| (c2) | the (b) grid × `contrastive_weight` ∈ {0.0005, 0.05, 1}, where 0.0005 is the upstream default | 54 |
| (c3) | the (b) grid × λ ∈ {0.01, 0.1, 1} | 54 |

The (c3) row was added before any (c3) measurement (section 16.4). The search for (c3) would have
been cut to λ = 0.1 alone (18 configurations) had it been projected to exceed 3 hours. The (c1) run
times, observed but not recorded in a file of this repository, projected less than 3 hours for 54
configurations, so the full grid is used.

Procedure:

1. Every configuration is trained once with the search seed 100, which is not among the
   reported seeds 0–4.
2. Each model is selected by its own training objective on the validation split, evaluated on
   the inference trajectory:
   - (a) and (b): MSE;
   - (c1): the contrastive loss, with the energy at the input of the final inference step as
     the fake energy;
   - (c2): MSE + `contrastive_weight` × contrastive loss;
   - (c3): contrastive loss + λ · mean(E_real² + E_fake²), with the same fake energy as (c1).
3. No G2 metric is used for selection. Selecting on the reported metric would be selection on
   the outcome. It could also favour an EBT whose thinking barely moves from its random initial
   candidate, because a random candidate lies closer to a branch on average than 0 does.
4. The chosen configuration is trained with seeds 0–4 and evaluated on the test split, which no
   earlier step uses.
5. `results/regression/g2/full/search.json` records every configuration tried, the number used
   for selection, the name of that number, and the G2 validation metrics, for information only.

Ties. The rule that was applied to the selections above, read from the code that made them
(Python's `min` over the configurations in grid order), and the rule for every selection in
this project (`src/ebt/experiments/runner.py: choose`):

- the selection values are compared exactly, as the floats that were computed, with no
  rounding;
- among exactly equal values, the first configuration in grid order is chosen. Grid order is
  the order of the table above, read as nested loops with the learning rate outermost, then α,
  then σ, then the last column;
- an undefined value ranks after every defined value. A value is undefined if it is NaN or
  infinite, or if the run stopped because the thinking gradient became non-finite. The search
  then goes on; the configuration is recorded with selection value null and the reason in
  `selection_value_undefined`.

Seeds. Every random source of a run has its own seed, from one function per experiment
(`regression_g2.py: run_seeds`, `runtime/seeding.py: run_seed`). Source number i of run seed s
(s from 0 to 999) gets 1 000 000 + 1000 i + s. The sources, in order: torch's global generator
(parameter initialization), batch order, thinking noise and the training randomness of each
objective, replay sampling, evaluation (initial candidates and DFO samples) and the
mixture-density readout. The fixed seeds lie below 1 000 000 and are kept unchanged: the data
splits use seeds 0 to 3 (section 16.1), and the selection counter-examples use 20 000. Across the
data seeds, the fixed seed, the search seed 100 and the final seeds 0–4, no two sources share a
seed (`tests/test_seeds.py`). Each result file records its seeds in `config.run_seeds`.
The evaluations on saved checkpoints of section 19 use their own range: source number i of
the checkpoint of run s gets 2 000 000 + 1000 i + s, above every training and fixed seed
(`inference_g3_g4.py: evaluation_seeds`, `tests/test_inference_g3_g4.py`).

The final runs of all eight models use these seeds. The search uses the seeds of the scheme of
the code it was run with, with s = 100: global generator 100, batch order 200, thinking noise
201, replay sampling 202, evaluation 10 100 and readout 30 100. These seeds do not coincide with
the data seeds or the counter-example seed. Because the search was run with code of another
fingerprint, its files are reused as stale files, and every final result lists them in
`code.reused_stale_files`. Under the scheme of the search, final runs share seeds: the global
generator is seeded with s, the replay seed 2s + 2 equals the batch seed of run s + 1, and runs 0
and 1 use the data seeds 0 to 3 for batch order and noise; final runs under that scheme are not
reused.

Checkpoints. Every final run saves its trained parameters to
`checkpoints/regression_g2/<tier>/<model>-seed<s>.pt` (not in the repository): the state of
the trained module, which for an EBT includes the thinking loop and its α, together with the
code fingerprint and the run's model, tier, seed and config (`checkpoints.py`).
`regression_g2.py: restore` rebuilds the model from that config and loads the parameters. A
reloaded model gives the same predictions on the CPU as the trained one
(`tests/test_checkpoints.py`).

Code fingerprint. Every search entry, every model's choice in search.json and every result
file records the SHA-256 of the source files that compute it (`fingerprint.py`): the
experiment's module and every `ebt` module it imports, directly or indirectly, with the
`__init__.py` of each package on the way. A search records the current fingerprint only for the
models it searches. An entry or a choice with another fingerprint or none is stale, including
an entry that exists only in a search.json `tried` list. The runner stops before any work if
the stage would use a stale entry or choice, and lists them. With `--reuse-stale` it goes on and
records, in each per-seed result file it writes, the stale entries and choice of its model
(`code.reused_stale_files`; an entry or choice held only in search.json is listed as
`search.json#models/<model>/tried/<configuration>` or `search.json#models/<model>/chosen`). The
v1.0 final runs (`--stage final`) record the reused files there only; no committed search.json
lists them.

No two values in the existing search were exactly equal. The closest pair was in (c3):
0.0906060717 (chosen: learning rate 3e-3, α 1, σ 0.1, λ 0.01) and 0.0906061608 (σ 0, otherwise
the same), a difference of 9e-8. Six (c3) configurations lie within 1e-4 of the chosen value,
with validation r from 0.81 to 2.87 (search.json). Differences this small decided which (c3)
configuration the results describe. It is reported, not changed: the selections were not
rerun.

### 16.7 Metrics

Computed on the test split (`regression.py: regression_metrics`):

- Normalized branch distance r = min(|ŷ − √x|, |ŷ + √x|) / √x, reported as its mean. r = 0 on
  a branch, r = 1 at the conditional mean 0.
- Fraction on a branch: r ≤ 0.1, i.e. within 5% of the branch separation 2√x.
- Branch split: the fraction of predictions above 0.
- For an EBT, dependence on the initial candidate: P(sign ŷ_final = sign ŷ_0). 0.5 means the
  branch is independent of ŷ_0.
- For an EBT, mean |ŷ_final − ŷ_0|, which shows whether thinking moves the candidate.
- Test MSE, compared with the mean-predictor floor 2.125 (section 16.5) and with the split's
  own mean of y².

Diagnostic for the contrastive EBTs, on seed 0: the mean |∂E/∂ŷ| over the test split, at the
targets (on the branches) and at ŷ = 0, together with α
(`regression.py: energy_gradient_magnitude`). α times this value is the size of one thinking
step at those points, which can be compared with the branch separation 2√x.

Each metric is reported as the mean over the 5 seeds with a 95% t-interval
(`src/ebt/statistics.py`).

Figures (seed 0; [reproducing.md, Figures](reproducing.md#figures)):
`results/regression/g2/full/predictions.png`, the test data and the predictions of every model;
`energy_ebt_mse.png`, `energy_ebt_contrastive.png`, `energy_ebt_combined.png` and
`energy_ebt_regularized.png` in the same directory, E(x, ŷ) minus its minimum over ŷ for (b),
(c1), (c2) and (c3).

### 16.8 Models from outside the EBT paper: (a2), (c4) and (c5)

Three further models answer G2 with the methods of their own sources. None of them is part of the
EBT paper or its code. They were added to G2 on 19 November 2026, together with (c3), so that G2
compares the main ways of obtaining multimodal predictions rather than the EBT against one
regressor ([changelog](../CHANGELOG.md#v10)). Each is
implemented from its paper and, where named, its code, both read for this project and described
here in our own words; no code is copied.

Common rules, the same as for models (a) to (c3):

- the data, splits, seeds and metrics of sections 16.1 and 16.7;
- about 4416 trainable parameters;
- the common training budget: 20 000 steps, batch size 256, AdamW with β = (0.9, 0.999) and
  weight decay 0.01, gradient-norm clip 1, 1000 warm-up steps and cosine decay to the learning
  rate / 10, with the learning rate searched. Each source's own batch size and schedule are
  listed below as deviations. The common budget keeps data and compute equal across the G2
  models;
- selection by each model's own training objective on validation data, with the tie rule of
  section 16.6, search seed 100, reported seeds 0–4.

#### (a2) Mixture density network

Source: [Bishop], section 3 (`src/ebt/models/mdn.py`).

- The network maps x to the parameters of a mixture of m Gaussian kernels in y:
  p(y | x) = Σ_i α_i(x) φ_i(y | x) (Bishop's eq. 22 and 23). The mixing coefficients are a
  softmax of m outputs (eq. 25), the widths σ_i the exponential of m outputs (eq. 26), and the
  centres μ_i are m outputs directly (eq. 27).
- Loss: the negative log-likelihood −ln Σ_i α_i φ_i(y | x), averaged over the batch (eq. 28
  and 29).
- Readouts. Primary, used in the G2 table: the centre of the kernel with the largest α_i, the
  "most probable branch" (eq. 49). Bishop recommends it for multi-valued mappings with a finite
  number of distinct branches. Also reported: the centre of the kernel with the largest
  α_i / σ_i (eq. 48, with c = 1), and the centre of a kernel drawn with probability α_i, which
  shows whether both branches carry weight.
- Network: tanh hidden layers, as in Bishop. The hidden widths depend on m so that the
  parameter count stays at the budget: m = 2: (49, 77), 4416 parameters; m = 5: (71, 49), 4420;
  m = 10: (47, 55), 4414.
- Deviations from the source: two hidden layers instead of Bishop's one, to match the other G2
  models; AdamW with the common schedule instead of BFGS.
- Number of kernels: Bishop uses "the minimum number needed" for the branches and calls the
  choice part of model-order selection (sections 5 and 7). Here m ∈ {2, 5, 10} is searched: 2
  is the minimum for two branches and 10 is the number IBC used for its MDN baseline.
- Selection objective: validation negative log-likelihood.
- P(sign ŷ = sign ŷ_0) and mean |ŷ − ŷ_0| are undefined: the model has no initial candidate.
- Plot: −ln p(y | x) on the grid of the energy plots.
- Inference cost: one forward pass per example.

#### (c4) Implicit behavioural cloning energy

Source: [IBC] and the authors' code `google-research/ibc` (Apache-2.0)
(`src/ebt/core/contrastive.py: info_nce`, `src/ebt/training/objectives.py: InfoNCEObjective`,
`src/ebt/inference/dfo.py`).

- Loss (§2): InfoNCE. For each example, N counter-examples ỹ_j are drawn, and the loss is
  −log(e^{−E(x, y)} / (e^{−E(x, y)} + Σ_j e^{−E(x, ỹ_j)})), the cross-entropy of the logits −E
  with the true sample as the correct class (the code's `ibc/losses/ebm_loss.py: info_nce`,
  temperature 1). With N = 1 this equals the contrastive loss of section 16.4.
- Counter-examples: uniform in [y_min, y_max] (Appendix B.1). The code's configuration for
  derivative-free inference widens the bounds of the training targets by 5% of their range on
  each side (`uniform_boundary_buffer` 0.05); this project does the same. N = 256, the code's
  default.
- Inputs and targets are standardized to zero mean and unit variance with training-split
  statistics (Appendix B.1). Predictions are mapped back to the original units before the
  metrics are computed.
- Inference: Algorithm 1, the derivative-free optimizer, with 16 384 samples, 3 iterations,
  σ_init = 0.33 and shrink factor K = 0.5 (Appendix B.1). Candidates start uniform in the
  bounds. At every iteration but the last they are resampled with replacement in proportion to
  softmax(−E), perturbed by N(0, σ²), clipped to the bounds, and σ is multiplied by K. The
  candidate with the highest probability at the last iteration is returned. Gradient-based
  thinking is not used for this model; the paper's section 3 experiments on multi-valued
  functions also use this optimizer.
- Disagreement between the paper and the code, resolved in favour of the paper:
  `src/ebt/inference/dfo.py` follows Algorithm 1 in pairing the probabilities p̃ with the
  samples on which they were computed, and returns the sample with the highest p̃ of the last
  iteration. In the code (`ibc/agents/mcmc.py: iterative_dfo`) every iteration computes the
  probabilities on its input samples and then resamples, the last iteration included; the
  function returns the probabilities of the previous set together with the resampled set, and
  `ibc/agents/ibc_policy.py` pairs them in one categorical distribution, so each probability
  belongs to a different sample than the one it is attached to. (With Langevin refinement
  after the optimizer, the code recomputes the probabilities on the final samples; this
  project does not use that refinement.)
- Network: ReLU, as in the paper's 1-D experiments (an MLP 2:512:512 there); here widths
  (64, 64), 4416 parameters. ReLU makes the energy piecewise linear in y, which does not matter
  to an optimizer that uses no gradients.
- Deviations from the source: batch size 256 instead of 512, AdamW with the common schedule
  instead of Adam with an exponential decay of 0.99 every 100 steps, 20 000 instead of 5000
  training steps (section 3 of the paper), smaller network.
- Selection objective: the InfoNCE loss on the validation split with one fixed set of
  counter-examples (seed 20 000), the same for every configuration.
- P(sign ŷ = sign ŷ_0) and mean |ŷ − ŷ_0| are undefined: the optimizer starts from uniform
  samples, not from one candidate.
- Plots: the energy landscape, in the original units, and the gradient diagnostic of section
  16.7 with no step size, since the model takes no gradient steps.
- Cost per example: training, N + 1 = 257 forward and 257 backward passes; inference,
  3 × 16 384 = 49 152 forward passes.

#### (c5) IRED

Source: [IRED] and the authors' code `yilundu/ired_code_release`
(MIT License), files `models.py` and `diffusion_lib/denoising_diffusion_pytorch_1d.py`
(`src/ebt/models/ired.py`, `src/ebt/training/ired.py`, `src/ebt/inference/annealed.py`).

Landscapes. The energy is conditioned on a landscape index k = 0, …, K − 1 with K = 10
(footnote 2 of the paper). Landscape k models the targets noised as √ᾱ_k y + √(1 − ᾱ_k) ε
(Eq. 4). ᾱ follows the cosine schedule with offset s = 0.008 (the code's
`cosine_beta_schedule`). Measured values, k = 0 to 9: 0.9721, 0.8987, 0.7869, 0.6475, 0.4938,
0.3408, 0.2031, 0.0940, 0.0241 and about 0. The gradient step size on landscape k is
λ_k = β_k / √(1 − ᾱ_k) (the code's `opt_step_size`), from 0.167 at k = 0 to 0.999 at k = 9.

Energy (the code's `models.py: EBM`): three swish layers on [x, y]. The index k enters through
a sinusoidal embedding followed by a small MLP, which gives a gain and a bias for layers 2 and
3 (h ← swish(W h · (1 + gain) + bias)). The energy is the squared norm of a final linear map,
‖fc4(h)‖². The code uses width 512 and embedding sizes 128; here width 37 and embedding sizes
8 and 8 give 4437 trainable parameters, 0.5% above the 4416 of the EBT energy. The index is
passed as a last context column, [x, k], so the energy contract of section 3 is unchanged.

Standardization. x and y are standardized to zero mean and unit variance with training-split
statistics, as for (c4). The code clamps predicted clean values to ±2 for continuous tasks; on
the full-tier training split the standard deviation of y is 1.4565 and |y| ≤ 2, so the
standardized targets satisfy |y| ≤ 1.38 and the clamp contains them. Predictions are mapped
back to the original units before the metrics are computed.

Training loss (the code's `p_losses`, continuous branch; Eq. 5 and §3.3 of the paper). For each
example a landscape k is drawn uniformly. With ε, ε' ~ N(0, 1):

1. Denoising: ‖∇_y E(x, √ᾱ_k y + √(1 − ᾱ_k) ε; k) − ε‖², the gradient of the energy
   supervised as the noise. Its parameter gradient is second order.
2. Negative: start from √ᾱ_k y + √(1 − ᾱ_k) · 3ε', take 2 accept-only-lower gradient steps
   (below) on landscape k without a graph, divide by √ᾱ_k, clamp to ±2 and noise again with
   the same ε'.
3. Contrastive: the two-class cross-entropy between the positive √ᾱ_k y + √(1 − ᾱ_k) ε' and
   the negative, with the positive as the correct class, i.e. softplus(E⁺ − E⁻). The paper
   states that both use "the same sampled noise value".
4. Total: denoising + 0.5 · contrastive (the code's `loss_scale` for continuous tasks).

Accept-only-lower step (the code's `opt_step`): ŷ' = ŷ − λ ∇_y E(x, ŷ; k), clamped to
±2√ᾱ_k; ŷ' replaces ŷ only where E(x, ŷ'; k) ≤ E(x, ŷ; k). In the code, the factor that
multiplies the step is 1 for the first step and is then overwritten by the continuous clamp
factor 2, so every later step on the same landscape uses 2λ_k. This project does the same
(`annealed.py: LATER_STEP_FACTOR`). Every step is one call of `ThinkingLoop.infer` with one
step and a fixed α; the clamp and the acceptance test are applied outside the loop, which is
not changed.

Inference (the code's `p_sample_loop`, which produced the paper's results). Start from
ŷ ~ N(0, 1) in standardized units. For k from 9 down to 0:

1. Jump: predict the clean value ŷ_0 = (ŷ − √(1 − ᾱ_k) ∇_y E(x, ŷ; k)) / √ᾱ_k (the gradient
   read as the predicted noise), clamp it to ±2 and set ŷ = √ᾱ_k ŷ_0. This is one thinking
   step with α = √(1 − ᾱ_k) followed by a rescaling.
2. Descend: 5 accept-only-lower steps on landscape k.
3. Clamp ŷ to ±2√ᾱ_k and, unless k = 0, rescale it to the next landscape:
   ŷ ← ŷ · √ᾱ_{k−1} / √ᾱ_k.

The prediction is ŷ after landscape 0. Like the code, it is not divided by √ᾱ_0 = 0.986.

Disagreement between paper and code. Algorithm 2 of the paper has no jump step and no clamps;
it takes gradient steps with acceptance on each landscape and rescales between landscapes. The
code adds the jump of step 1 and the clamps. This project follows the code.

Deviations from the code:

- The clamp bound of the training negatives is computed for each example's own landscape. The
  code takes the bound of the first example of the batch for all examples.
- No exponential moving average of the weights. The code evaluates an averaged copy of the
  model; this project evaluates the trained weights, as for every other G2 model.
- The common training budget instead of the code's batch size 2048 and Adam at learning rate
  1e-4 (about 50 000 iterations according to the paper's appendix).

Selection objective: the IRED training loss on the validation split, with one fixed draw of
landscapes and noise (seed 20 000), the same for every configuration.

Metrics: all G2 metrics, including P(sign ŷ = sign ŷ_0) and mean |ŷ − ŷ_0|, with ŷ_0 the
initial sample mapped to the original units. Plots: the energy landscape on k = 0 and on k = 9,
in the original units. Gradient diagnostic on k = 0, in the original units of y, with step size
σ_y² λ_0: one step of λ_0 in standardized units moves y by σ_y² λ_0 |∂E/∂y|.

Cost per example: training, 7 forward, 5 first-order backward and 1 second-order pass;
inference, 110 forward and 60 backward passes (per landscape 1 + 2 · 5 forward and 1 + 5
backward).

#### Hyperparameter rule for (a2), (c4) and (c5)

Written before any full-tier measurement of these models.

| Model | Grid | Configurations | Fixed at the source's values |
|---|---|---|---|
| (a2) | learning rate ∈ {1e-3, 3e-3, 1e-2} × m ∈ {2, 5, 10} | 9 | tanh |
| (c4) | learning rate ∈ {3e-4, 1e-3, 3e-3} | 3 | N = 256, 5% bound buffer, DFO settings, ReLU |
| (c5) | learning rate ∈ {1e-4, 3e-4, 1e-3} | 3 | K = 10, 5 descent steps, 2 negative steps, weight 0.5, clamp 2 |

Grid order for ties: learning rate outermost, then m.

#### Measured: one branch for each x

(a2), (c4) and (c5) place their test predictions on a branch: the fraction with r ≤ 0.1 is
1.000 on every seed. Each returns one prediction per x, and that prediction lies on a single
branch. For (a2), readout eq. 49, it is the centre of the kernel with the largest α_i, a function
of x alone. The fraction on the positive branch per seed, and the x at which (a2) changes branch on
seed 0, are in [questions.md, G2](questions.md#g2-why-use-an-energy).

Both-branch coverage, measured on the final checkpoints (question G4, section 19.4;
`results/regression/g3_g4/full/summary.json`): the fraction of the 4 096 test x whose first N
candidates include one with r ≤ 0.1 on each branch, mean over seeds 0–4 ± half-width of the 95%
t-interval:

| Model | N = 1 | N = 2 | N = 4 | N = 8 | N = 16 | N = 32 |
|---|---|---|---|---|---|---|
| (b), (c2), (c3) | 0 | 0 | 0 | 0 | 0 | 0 |
| (c1) | 0 | 0 | 0 | 0.00005 ± 0.00014 | 0.00010 ± 0.00017 | 0.00029 ± 0.00040 |
| (c5) | 0 | 0.00020 ± 0.00054 | 0.00039 ± 0.00108 | 0.00088 ± 0.00244 | 0.00117 ± 0.00325 | 0.00122 ± 0.00339 |
| (c4) | 0 | 0.01313 ± 0.01066 | 0.02598 ± 0.01947 | 0.04102 ± 0.03087 | 0.06338 ± 0.04317 | 0.08813 ± 0.05304 |
| (a2) | 0 | 0.50186 ± 0.01338 | 0.87480 ± 0.00315 | 0.99355 ± 0.00108 | 0.99990 ± 0.00027 | 1 |

N = 1 is 0 by definition: one candidate cannot lie on both branches.

Figures (seed 0; [reproducing.md, Figures](reproducing.md#figures)):
`results/regression/g2/full/energy_ibc.png` for (c4), `energy_ired.png` and `energy_ired_k9.png`
for (c5) on landscapes k = 0 and k = 9, and `density_mdn.png`, −ln p(ŷ | x) for (a2).

## 17. Image denoising task (MNIST)

The image task of the paper (§4.3) at a size that runs on one machine. Implementation:
`src/ebt/tasks/mnist.py`, `src/ebt/models/patch_transformer.py`, `src/ebt/models/attention.py`,
`src/ebt/experiments/mnist_denoising.py`.

### 17.1 Data

- MNIST, from the original gzip-compressed IDX files (`train-images-idx3-ubyte.gz`,
  `t10k-images-idx3-ubyte.gz`), read without any image library. The files are not in the
  repository; `load_images` checks their SHA-256 (listed in `mnist.py`). Their MD5 checksums
  match those published with torchvision's MNIST dataset class. Expected location:
  `data/mnist/raw/`.
- Splits: 55 000 training and 5 000 validation images, from one fixed permutation (data seed 0)
  of the 60 000 official training images; the 10 000 official test images.
- One channel, 28 × 28 pixels. The paper's image experiments use three RGB channels (section
  4.2); nothing else in the method depends on the number of channels.
- Pixels are scaled to [−1, 1] by (p / 255 − 0.5) / 0.5, as the authors' data pipeline does
  (`base_model_trainer.py`, lines 105–111).

### 17.2 Noise

The variance-preserving DiT linear schedule of section 4.2: noise level f builds a schedule of
int(100 / f) timesteps with β from 1e-4 · 1000 / T to 2e-2 · 1000 / T, and the noisy image is
√ᾱ x + √(1 − ᾱ) ε with ᾱ the product of 1 − β over the first 100 steps
(`mnist.py: alpha_bar`, `add_noise`).

| Level | Timesteps | ᾱ | √ᾱ | √(1 − ᾱ) |
|---|---|---|---|---|
| 0.1, training and evaluation | 1000 | 0.89702 | 0.9471 | 0.3209 |
| 0.2, out of distribution (question G5, section 20) | 500 | 0.65972 | 0.8122 | 0.5833 |

Training draws fresh noise for every batch, as upstream does. Validation and test images are
noised once with fixed generators (seeds 40 000 and 40 001), so every model and seed sees the
same noisy images.

### 17.3 Models

Energy (`PatchEnergy`), following the authors' image EBT (section 3.2):

- convolutional patch embedding with patch size 4, giving 7 × 7 = 49 patches (the authors'
  128-pixel images with patch size 16 give 64);
- fixed 2-D sine-cosine positions, half of the channels for the column and half for the row;
- 4 blocks of width 128 with 4 attention heads. Each block applies attention and an MLP
  (hidden width 128, `mlp_ratio` 1 as in the authors' image scripts, GELU with the tanh
  approximation), each after a layer norm without affine parameters, modulated by shift,
  scale and residual-scale vectors computed from the condition vector (adaptive layer norm);
- one learned condition vector, the same for every image, initialized from N(0, 1) like the
  authors' one-entry embedding (`learned_adaln_condition`);
- a final modulated layer norm and a bias-free linear map to one scalar per patch; the energy
  is the mean over patches;
- initialization as upstream: Xavier-uniform weights and zero biases for the linear maps and
  the patch embedding, zero for every modulation layer and for the head. The initial energy is
  therefore zero for every image.

Trainable parameters: 828 032, plus the learnable α (828 033).

Attention is computed explicitly as softmax(QKᵀ / √d) V (`attention.py`). PyTorch's fused
attention has no second derivative on the CPU (measured on torch 2.15: "derivative for
aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented"), and S1 training
differentiates the energy's gradient. The authors force the plain-math attention kernel for the
same reason. Measured before the runs: with explicit attention, second-order gradients on MPS
agree with the CPU within the float32 tolerances of `docs/numerics.md` for S1 (1 and 2 steps),
S2 and full unroll. The differences were observed but are not recorded in a file of this
repository.

Baseline (`PatchDenoiser`): the same backbone with a head that maps each patch to its 16 pixels,
with bias, reassembled into the image. It predicts the clean image from the noisy one in one
forward pass. Trainable parameters: 829 968, 0.23% more than the energy.

### 17.4 Thinking and training

Fixed at the authors' values for image denoising (`job_scripts/img/pretrain_denoising/ebt_s1.sh`,
`job_scripts/img/inference/ebt_denoising.sh`):

- mode S1 with 1 training step; 2 inference steps;
- thinking starts from the noisy image (section 4.2);
- learnable α in its own optimizer group, learning-rate multiplier 3 × the initial α
  (§I.2; the authors' 3000 and 9000);
- Smooth L1 loss with β = 1 between the denoised and the clean image;
- no Langevin noise, step-size jitter, random step count or replay buffer;
- AdamW with β = (0.9, 0.999), weight decay 0.01, gradient-norm clip 1, batch size 128, linear
  warm-up for 10% of training (upstream 1e4 of 1e5 steps), cosine decay to the learning
  rate / 10.

Changed for the size of this run: 10 000 training steps (1000 warm-up), and the model size of
section 17.3 instead of the paper's (Table D.1).

Deviation in the learning-rate schedule, found while checking the script after the runs had
started: the authors' script trains for 1e5 steps but sets the cosine schedule's length to 1e6
steps (`max_steps 100000`, `max_scheduling_steps 1000000`; `base_model_trainer.py`, line 354).
Their learning rate therefore ends training at about 98% of its peak. This project decays it to
the learning rate / 10 over its 1e4 steps, as for every other task.

The baseline uses the same optimizer, schedule, batch size, loss and noise.

### 17.5 Hyperparameter rule

Written before any full-tier measurement.

Transfer. The [−1, 1] pixel scale transfers. The energy is a mean over patches, so the gradient
per pixel has a similar scale for 49 and for 64 patches, which makes the authors' α = 3000 a
plausible start. That is weak evidence, so α is searched around it. The authors' learning rate
is set for a larger model and batch, so it is searched too.

| Model | Grid | Configurations |
|---|---|---|
| EBT | learning rate ∈ {1e-4, 3e-4, 1e-3} × initial α ∈ {300, 3000, 30 000} | 9 |
| Baseline | learning rate ∈ {1e-4, 3e-4, 1e-3} | 3 |

Extension. Both selections of this grid lay at its edge: the EBT at learning rate 1e-3 and
α = 30 000 (validation Smooth L1 0.004186), the baseline at learning rate 1e-3 (0.003702).
The grid was therefore extended once, at that edge only, with the same seed, training length
and rule, and the extension is part of the grid:

| Model | Added configurations, in grid order after the first grid |
|---|---|
| EBT | (learning rate, α) = (3e-3, 30 000), (3e-3, 100 000), (1e-3, 100 000) |
| Baseline | learning rate 3e-3 |

Measured validation Smooth L1 of the added configurations: EBT 0.003987, 0.004021 and
0.004153; baseline 0.003566. The selections after the extension are the EBT at learning rate
3e-3 and α = 30 000 (0.003987) and the baseline at learning rate 3e-3 (0.003566), and the
final runs use them. Both again lie at the edge of the extended grid (learning rate 3e-3).
The grid is not extended further: this is recorded as a limit of the search, and a higher
learning rate may give a lower validation loss for either model.

Procedure: as in section 16.6, including the tie rule (grid order: learning rate outermost, then
α, with the extension after the first grid). Each configuration is trained once with seed 100.
Each model is selected by its own objective, the Smooth L1 loss on the 5 000 noisy validation
images: for the EBT after the 2 inference steps that are reported, for the baseline after its
forward pass. The chosen configurations are trained with seeds 0–4 and evaluated on the test
split.

Validation curve: every 500 training steps, the validation Smooth L1 loss and PSNR after 2
inference steps (for the baseline, after its forward pass) are recorded. The results state
whether the curve was still falling at the end of training.

Seeds, checkpoints and fingerprint: as in section 16.6 (`mnist_denoising.py: run_seeds`). The
sources are the global generator (parameter initialization), batch order and training noise,
each with seed 1 000 000 + 1000 i + s for source number i. The fixed seeds are the split
permutation (0) and the noise of the validation and test images (40 000 and 40 001). The final
runs use these seeds. The search uses the seeds of the scheme of the code it was run with, with
s = 100 (global generator 100, batch 200, noise 201), and its files are reused as stale files and
listed in each result. Under the scheme of the search the global generator of run s uses seed s,
the same as the batch seed of run s/2 (s even) or the noise seed of run (s − 1)/2 (s odd), and
run 0's batch seed is the split seed 0; final runs under that scheme are not reused. Checkpoints:
`checkpoints/mnist_denoising/<tier>/<model>-seed<s>.pt`, rebuilt by
`mnist_denoising.py: restore`.

### 17.6 Metrics

- PSNR on the test split: pixels are mapped back by x · 0.5 + 0.5, clamped to [0, 1] and
  multiplied by 255; PSNR = 10 log10(255² / MSE), with the MSE taken over every pixel of every
  test image (`mnist.py: psnr`). Upstream computes PSNR from the MSE of each batch
  (`inference/img/generate_image.py`); how it combines the batches is not shown there. This
  project uses one MSE over the whole test set.
- The same PSNR for the noisy input, as a reference.
- Smooth L1 and MSE in the [−1, 1] scale.
- Cost per example as pass counts (section 6.3): EBT training 1 forward, 1 first-order backward
  and 1 second-order pass; EBT inference 2 forward and 2 backward passes (the final energy is not
  evaluated); baseline training 1 forward and 1 backward pass; baseline inference 1 forward pass.

Each metric is reported as the mean over seeds 0–4 with a 95% t-interval.

Figures ([reproducing.md, Figures](reproducing.md#figures)):
`results/mnist/denoising/full/examples.png`, test images of seed 0 (clean, noisy, and the output
of each model); `results/mnist/denoising/full/validation.png`, the validation PSNR and Smooth L1
against the training step, one line per seed.

## 18. Character-level language model

This task supplies the language-model curves of questions G3 to G5. It compares a causal S2 EBT
with a causal Transformer of the same backbone (`experiments/text_lm.py`).

### 18.1 Data

Training corpus: Tiny Shakespeare [Tiny Shakespeare], the file `data/tinyshakespeare/input.txt` of
Karpathy's char-rnn repository, saved as `data/text/raw/tinyshakespeare.txt`. It has 1 115 394
characters (all ASCII), 65 distinct characters and SHA-256
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed. The text is Shakespeare's plays,
which are public domain by age. The repository that distributes the file has no licence file. The
corpus is used because it is the standard character-level benchmark, so the results can be compared
with other work. It is read from `data/text/raw` and checked by its SHA-256 (`tasks/text.py`); it
is not stored in this repository.

Splits: contiguous blocks. Training takes the first floor(0.9 N) characters and the rest is
halved, giving 1 003 854 / 55 770 / 55 770 characters. Contiguous blocks mean that no training
window overlaps validation or test text. Validation and test use no character that is absent
from the training split.

Out-of-distribution text (question G5): Jane Austen, Pride and Prejudice, Project Gutenberg
eBook 1342. The UTF-8 file is 772 386 bytes with SHA-256
3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75, downloaded on 20 November
2026 from gutenberg.pglaf.org/cache/epub/1342/pg1342.txt. It is public domain in the USA, and
only its body is used. This edition contains the 1894 illustrated title pages, a preface by
George Saintsbury and illustration markers. The evaluated block was fixed by the following rule,
written down before the block was read (`tasks/text.py: out_of_distribution_text`):

1. Decode as UTF-8; replace CRLF by LF.
2. Body: the lines strictly between the line starting "*** START OF THE PROJECT GUTENBERG
   EBOOK" and the line starting "*** END OF THE PROJECT GUTENBERG EBOOK".
3. Novel: the body from the line after the first line equal to "Chapter I.]", the end of the
   illustrated heading of chapter 1. This drops the title pages and the preface.
4. Remove every illustration marker (regular expression `\[Illustration[^\]]*\]`).
5. Fold U+2018 and U+2019 to `'`, U+201C and U+201D to `"`, U+2013 and U+2014 to `-`, and
   U+00A0 to a space.
6. Drop every character outside the 65-character training vocabulary.
7. The evaluated block is the first 55 770 characters (the length of the test split).

Measured: the body has 728 749 characters and the novel 688 551 after step 4. Step 6 drops
4 564 characters (0.66%): 3 502 double quotes (the training text has none), 934 underscores
(italics), 35 `]`, 18 `/`, 18 `*`, 17 `(`, 17 `)`, 14 non-ASCII letters (ê, à, é, œ) and 9
digits. The evaluated block has SHA-256
6d84b7147a15e208e68f4fcc00fa31727da0667d9eedf3550031eb3264ffbf5f (UTF-8). It is reproduced from the
two data files, from the repository root, by

```python
import hashlib
from pathlib import Path
from ebt.tasks.text import load_data

block = load_data(Path("data/text/raw"), 128).ood_text.block
print(hashlib.sha256(block.encode()).hexdigest(), len(block))
```

which prints the hash and 55770. Out-of-distribution BPC is therefore BPC on the folded and
filtered text, and the results state this.

### 18.2 Vocabulary and windows

The vocabulary is the 65 characters of the training split, sorted by code point; a character's
index is its rank. There is no unknown token and no padding. Uniform prediction gives
log2 65 = 6.022 bits per character.

Training rows are every window of S characters with the next character of each position as the
target, one row per start offset. Evaluation uses non-overlapping windows,
floor((N − 1) / S) per split; early positions of a window have a short context, the same for
both models.

### 18.3 Models

Status: this project: matches the code, corrections 13.3 and 13.4.

Both models share the backbone of the authors' text models (§I.1.2, §D.1):

- pre-norm blocks with RMSNorm (ε = 1e-5);
- attention with rotary positions (base 10 000);
- a SwiGLU feed-forward of hidden width 4D, the authors' code default (D11; §D.1.1 says no
  multiplier);
- bias-free linear layers and a final RMSNorm;
- head width 64 (`models/text_transformer.py`);
- initialization as the authors' `init_whole_model_weights`: Xavier-normal weights for every
  linear layer, with the query, key and value maps initialized as three separate D × D blocks
  of the fused projection; N(0, 0.02²) for the token embedding; RMSNorm weights 1. The
  step-index embedding keeps PyTorch's default N(0, 1), as upstream, which never re-initializes
  it (correction 13.3). The image models keep the initialization of section 17.3.

- Baseline: token embedding, the blocks with a causal mask, and a bias-free D → V output layer;
  trained with cross-entropy.
- EBT (sections 3.1, 3.3 and 4.1): one learned step-index embedding, the S observed tokens and
  the candidate of every next position (B × S × V). The step-index token has rotary position
  0, observed token i position i + 1, and the prediction for the character after observed token
  j position j + 2. Every observed token and every prediction attends to the step-index token;
  its output is not passed to the energy head. The thinking variable is p = softmax(ŷ + η)
  over V, and a bias-free V → D map (`vocab_to_embed`) encodes it. A bias-free D → 1 head gives
  one energy per predicted position, and the thinking gradient of their sum is taken with
  respect to p (section 4.1). `norm_pred`, a learnable RMSNorm over V, is applied after every
  step, including the last; its ε is PyTorch's default for `nn.RMSNorm(V)` (eps=None, the
  machine epsilon of the dtype), as upstream. The backbone's RMSNorms use ε = 1e-5.
- The baseline has no step-index token (the authors' Transformer++ has none); its positions
  are 0..S − 1.
- The EBT has 2D + V more parameters than the baseline: D for the step-index embedding, V·D for
  `vocab_to_embed`, D for the energy head and V for `norm_pred`, against V·D for the baseline's
  output layer. With V = 65 and the 4D feed-forward: 128 × 4, EBT 1 066 689 and baseline
  1 066 368; 256 × 6, EBT 6 328 641 and baseline 6 328 064; 384 × 6, EBT 14 211 521 and
  baseline 14 210 688.

Attention layouts (U6, `models/causal_attention.py`):

- full: one sequence of length 2S + 1 (the step-index token, the observed tokens, the
  predictions) with the square mask of §C.4, extended by the step-index token;
- split: the observed stream (the step-index token and the observed tokens) attends causally to
  itself, and each prediction attends to the observed stream's keys 0..j + 1 plus its own key,
  as one extra softmax column (§C.3). There is no square score matrix over both streams.

The two layouts give the same energies, candidate gradients, parameter gradients and
second-order parameter gradients (of the squared candidate gradient, through `create_graph`)
in float64: energies and candidate gradients to 1e-10 relative, parameter gradients to 1e-9
(`tests/test_text_models.py`). This settles U6 for this implementation. The runs use the
layout chosen by the budget rule of section 18.7.

### 18.4 Thinking and training

EBT (S2, section 11):

- Every step is the update of section 4.1: p = softmax(ŷ + η), the gradient of the summed
  energy with respect to p, p − α · exp(E / 9) · ∇_p E, then `norm_pred`.
- No detach between steps, a second-order graph at the final step only, and cross-entropy of
  the final candidate (after `norm_pred`) as logits.
- α fixed (not learnable, Table D.4), energy-scaled by exp(E / 9) (D7), with jitter r = 2 per
  position.
- k uniform in {2, 3, 4} per batch (D9).
- Langevin σ = 3 in training, added before the energy (D4).
- Replay buffer with capacity 192 and fraction 0.5 (U2). A replayed row holds its window, its
  targets and its final candidate after `norm_pred`, as upstream.
- Initial logits from N(0, 1).
- Evaluation: 4 steps without noise, from initial logits drawn with a fixed seed per split, the
  same for every run and checkpoint.

Precision: the authors' text scripts set the float32 matrix-multiplication precision to
"medium" on the GPU; this project computes in strict float32.

Both models: AdamW (0.9, 0.999), weight decay 0.01, gradient-norm clip 1, linear warm-up over
10% of the T steps, then cosine decay to lr / 10. The validation BPC is measured every T / 10
steps.

Best checkpoint (§D.1.2): the test and out-of-distribution BPC are reported at the checkpoint
with the lowest validation BPC. The same rule applies to both models. The parameters of that
checkpoint are kept in memory during training and saved as the run's checkpoint for later
inference experiments. The validation curve is reported. Test data are never used for this
choice.

Seeds, fingerprint and checkpoints: as in section 16.6 (`text_lm.py: run_seeds`). The sources
are parameter initialization, batch order, training noise (initial logits, jitter, Langevin and
step count), replay sampling and generation. The fixed seeds 50 000, 50 001 and 50 002 give the
initial logits of the validation, test and out-of-distribution evaluations.

### 18.5 Hyperparameter rule

Written before any full-tier measurement.

| Model | Grid | Configurations |
|---|---|---|
| Baseline | learning rate ∈ {3e-4, 1e-3, 3e-3, 1e-2} | 4 |
| EBT, stage 1 | α ∈ {0.5, 5, 50} at learning rate 0.0012 (Table D.4) | 3 |
| EBT, stage 2 | learning rate 3e-3 at the α chosen in stage 1 | 1 |

Reasons:

- α: the sources disagree (D8: 0.5 in the S2 script, 5 in Table D.4, 5 to 500 in §I.1.1).
  With V = 65 instead of 50 277, the softmax gradient per logit has another scale, so the
  authors' value may not transfer.
- EBT learning rate: on MNIST both models selected the highest learning rate of their grid
  (section 17.5), so it is searched. The two-stage form keeps the EBT search at 4
  configurations.

Procedure: each configuration is trained once with seed 100. Each model is selected by its own
objective, the validation BPC at its best checkpoint, over all of its configurations (for the
EBT, all four), with the comparison and tie rule of section 16.6. Grid order is stage 1 in
increasing α, then stage 2. The stage 2 configuration exists only once every stage 1
configuration is on disk (`text_lm.py: search_space`). The chosen configurations are trained
with seeds 0–4 and evaluated on the test split and the out-of-distribution block. The search
and the final runs are two separate stages, each within the wall-clock budget of section 18.7.

Measured, full-tier search of 21 November 2026 (`results/text/lm/full/search/ebt/`): the
validation BPC at learning rate 1.2e-3 is 2.1017, 2.1185 and 2.1008 for α = 0.5, 5 and 50
(2.101663604953629, 2.1185162126041734 and 2.1008074758089124 in the search files). Across this
100× range of α the values differ by at most 0.018 BPC, and each comes from one seed (100) per
setting. The chosen α = 50 is the largest value of the grid. The grid was fixed before the
search and is not extended afterwards. Whether thinking changes the result at this scale is
measured by question G3 (BPC against the number of inference steps) on the final checkpoints;
the results are in section 19.5.

The whole search (`results/text/lm/full/search.json`, validation BPC at the best checkpoint,
seed 100):

| Model | Configuration | Validation BPC |
|---|---|---|
| EBT, stage 1 | learning rate 1.2e-3, α = 0.5 | 2.101663604953629 |
| EBT, stage 1 | learning rate 1.2e-3, α = 5 | 2.1185162126041734 |
| EBT, stage 1 | learning rate 1.2e-3, α = 50 | 2.1008074758089124 |
| EBT, stage 2 | learning rate 3e-3, α = 50 | 2.125150615921709 |
| Baseline | learning rate 3e-4 | 2.0759426639189273 |
| Baseline | learning rate 1e-3 | 2.084121522187001 |
| Baseline | learning rate 3e-3 | 2.054657803427174 |
| Baseline | learning rate 1e-2 | 2.2878877142940333 |

Chosen: the EBT with learning rate 1.2e-3 and α = 50; the baseline with learning rate 3e-3.

The baseline's chosen learning rate, 3e-3, has a
tried value on each side (1e-3 and 1e-2), both with a higher validation BPC. The EBT's chosen
learning rate, 1.2e-3, is the lower of its two learning rates, and no lower one was tried. The
EBT's choice of α = 50 over α = 0.5 rests on a difference of 0.0009 BPC (2.1008 against
2.1017), smaller than the 0.018 BPC spread of the three stage 1 values, each from a single
seed. The grid was fixed before the search and is not extended.

Why α can matter little in this model: after every step, p − α · exp(E/9) · ∇_p E passes
through `norm_pred`, an RMSNorm over the V entries with a learned weight w (section 18.3), where
p = softmax(ŷ + η) is the thinking variable, the probabilities of the candidate logits ŷ plus
the Langevin noise η (`src/ebt/core/thinking.py`), and α is drawn from the jitter range. RMSNorm
is invariant to the scale of its input. When α · exp(E/9) · ‖∇_p E‖ is much larger than ‖p‖,
the input of `norm_pred` is dominated by the gradient term, and its output is close to
−w ⊙ g / rms(g) with g = ∇_p E, independent of α and of the jitter. The authors' code has the
same structure (`model/nlp/ebt.py`: `norm_pred` applied after the step). Whether the trained
models are in that regime is measured by G3 on the final checkpoints (section 19.5).

Code of the search and of the final runs. The search was run under the text_lm code
fingerprint f448d4e81408 (f448d4e814083f64dd087bf6676f3e56300bba02c99308d629962c45dc26deb0),
which every search entry and both choices in `search.json` record. The only later change to the
code of that fingerprint is one line of `generate()` in `text_lm.py`: the logits are moved to
the CPU before the conversion to float64 (`logits.cpu().double().softmax(dim=-1)` instead of
`logits.double().softmax(dim=-1).cpu()`), because MPS has no float64. The sampled
characters come from the same CPU generator as before. `generate()` writes the illustration
samples of a final run and is not called by the search (`runner.search` trains each
configuration without test data, and `run` calls `generate()` only with test data). The
fingerprint changes with this line, so the search entries are stale by the rule of
`runner.py`. The final runs reuse them with `--reuse-stale`, and every final result file lists
the reused entries and choice of its model in `code.reused_stale_files`.

### 18.6 Metrics

- Bits per character: mean cross-entropy in nats, divided by ln 2, over every position of the
  non-overlapping windows of a split. For the EBT the logits are the final logits after 4
  inference steps.
- Reported for the test split and the out-of-distribution block at the best checkpoint, as the
  mean over seeds 0–4 with a 95% t-interval; the validation curve per seed.
- Generation samples (illustration, not a metric): 2 × 500 characters per model from seed 0's
  final run, at temperature 1. The prompt is the first 32 characters of the validation split. For
  each new character the window is the last S characters; the EBT thinks 4 steps from N(0, 1)
  logits and samples from its last position.
- Cost as pass counts per window (section 6.3): EBT training k forward, k first-order backward and
  1 second-order pass; EBT inference 4 forward and 4 backward passes; baseline training 1 and 1;
  baseline inference 1 forward pass.
- Training cost with replay. From the second training step on, each EBT batch of B fresh windows
  is extended by B / 2 replayed windows (fraction 0.5; full tier 32 and 16), so the training loss
  and its gradient are taken over 1.5 · B rows, 1.5 times the fresh rows (the first step has an
  empty buffer and B rows). The pass counts are counted over every row of the batch, replayed rows
  included (`EBTObjective.batch_loss`), and the result's `train_examples` counts fresh windows
  only. The training passes per window of the report (`tools/text_report.py`) are their quotient:
  the passes spent per fresh training window, about 1.5 times the per-row counts above ((32 + 48 ·
  (T − 1)) / (32 · T) = 1.49975 rows per fresh window at T = 2 000), with k averaged over the
  steps. The baseline has no replay; its figure is 1 and 1 per window.

### 18.7 Budget rule

Written before the measurement of 21 November 2026 was applied. Item 3 was reworded before that
measurement ([changelog](../CHANGELOG.md#v10)).

1. Layout: the faster of the two attention layouts at equal results. Equal results means that
   the U6 test passes (section 18.3). "Faster" is the EBT training step time at k = 4.
2. Size: the largest of {128 × 4, 256 × 6, 384 × 6} (width × depth; 384 × 6 is the authors'
   xxs size) whose EBT peak MPS memory at k = 4 is at most 75% of the free memory recorded
   before the measurements, and whose training length T ≥ 1 000 fits the time rule below.
3. Context and batch: S = 256 and batch 64 are preferred (Table D.2). The sizes are tried from
   the largest to the smallest. For each size the settings are tried in this order:
   (S 256, batch 64), (S 256, batch 32), (S 128, batch 64), (S 128, batch 32). The first size
   and setting that meet items 2 and 4 are chosen; the next smaller size is tried only after
   all four settings of the larger size have failed.
4. Training length: T is the largest of {1 000, 2 000, 5 000} for which the projected search
   (4 baseline and 4 EBT configurations) takes at most 8 hours and the projected 5 + 5 final
   runs take at most 8 hours. A projection is the measured step time times T, plus 10
   validation evaluations at their measured time; final runs add one test, one
   out-of-distribution evaluation and, for seed 0, the samples. The EBT step time is the mean of
   the measured times at k = 2, 3 and 4. The samples are one call of `text_lm.generate` per
   model: 500 characters, both samples of section 18.6 in one batch of 2 rows. Each model's
   generation is counted once, at 500 times its time per character on 2 rows. The EBT's time
   per character is measured; the baseline's is not, and is counted at the EBT's, an upper
   bound (one forward pass per character against 4 forward and 4 backward passes on the same
   rows).
5. Every departure from Table D.2 (S = 256, batch 64) is recorded here with the measurement that
   forced it.

The budget of item 4 is the wall-clock budget of v1.0: 8 hours per stage (the search; the final
runs) on one Apple M5 Mac with 16 GB of unified memory. Later versions record their own budget
where it applies.

Generation count. `tools/text_budget.py` counts one EBT and one baseline generation for the final
runs, as in item 4, because `text_lm.generate` produces both samples of a model in one batch. A
count of 2 · 500 EBT generation times and none for the baseline, which the tool used before 23
November 2026, gives the same sum with the baseline at the EBT's time, so the projected hours and
the choice of section 18.8 are the same under both counts.

Implementation: `tools/text_budget.py: decide` applies items 1 to 4 to measurement records and
the free memory, and names the next measurement it needs until it can decide
(`tests/test_text_budget.py`). `python -m tools.text_budget OUTPUT_DIR` reads the free memory,
measures each setting the rule asks for in its own process with
`python -m tools.text_step_measurement` (step times, peak memory, validation evaluation and
generation time of one setting), records an out-of-memory process as out of memory, and writes
the records, the decision (choice.json) and the text for section 18.8 to OUTPUT_DIR. A process
that exits with 0 is a measurement. A process with a non-zero exit is out of memory
(`is_out_of_memory`) if its error output contains "out of memory" (case ignored), at any
non-zero exit code; if it was killed by the system (exit 137, signal 9); or if it is a Metal
driver abort (SIGABRT, exit 134 or signal 6) whose error output reports "Insufficient Memory" or
"kIOGPUCommandBufferCallbackErrorOutOfMemory". Any other non-zero exit, including a SIGABRT
without these, is a failure and stops the measurements.

A measurement process that runs longer than 1 800 s is killed with its process group and
recorded as over the time limit; the rule treats it exactly as out of memory for that layout and
setting.

Two further rules of `decide`, for a size and setting:

- if the EBT in the split layout is out of memory or over the time limit, the setting fails
  without measuring the full layout;
- if the baseline is out of memory or over the time limit, the setting fails.

Why a process over 1 800 s cannot fit: `tools/text_step_measurement.py` runs 2 warm-up and 5
timed training steps, 7 in all, once for the baseline and once for each k = 2, 3, 4 for the EBT,
21 in all, then one validation evaluation and, for the EBT, 10 generations. Taking warm-up steps
as long as timed ones and leaving out the time to build the model, a process over 1 800 s spent
more than 900 s either in its n training steps (n = 7 or 21, mean step time m) and its
validation evaluation v, or in its 10 generations. In the first case n · m + v > 900 s. One run
at the smallest T = 1 000 is projected at 1 000 · m + 10 · v, at least 10 · (n · m + v) because
1 000 ≥ 10 · 21, so above 9 000 s, and the 4 runs of that model in the search above 36 000 s =
10 h, over the 8 h budget. In the second case one EBT generation takes more than 90 s, and the
projected final runs include 500 of them (the EBT's samples, item 4), above 45 000 s = 12.5 h,
without the baseline's samples. Either way no T in {1 000, 2 000, 5 000} fits
(`tests/test_text_budget.py` checks this from the tool's counts).

### 18.8 Timings and the chosen setting

Measurements of 20 November 2026, model of that date. Measured on one machine (Apple M5, 16 GB
unified memory, MPS limit 11.84 GiB), with no other training process running. Before the
measurements: `memory_pressure` reported 64% system-wide free memory, that is 10.24 GiB of 16 GiB;
`vm_stat` reported 3.81 GiB free pages and 6.88 GiB free, inactive and speculative pages together.
After the measurements: 61% free. The free memory of the rule is the `memory_pressure` value, 10.24
GiB, so the memory limit is 7.68 GiB.

Each setting ran in its own process on random tokens, with the modules of this repository: the
training step (loss, backward, gradient clip, AdamW) timed as the mean of 5 steps after 2 warm-up
steps; for the EBT the step count k was fixed to 2, 3 and 4 in turn, with replay and the training
noise of section 18.4. Memory is the MPS driver allocation after the timed steps of each k (the
allocator cache is emptied before each k and kept during it, so the value includes the peak). The
validation evaluation is one pass over 55 770 characters in windows of S. Generation is one EBT
inference of 4 steps on a window of 2 rows of length S, per character.
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` held the allocator to the MPS limit, so that memory
exhaustion raises an error instead of swapping. The measurement code is
`tools/text_step_measurement.py`; for these measurements it ran as a separate script that differed
only in having no device argument and four fewer fields in its record. The records the rule uses
are in `tests/data/text_budget_2026-11-20.jsonl`, and `tests/test_text_budget.py` reproduces the
application with item 3 as stated in section 18.7 from them. All 23 records of these measurements
are in `results/text/measurements-2026-11-20/measurements.jsonl`, with the memory readings before
and after (`memory-before.txt`, `memory-after.txt`) and the one setting that ran out of memory with
its error message (`oom.jsonl`); `tests/test_recorded_measurements.py` reproduces from them the
128 × 4 projections and the application of item 3 as worded now. The application with the
wording of 20 November 2026, which chose 128 × 4, is recorded in the
[changelog](../CHANGELOG.md#v10) and reproduced by no test. The parameter column is that of the
model of 20 November 2026, without corrections 13.1 to 13.4 (section 13); the counts of the
current model are in section 18.3.
Settings ran from the smallest upward; the sweep stopped at the first out-of-memory error, 256 × 6
with S 256, batch 64 and the full layout (11.82 GiB allocated). 384 × 6 was therefore not measured.

Layouts: the U6 test (section 18.3) passes, and the split layout is faster in every measured
setting, by a factor of 1.5 to 1.8 at k = 4, with 31% to 51% less memory.

| size | S | batch | model | layout | parameters | step k=2 (s) | step k=3 (s) | step k=4 (s) | step (s) | peak k=4 (GiB) | validation eval (s) | generation (s/char) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 × 4 | 128 | 32 | transformer | - | 476,544 | | | | 0.0225 | 0.21 | 0.10 | |
| 128 × 4 | 128 | 32 | EBT | split | 476,737 | 0.1886 | 0.2379 | 0.2872 | 0.2379 | 1.03 | 1.80 | 0.0211 |
| 128 × 4 | 128 | 32 | EBT | full | 476,737 | 0.3231 | 0.3960 | 0.4735 | 0.3976 | 2.09 | 2.62 | 0.0176 |
| 128 × 4 | 128 | 64 | transformer | - | 476,544 | | | | 0.0461 | 0.36 | 0.11 | |
| 128 × 4 | 128 | 64 | EBT | split | 476,737 | 0.3995 | 0.5038 | 0.6089 | 0.5041 | 2.06 | 1.74 | 0.0177 |
| 128 × 4 | 128 | 64 | EBT | full | 476,737 | 0.6724 | 0.8323 | 0.9985 | 0.8344 | 4.12 | 2.70 | 0.0171 |
| 128 × 4 | 256 | 32 | transformer | - | 476,544 | | | | 0.0522 | 1.08 | 0.12 | |
| 128 × 4 | 256 | 32 | EBT | split | 476,737 | 0.4721 | 0.5970 | 0.7400 | 0.6030 | 3.19 | 2.14 | 0.0293 |
| 128 × 4 | 256 | 32 | EBT | full | 476,737 | 0.8894 | 1.0934 | 1.2947 | 1.0925 | 5.13 | 3.57 | 0.0376 |
| 128 × 4 | 256 | 64 | transformer | - | 476,544 | | | | 0.1116 | 1.05 | 0.12 | |
| 128 × 4 | 256 | 64 | EBT | split | 476,737 | 0.9924 | 1.2565 | 1.5205 | 1.2565 | 5.16 | 2.21 | 0.0247 |
| 128 × 4 | 256 | 64 | EBT | full | 476,737 | 1.8775 | 2.3132 | 2.7401 | 2.3103 | 9.07 | 3.58 | 0.0327 |
| 256 × 6 | 128 | 32 | transformer | - | 2,789,120 | | | | 0.0806 | 1.49 | 0.35 | |
| 256 × 6 | 128 | 32 | EBT | split | 2,789,441 | 0.6936 | 0.8612 | 1.0390 | 0.8646 | 3.32 | 6.11 | 0.0376 |
| 256 × 6 | 128 | 32 | EBT | full | 2,789,441 | 1.1054 | 1.3668 | 1.6345 | 1.3689 | 5.19 | 9.29 | 0.0434 |
| 256 × 6 | 128 | 64 | transformer | - | 2,789,120 | | | | 0.1797 | 1.08 | 0.37 | |
| 256 × 6 | 128 | 64 | EBT | split | 2,789,441 | 1.4270 | 1.7942 | 2.1645 | 1.7953 | 6.32 | 6.12 | 0.0348 |
| 256 × 6 | 128 | 64 | EBT | full | 2,789,441 | 2.2324 | 2.7689 | 3.3146 | 2.7720 | 9.10 | 9.29 | 0.0438 |
| 256 × 6 | 256 | 32 | transformer | - | 2,789,120 | | | | 0.1913 | 1.08 | 0.42 | |
| 256 × 6 | 256 | 32 | EBT | split | 2,789,441 | 1.6416 | 2.0500 | 2.4708 | 2.0542 | 7.26 | 7.37 | 0.0824 |
| 256 × 6 | 256 | 32 | EBT | full | 2,789,441 | 2.8861 | 3.6033 | 4.1866 | 3.5587 | 11.18 | 11.90 | 0.1019 |
| 256 × 6 | 256 | 64 | transformer | - | 2,789,120 | | | | 0.4007 | 3.05 | 0.43 | |
| 256 × 6 | 256 | 64 | EBT | split | 2,789,441 | 5.2751 | 4.3393 | 5.1934 | 4.9359 | 11.73 | 7.69 | 0.0689 |

The column "step (s)" is the EBT step time averaged over k = 2, 3 and 4, the value the rule uses;
for the transformer it is its training step. In the 256 × 6, S 256, batch 64 split row, k = 2 is
slower than k = 4; that setting runs at 11.73 of 11.84 GiB.

The two applications of the rule to these measurements, one with item 3 as worded on 20 November
2026 and one with its new wording, were not used. They are recorded in the
[changelog](../CHANGELOG.md#v10).

The model of 20 November 2026 differs from the current model by corrections 13.1 to 13.4 (section
13). Correction 13.1 takes the thinking gradient with respect to the probabilities instead of
through the softmax; correction 13.2 adds one token, the step-index token, to the observed stream;
correction 13.3 changes initial values only. These three change step time and memory very little.
Correction 13.4 makes the feed-forward four times wider, which changes step time and memory
substantially. The measurements of 20 November 2026 therefore do not describe the current model.
The current model was measured on 21 November 2026 at the settings the rule needs, and the rule was
applied to that measurement before the search.

Measurements of 21 November 2026, current model, applied.

Measurement of the current model (sections 4.1 and 18.3, corrections 13.1 to 13.4), 2026-11-21, no
other training process running. Before the measurements: `memory_pressure` reported 61% free, that
is 9.76 GiB of 16 GiB, so the memory limit is 7.32 GiB; load averages { 2.42 2.52 2.91 }. Same
procedure as above; the measurements follow the order of item 3 and stop at the first size and
setting that meet items 2 and 4. A process that ran out of MPS memory, was killed by the system or
was ended by a Metal out-of-memory abort is recorded as out of memory; one that ran longer than
1 800 s is stopped and recorded as over the time limit. Measured and applied with `python -m
tools.text_budget` (`tools/text_budget.py`, one process per setting with
`tools/text_step_measurement.py`).

The measurement files of this run are in `results/text/measurements-2026-11-21/budget/`
(`measurements.jsonl`, `oom.jsonl`, `memory.json`, `choice.json` and `section-18.8.md`, from which
the table, the list and the choice below are copied). `tests/test_recorded_measurements.py` checks
that `tools/text_budget.py: decide` on them gives `choice.json` and `section-18.8.md`.
`section-18.8.md` is generated by the command below, last on 23 November 2026; `choice.json` and
the projected hours are those of the measurement of 21 November 2026. The recorded files hold the
date of the measurement without the time of day. The command applies the rule again, without
measuring, to a copy of the folder in a temporary directory outside the repository. It writes into
the folder it is given: `memory.json`, written again with the time limit of the command
(`timeout_s`, 1 800 by default; the free memory in it is read, not measured again); `timing.log`,
appended to, each line starting with the seconds since the command started; `section-18.8.md`; and
`choice.json` if a setting fits. `measurements.jsonl` and `oom.jsonl` are only read, as no
measurement is missing. The generated `section-18.8.md` is then copied back:

```sh
copy=$(mktemp -d)
cp -R results/text/measurements-2026-11-21/budget/. "$copy"
python -m tools.text_budget "$copy" --measure-command false
cp "$copy/section-18.8.md" results/text/measurements-2026-11-21/budget/
```

| size | S | batch | model | layout | parameters | step k=2 (s) | step k=3 (s) | step k=4 (s) | step (s) | peak k=4 (GiB) | validation eval (s) | generation (s/char) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 384 × 6 | 128 | 32 | EBT | split | 14,211,521 | 1.9872 | 2.3218 | 2.7650 | 2.3580 | 10.13 | 16.20 | 0.1055 |
| 256 × 6 | 256 | 32 | EBT | split | 6,328,641 | 2.2643 | 2.8169 | 3.4150 | 2.8321 | 11.02 | 9.99 | 0.1059 |
| 256 × 6 | 128 | 64 | EBT | split | 6,328,641 | 2.0599 | 2.5765 | 3.1018 | 2.5794 | 10.25 | 8.78 | 0.0700 |
| 256 × 6 | 128 | 32 | EBT | split | 6,328,641 | 1.0127 | 1.2627 | 1.5188 | 1.2648 | 5.21 | 8.90 | 0.0484 |
| 256 × 6 | 128 | 32 | EBT | full | 6,328,641 | 1.6077 | 1.9714 | 2.3355 | 1.9715 | 8.21 | 12.97 | 0.0724 |
| 256 × 6 | 128 | 32 | transformer | - | 6,328,064 | | | | 0.1255 | 1.20 | 0.56 | |

Out of memory: 256x6 S128 B64 ebt full; 256x6 S256 B32 ebt full; 256x6 S256 B64 ebt split; 384x6
S128 B32 ebt full; 384x6 S128 B64 ebt split; 384x6 S256 B32 ebt split; 384x6 S256 B64 ebt split.

Application of section 18.7 (item 3):

- 384 × 6, S 256, batch 64: the EBT ran out of memory (split layout).
- 384 × 6, S 256, batch 32: the EBT ran out of memory (split layout).
- 384 × 6, S 128, batch 64: the EBT ran out of memory (split layout).
- 384 × 6, S 128, batch 32: faster layout split (k = 4: split 2.765 s, full out of memory); 10.13
  GiB at k = 4, above 7.32 GiB.
- 256 × 6, S 256, batch 64: the EBT ran out of memory (split layout).
- 256 × 6, S 256, batch 32: faster layout split (k = 4: split 3.415 s, full out of memory); 11.02
  GiB at k = 4, above 7.32 GiB.
- 256 × 6, S 128, batch 64: faster layout split (k = 4: split 3.102 s, full out of memory); 10.25
  GiB at k = 4, above 7.32 GiB.
- 256 × 6, S 128, batch 32: faster layout split (k = 4: split 1.519 s, full 2.336 s); 5.21 GiB at
  k = 4, within 7.32 GiB; mean EBT step 1.265 s, baseline 0.126 s; T = 1 000: search 1.65 h,
  finals 2.10 h; T = 2 000: search 3.19 h, finals 4.03 h; T = 5 000: search 7.83 h, finals 9.83 h.

Chosen: width 256, depth 6 (6 328 641 EBT and 6 328 064 baseline parameters), S = 128, batch 32, T
= 2 000, split layout, validation every 200 steps; projected search 3.19 h and final runs 4.03 h,
each within the v1.0 budget of 8 h per stage.

Departures from Table D.2: S 128 and batch 32 instead of S 256 and batch 64, forced by the rows
above; a size smaller than the authors' xxs (384 × 6), forced by the rows above.

### 18.9 Results of the final runs

Final runs of 23 November 2026, the configurations chosen in section 18.5, seeds 0–4, full tier
(`results/text/lm/full/table.md` and the per-seed files `ebt-seed<s>.json` and
`transformer-seed<s>.json`). Values at each run's best checkpoint (lowest validation BPC); mean
over the 5 seeds ± half-width of the 95% t-interval.

| Model | test BPC | out-of-distribution BPC | validation BPC | parameters |
|---|---|---|---|---|
| EBT, S2, 4 inference steps | 2.332 ± 0.022 | 2.612 ± 0.061 | 2.115 ± 0.011 | 6 328 641 |
| Causal Transformer | 2.266 ± 0.020 | 2.538 ± 0.034 | 2.045 ± 0.008 | 6 328 064 |

Per seed (test / out-of-distribution / validation BPC, best step):

| Seed | EBT | Transformer |
|---|---|---|
| 0 | 2.3485 / 2.6380 / 2.1274, step 1800 | 2.2799 / 2.5603 / 2.0429, step 1800 |
| 1 | 2.3326 / 2.6792 / 2.1151, step 1800 | 2.2727 / 2.5352 / 2.0380, step 1800 |
| 2 | 2.3502 / 2.5953 / 2.1145, step 2000 | 2.2536 / 2.5061 / 2.0429, step 1800 |
| 3 | 2.3115 / 2.5997 / 2.1155, step 2000 | 2.2441 / 2.5173 / 2.0455, step 1600 |
| 4 | 2.3164 / 2.5486 / 2.1027, step 1800 | 2.2780 / 2.5689 / 2.0562, step 1800 |

Passes per window (forward / first-order backward / second-order, section 18.6): training, per
fresh window with the replayed rows included, EBT forward and first-order backward 4.48715 on
average over the seeds (4.43875 to 4.55700; the step count k is drawn per step) and second-order
1.49975 on every seed, transformer 1 / 1 / 0; inference, EBT 4 / 4 / 0 and transformer 1 / 0 / 0.
Every final result file records the reused search entries and choice of its model
(`code.reused_stale_files`, section 18.5). The samples of seed 0 are in
`results/text/lm/full/samples.md`. The G3 and G4 results on the EBT checkpoints are in section
19.5.

Figure ([reproducing.md, Figures](reproducing.md#figures)):
`results/text/lm/full/validation.png`, the validation bits per character against the training
step, one line per seed.

## 19. Inference on saved checkpoints (questions G3 and G4)

Implementation: the shared code in `src/ebt/experiments/inference_g3_g4.py` and one module per
task, `inference_regression.py`, `inference_mnist.py` and `inference_text.py`, run as

```sh
python -u -m ebt.experiments.inference_g3_g4 --task regression --tier full --device cpu
python -u -m ebt.experiments.inference_g3_g4 --task mnist --tier full --device mps
python -u -m ebt.experiments.inference_g3_g4 --task text --tier full --device mps
```

Written before any G3 or G4 result was read. The rules of sections 19.2 and 19.4 were revised on
21 November 2026, after the first regression and MNIST results had been seen, and every result of
section 19.5 was computed with the revised rules ([changelog](../CHANGELOG.md#v10)).

### 19.1 Procedure and files

- Only the final checkpoints of seeds 0–4 are loaded (`restore` of each experiment); nothing
  is trained. Every evaluation uses the test split. Only the EBT of the text task is evaluated:
  the transformer has no thinking loop.
- Code fingerprint per task. The fingerprint of `src/ebt/fingerprint.py` covers one module of
  `ebt.experiments` and the `ebt` modules it imports. The shared module imports no task code;
  each task module imports the shared module and the experiment of its own task only, and the
  command line loads the task module by name. A result carries the fingerprint of its task
  module (`code_fingerprint("inference_regression")`, `"inference_mnist"` or
  `"inference_text"`, recorded as `fingerprint_module`), so a change in the code of one task, or
  of a module only it imports, leaves the results of the other tasks current; a change in the
  shared module makes all of them stale.
- One result file per checkpoint, `results/<task>/g3_g4/<tier>/<model>-seed<s>.json`, written
  as soon as it is finished. It records that code fingerprint, the checkpoint's path, its code
  fingerprint and its SHA-256, the evaluation seeds and any limit on the test rows.
  `summary.json` and `table.md` collect the seeds: each value as the mean over seeds with a 95%
  t-interval (`src/ebt/statistics.py`). Undefined values of a seed (None or not finite) are
  left out of the mean and its interval; the table then shows "(k of 5 defined)", and a value
  with no defined seed as "undefined".
- A rerun skips a result whose fingerprints, checkpoint and settings equal the current ones.
  Before any work, the run stops and lists every stale result (other code, another checkpoint
  or another setting) and every checkpoint whose fingerprint differs from the current code of
  its experiment. With `--reuse-stale` the stale results are kept, not recomputed, and listed
  in `summary.json` (`reused_stale_files`); a checkpoint from other code is then evaluated and
  its result marked `checkpoint_from_other_code: true`.
- Seeds: source 0 (`g3_initial`) gives the initial candidates of G3 and of the (c3)
  experiment, source 1 (`g4_candidates`) the candidates of G4; 2 000 000 + 1000 i + s for the
  checkpoint of run s (section 16.6). All random draws are made on the CPU, except the
  derivative-free optimizer and the IRED sampler, which draw on the evaluation device as in
  their final runs.

### 19.2 G3: error against the number of inference steps

- Step counts 1, 2, 4, 8, 16, 32 and 64 (the grid of questions.md G3). All step counts
  start from the same initial candidates; one step at a time gives the same candidates as one
  call with k steps, because inference has no step-size or step-count randomness. No noise.
- Models: those whose inference is the thinking loop, so that a step count is defined: the
  regression EBTs (b), (c1), (c2) and (c3); the MNIST EBT; the text EBT. The training mode,
  step range and trained inference step count are recorded with each result.
- Initial candidates and error: regression, ŷ_0 ~ N(0, s²) as in training (section 16.2) and the
  mean normalized branch distance r (section 16.7), with the fraction r ≤ 0.1 also recorded;
  MNIST, the noisy test image as in training (section 17.4) and the PSNR of section 17.6;
  text, N(0, 1) logits and the bits per character of section 18.6.
- Ranges (questions.md G3: a range is flat when its change is smaller than the spread across
  seeds). The rule works on ranges, not on neighbouring step counts only:
  1. Pairs. For step counts i < j, m_i is the mean over the 5 seeds and s_i the sample standard
     deviation. The change is m_j − m_i and the spread max(s_i, s_j). The pair is level if the
     change is zero (also when the spread is zero) or its magnitude is smaller than the spread;
     otherwise it is falling or rising by the sign of the change in error. For PSNR a rising
     value is a falling error.
  2. Ranges. A flat range has every pair of its step counts level. A falling (rising) range
     has falling (rising) end points and no rising (falling) pair; pairs inside it may be
     level.
  3. Segmentation, from the first step count a = 1. Let F be the end of the longest flat range
     from a and M the end of the longest falling or rising range from a. If there is no
     falling or rising range from a, or F > M, the next range is the flat range a–F.
     Otherwise it is the falling (rising) range from a to M, ended early at the smallest step
     count c at least two intervals before M such that c–M is flat and the change from a to c
     is still falling (rising). The next range starts where this one ends, until the last
     defined step count.
  4. Undefined. From the first step count at which a value of any seed is undefined, the steps
     form one undefined range from the last defined step count to 64.

  So no flat range holds two step counts whose change is at least the larger of their two
  spreads, and every falling or rising range is monotone in the same sense. A flat stretch of
  two or more intervals at the end of a change becomes its own range; a flat stretch before a
  change is part of the change. Each kind is reported with its ranges, or as absent, and the
  ranges also in order.
- A step whose gradient or candidate is not finite ends the curve of that checkpoint; the
  later values are recorded as undefined, and step 4 above applies.

### 19.3 G3: inference experiment on the (c3) checkpoints

Without retraining, for each (c3) checkpoint (`ebt_regularized`): the trained α divided by 1,
2, 4, 8 and 16, each with plain gradient steps and with the accept-only-lower step, each on the
step counts of section 19.2, from the G3 initial candidates. The accept-only-lower step is the
acceptance rule of section 16.8: a step is kept only where it does not raise the energy. The
clamp to ±2√ᾱ_k and the factor 2 on later steps belong to IRED's landscapes and are not used.
Every setting is reported, with r and the fraction r ≤ 0.1 at every step count; none is chosen,
and the G2 metrics are not used to choose one.

### 19.4 G4: energy as a verifier

Candidates, 32 per test example (a window position for text). The candidate count is named N, as in
the result files (section 1):

- regression EBTs (b) to (c3): 32 initial candidates ŷ_0 ~ N(0, s²), each thought for the
  trained 4 inference steps; the energy is E(x, ŷ_4);
- (c5) IRED: 32 annealed samples from 32 initial draws; the energy is that of landscape 0 in
  the original units;
- (c4) IBC: 32 restarts of the derivative-free optimizer; the energy is E(x, ŷ) at the returned
  candidate;
- (a2) MDN: 32 samples y ~ p(y | x), a kernel drawn with probability α_i and then a draw from
  it; no energy, so the MDN enters the coverage only;
- MNIST EBT: 32 starts, each the noisy test image plus independent N(0, σ²) noise with
  σ = √(1 − ᾱ) = 0.3209 at noise level 0.1, the scale of the corruption already in the image,
  followed by the trained inference (2 steps, no Langevin noise). σ is fixed by this definition
  and not tuned. The definition is needed because the authors' image denoising code always
  starts from the noisy image and has no multi-candidate inference; it follows the
  independently initialized candidates of Algorithm 2 (section 5.2). The PSNR of the standard
  output (from the unperturbed noisy image) is reported next to that of the first perturbed
  candidate;
- text EBT: 32 sets of N(0, 1) initial logits, each thought for the trained 4 steps; energy and
  error per position.

Error: regression, r; MNIST, the per-image mean squared error in pixel units after the clamp
of section 17.6, on the 0–255 scale of the PSNR (its mean over images is the MSE of the PSNR);
text, the cross-entropy in bits of each position. The MNIST error is therefore not on the scale of
the test MSE of the denoising table, which is the [−1, 1] scale of section 17.6.

- Spearman correlation between final energy and error over the test examples (positions for
  text), ties given their mean rank, one value per checkpoint; regression and text use the
  first of the 32 candidates, MNIST the standard output (the first perturbed candidate is also
  recorded). The interval is the 95% t-interval over the 5 per-seed values. The correlation is
  undefined (None) when the energy or the error is constant over the test examples.
- Best-of-N for N = 1, 2, 4, 8, 16 and 32 over the first N of the 32 candidates: the mean error
  of the lowest-energy candidate (the first in candidate order among equal energies), of
  random selection (the expected error of a uniformly random candidate, the mean of the N
  errors) and of the oracle (the lowest of the N errors). The fraction is
  (random − lowest energy) / (random − oracle), per checkpoint, and its summary is the mean of
  the per-seed fractions. It is undefined when the oracle's gain over random selection is at
  most 1e-5 · |random|, relative to the random error. The errors come from float32 candidates
  (resolution about 1.2e-7 relative); the factor leaves room for the rounding accumulated
  over the inference steps. The rule includes N = 1 and N candidates with equal errors, and it
  is the only undefined rule of the best-of-N.
- For MNIST, each best-of-N row also records the PSNR of the lowest-energy and of the oracle
  candidates (`lowest_energy_psnr`, `oracle_psnr`), and the G4 record holds the PSNR of the
  standard output (`standard_psnr`) and of the first perturbed candidate
  (`perturbed_first_psnr`), with the Spearman correlation of each (`spearman`,
  `spearman_perturbed_first`).
- Non-finite values. If any energy, error or candidate of a checkpoint's G4 evaluation is not
  finite, or a step raises a floating-point error, every G4 value of that checkpoint
  (Spearman, best-of-N, coverage) is undefined, as a non-finite step makes the later G3 values
  undefined. A floating-point error during a step is recorded as its message
  (`floating_point_error`) with no count (`non_finite_values` null); otherwise the number of
  non-finite values found by the check after the steps (`g4_non_finite`) is recorded in
  `non_finite_values`. No non-finite value reaches the ranks or the best-of-N. No current result
  has either.
- Both-branch coverage (regression; the EBTs, (c5), (c4) and (a2)): the fraction of test x whose
  first N candidates include one with r ≤ 0.1 on the positive branch and one with r ≤ 0.1 on
  the negative branch.

### 19.5 Results

Final checkpoints of seeds 0–4, full tier; mean over the 5 seeds ± half-width of the 95%
t-interval. Files: `results/regression/g3_g4/full/`, `results/mnist/g3_g4/full/` and
`results/text/g3_g4/full/` (`summary.json`, `table.md` and one file per checkpoint). No value of
any checkpoint is undefined by the non-finite rule of section 19.4 (`non_finite_values` is 0
everywhere).

Range labels. The ranges of section 19.2 describe the error: "falling" means the error falls
and "rising" that it rises. For MNIST the error is read from the PSNR, so "rising" means that
the PSNR falls.

G3 ranges:

| Model | Error | Ranges |
|---|---|---|
| regression (b) `ebt_mse` | branch distance r | rising 1–4; flat 4–64 |
| regression (c1) `ebt_contrastive` | branch distance r | falling 1–16; flat 16–64 |
| regression (c2) `ebt_combined` | branch distance r | rising 1–4; flat 4–64 |
| regression (c3) `ebt_regularized` | branch distance r | rising 1–2; flat 2–64 |
| MNIST EBT | PSNR | rising 1–64 |
| text EBT | bits per character | flat 1–64 |

G3, MNIST (PSNR in dB; trained inference: 2 steps):

| Steps | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| PSNR | 27.6317 ± 0.0536 | 27.0003 ± 0.0307 | 25.4051 ± 0.0422 | 23.4187 ± 0.0807 | 21.2754 ± 0.1459 | 19.1120 ± 0.2524 | 17.1189 ± 0.3812 |

G3, text (test bits per character; trained inference: 4 steps):

| Steps | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| BPC | 2.3208 ± 0.0218 | 2.3236 ± 0.0214 | 2.3318 ± 0.0222 | 2.3338 ± 0.0218 | 2.3343 ± 0.0216 | 2.3343 ± 0.0216 | 2.3344 ± 0.0215 |

G3 at the trained step count and the final test value. The text G3 curve starts from its own
initial logits (seed source `g3_initial`, section 19.1), not from the initial logits of the
final run's test evaluation (`fixed_seeds.test_initial`), so its value at 4 steps differs from
the final test BPC of section 18.9 within noise. Differences per seed, G3 at 4 steps minus final
test BPC, seeds 0–4: +0.000108, +0.000034, +0.000014, −0.000017, −0.000129. The MNIST G3 curve
starts from the noisy test image, as the final run does, and its value at 2 steps equals the
final test PSNR exactly on seeds 1–4 and to within 4e-15 dB on seed 0 (26.96023075042699
against 26.960230750426994); the mean is 27.0003 ± 0.0307 in both.

G4, Spearman correlation between final energy and error. The number of pairs per seed is the
number of test examples: regression 4 096 x, MNIST 10 000 images, text 55 680 positions (435
test windows × 128 positions) on every seed.

| Model | Spearman ρ |
|---|---|
| regression (b) `ebt_mse` | −0.1694 ± 0.1290 |
| regression (c1) `ebt_contrastive` | 0.8650 ± 0.0447 |
| regression (c2) `ebt_combined` | 0.0168 ± 0.2449 |
| regression (c3) `ebt_regularized` | 0.1674 ± 0.3707 |
| regression (c4) IBC | −0.2358 ± 0.1165 |
| regression (c5) IRED | 0.0850 ± 0.4003 |
| MNIST EBT, standard output | 0.6124 ± 0.0083 |
| MNIST EBT, first perturbed candidate | 0.6434 ± 0.0095 |
| text EBT | 0.0482 ± 0.2094 |

G4, best-of-N fraction (random − lowest energy) / (random − oracle). N = 1 is undefined for every
model; for (c5) the fraction is defined on 1 of the 5 seeds at every N > 1.

| Model | N = 2 | N = 4 | N = 8 | N = 16 | N = 32 |
|---|---|---|---|---|---|
| regression (b) | −0.065 ± 0.073 | −0.016 ± 0.072 | 0.000 ± 0.068 | 0.002 ± 0.060 | 0.003 ± 0.050 |
| regression (c1) | 0.992 ± 0.006 | 0.997 ± 0.003 | 0.999 ± 0.001 | 0.999 ± 0.001 | 1.000 ± 0.000 |
| regression (c2) | −0.096 ± 0.150 | −0.063 ± 0.082 | −0.038 ± 0.056 | −0.027 ± 0.049 | −0.020 ± 0.044 |
| regression (c3) | 0.135 ± 0.474 | −0.109 ± 0.772 | −0.552 ± 1.246 | −1.203 ± 1.821 | −2.049 ± 2.490 |
| regression (c4) | 0.077 ± 0.098 | 0.042 ± 0.079 | 0.031 ± 0.059 | 0.024 ± 0.047 | 0.020 ± 0.040 |
| regression (c5), 1 of 5 seeds | −1.000 | −0.684 | −0.618 | −0.507 | −0.558 |
| MNIST EBT | −0.119 ± 0.033 | −0.198 ± 0.030 | −0.264 ± 0.031 | −0.323 ± 0.029 | −0.379 ± 0.025 |
| text EBT | −0.399 ± 0.152 | −0.408 ± 0.137 | −0.436 ± 0.157 | −0.442 ± 0.185 | −0.450 ± 0.242 |

The errors of the random, lowest-energy and oracle selections at every N are in the `table.md`
of each task. MNIST: PSNR of the standard output 27.0003 ± 0.0307, of the first perturbed
candidate 25.4971 ± 0.0508.

The both-branch coverage of every model is in section 16.8.

The (c3) inference experiment of section 19.3 is in `results/regression/g3_g4/full/table.md`;
the fraction with r ≤ 0.1 after 64 steps is in questions.md G3.

Figure ([reproducing.md, Figures](reproducing.md#figures)): `results/g3_steps.png`, the G3 curves
of the three tasks, drawn from the three `summary.json` files by `python -m tools.g3_figure`.

## 20. Thinking out of distribution (question G5)

Status: measured; the results are in section 20.5.

Written on 23 November 2026, before any out-of-distribution G3 value was computed; the only
out-of-distribution value seen was the text final runs' BPC at the trained 4 steps (section 18.9).
The rule of section 20.2 and its outcome texts were refined on the same day
([changelog](../CHANGELOG.md#v10)).

Implementation: `src/ebt/experiments/inference_g5.py`, run as

```sh
python -u -m ebt.experiments.inference_g5 --task regression --tier full --device cpu
python -u -m ebt.experiments.inference_g5 --task mnist --tier full --device mps
python -u -m ebt.experiments.inference_g5 --task text --tier full --device mps
python -m tools.g5_report
```

### 20.1 What is measured

Only G3 is measured out of distribution: the task error after 1, 2, 4, 8, 16, 32 and 64
inference steps from the same initial candidates, with no noise, exactly as in section 19.2,
on the final checkpoints of seeds 0–4. Nothing is trained and no G4 value is computed out of
distribution. The in-distribution side is the existing G3 results of section 19
(`results/<task>/g3_g4/full/<model>-seed<s>.json`, field `g3`); it is not recomputed.

Results are written to `results/regression/g5/full/`, `results/mnist/g5/full/noise-<level>/`
and `results/text/g5/full/`, one file per checkpoint with the code fingerprint of
`inference_g5` (it covers the three task modules of section 19 and their experiments), the
checkpoint's path, fingerprint and SHA-256, the row limit, and the split and, for MNIST, the
noise level and seed (`settings`); a rerun skips a current result and stops at a stale one, as
in section 19.1. The fingerprints of the G3 and G4 results do not cover `inference_g5`. The split
is chosen by name (`split_of`, which returns the rows of the named split of a task's data
without changing the data); regression uses the G3 function of `inference_regression` on those
rows, text the G3 part of `inference_text.evaluate` as a function of the rows (`text_g3`;
`evaluate` always also computes G4), and MNIST `mnist_g3` on the test images noised at a level
(`noisy_images`). Seeds of the initial candidates: the `g3_initial` source of section 19.1, the
same as in distribution. The run time of each checkpoint is recorded in its result file
(`seconds`).

`tools/g5_report.py` computes I_ID, I_OOD and Δ and the outcome of section 20.2 and writes
`g5.json` and `table.md` next to the out-of-distribution results (`tests/test_g5.py`). Before
computing it checks every result: each out-of-distribution file of every model, seed and level
must exist, carry the current `inference_g5` fingerprint, have no row limit and the expected
split and level (for MNIST the noise level and its noise seed), and come from the same checkpoint
(equal SHA-256) as the in-distribution result of its seed. Both files must have every field the
report reads: the seed, `checkpoint.sha256`, the trained step count and a G3 curve with a value
at every step of the grid. Any failure stops the report with the file named, and for a missing
field the field named (`G5ReportError`).

### 20.2 Relative improvement, its interval and the outcome

Definition C.1 of the paper: STT(x, θ, F) = E_x[P(x, θ, F) / P(x, θ, F₀) − 1], the expected
relative change of a performance P from F₀, the fewest function evaluations that give a
prediction, to F evaluations. Here F₀ = 1 inference step and F = 64, the ends of the G3 grid.
The paper does not give the step counts behind its Figure 7, so F₀ and F follow Definition C.1
only. The trained step count (regression 4, MNIST 2, text 4) is reported alongside, with the
error at it.

Departure from the paper. Figure 7 (§4.1.3, language) plots the improvement of "Max Thinking",
which combines thinking longer with self-verification (best of several candidates), against the
size of the distribution shift. G5 measures thinking longer only, as G3 does; self-verification
is measured by G4, in distribution. For images the paper (§4.3) compares the EBT with a
diffusion transformer at a test noise level of 0.2 after training at 0.1; it does not claim that
thinking helps more out of distribution there. The MNIST part of G5 is this project's test, at
the paper's test level.

The errors here are lower-is-better, so the improvement is the relative reduction of the error,
the form of the paper's percent perplexity improvement:

    I = 1 − e₆₄ / e₁

per checkpoint, with e_k the error after k steps:

- regression: e_k the mean normalized branch distance r over the split (section 16.7);
- MNIST: e_k the MSE in pixel units over every pixel of the split, the MSE behind the PSNR of
  section 17.6, so I = 1 − 10^((PSNR₁ − PSNR₆₄) / 10);
- text: e_k the perplexity 2^BPC over every position of the split, so
  I = 1 − 2^(BPC₆₄ − BPC₁).

I lies in (−∞, 1]: I = 1 means the error vanished, I = 0 that it did not change, I = −1 that it
doubled. Negative values are therefore not symmetric with positive ones: halving the error gives
0.5, doubling it −1. I is undefined for a checkpoint if e₁ or e₆₄ is undefined (a non-finite
step, section 19.2) or e₁ = 0.

Ratio of dataset-level errors instead of the per-input average. Definition C.1 averages the
ratio over inputs. Here e₁ and e₆₄ are each the error over the whole split and I is computed
from their ratio. For regression the per-input ratio is unbounded: one step can bring a
prediction onto a branch (r ≈ 0), and the ratio of that input then has a near-zero
denominator. For text the per-position perplexity is at least 1, so its ratio is bounded there;
the dataset-level form is used for every task so that I is computed in one way, from the
quantities the G3 results record (per-input values are not stored), and it is the form of a
perplexity or a PSNR over a split.

Starting point. With F₀ = 1 the error after one step depends largely on the initial candidates,
and these relate differently to the two splits before any step: for regression the initial
candidates N(0, s²) have the scale s of the training targets, while the out-of-distribution
targets reach |y| = 3; for MNIST the out-of-distribution start is a noisier image. Δ therefore
includes an effect of the starting point as well as of thinking. The errors at the trained step
count are reported next to I so that this can be seen.

Comparison. For each seed s, I_ID(s) and I_OOD(s) come from the same checkpoint, and
Δ(s) = I_OOD(s) − I_ID(s). I_ID, I_OOD and Δ are reported as the mean over the seeds ± the
half-width of the 95% t-interval (`src/ebt/statistics.py`, t with n − 1 degrees of freedom);
a seed with an undefined I is left out of that quantity and counted. An interval is above 0 if
its lower end is above 0 and below 0 if its upper end is below 0; an end exactly at 0 is neither.

Outcome, the first that applies, in this order, with the text the report writes
(`tools/g5_report.py: outcome_text`):

1. Undefined: I_ID, I_OOD or Δ has fewer than 2 defined seeds, so it has no interval. Text:
   "Undefined: I_ID, I_OOD or Δ has fewer than 2 defined seeds."
2. Thinking helps more out of distribution: the Δ interval and the I_OOD interval are both above
   0. Text: "Thinking helps more out of distribution." If the mean of I_ID is below 0, the text
   is instead "Thinking helps out of distribution and degrades the result in distribution; Δ
   therefore includes the in-distribution degradation." For the text task only, the text adds
   "This is the effect the paper reports for language (§4.1.3).", or, when the mean of I_ID is
   below 0, "Out of distribution, this matches the effect the paper reports for language
   (§4.1.3)."; for MNIST and regression it makes no reference to the paper (§4.3 makes no such
   claim for images, and the regression task has no counterpart in the paper).
3. Thinking degrades the result in both: the mean of I_ID is below 0, and the I_OOD interval is
   below 0 or the mean of I_OOD is below 0. The relative degradation out of distribution is
   smaller (Δ interval above 0), larger (Δ interval below 0) or not measurably different
   (otherwise). Text: "Thinking degrades the result in both; the relative degradation out of
   distribution is smaller." (or "larger", or "not measurably different"), followed for the text
   task only by "This is not the paper's effect." A side whose interval is not below 0, so that
   it is below 0 on its mean only, is marked "on average": "in both" then reads "in distribution
   on average and out of distribution", "in distribution and out of distribution on average" or
   "in distribution on average and out of distribution on average".
4. Thinking degrades the result out of distribution only: the I_OOD interval is below 0 and the
   mean of I_ID is not below 0. Text: "Thinking degrades the result out of distribution only.",
   followed for the text task only by "This is not the paper's effect."
5. The difference is above 0 but the out-of-distribution improvement is not measurably above 0:
   the Δ interval is above 0 and the I_OOD interval is not. Text: "The difference is above 0 but
   the out-of-distribution improvement is not measurably above 0.", followed for the text task
   only by "This is not the paper's effect."
6. Thinking helps less out of distribution: the Δ interval is below 0. Text: "Thinking helps
   less out of distribution." If the mean of I_OOD is below 0, the text is instead "Thinking
   does not help out of distribution on average; its effect there is smaller than in
   distribution."
7. No measurable difference: otherwise. Text: "No measurable difference."

For MNIST and regression outcomes 3, 4 and 5 make no reference to the paper, for the reason
given under outcome 2.

For MNIST the G5 outcome is that of level 0.2; levels 0.15 and 0.3 are reported with their
intervals and no outcome.

### 20.3 Out-of-distribution settings

MNIST. The test images (10 000) noised once at three levels above the training level 0.1, each
with its own fixed generator, with the noise of section 17.2 (`mnist.add_noise` with
`mnist.alpha_bar(level)`); thinking starts from the noisy image and runs the trained inference
loop without noise, as in section 19.2. The model is not told the level, and α is not changed.

| Level | Role | ᾱ | √ᾱ | √(1 − ᾱ) | Noise seed |
|---|---|---|---|---|---|
| 0.1 | in distribution (section 19) | 0.89702 | 0.9471 | 0.3209 | 40 001 |
| 0.15 | additional | 0.78836 | 0.8879 | 0.4600 | 40 002 |
| 0.2 | primary | 0.65972 | 0.8122 | 0.5833 | 40 003 |
| 0.3 | additional | 0.39587 | 0.6292 | 0.7773 | 40 004 |

- 0.2 is the primary level: it is the paper's test level for images (§4.3, training at 0.1),
  and section 17.2 named it as the out-of-distribution level of G5.
- 0.15 and 0.3 are a smaller and a larger shift, so that the improvement can be read against the
  size of the shift; they are reported without an outcome.
- No level of 0.4 or above: at 0.4 the clean image enters with weight √ᾱ = 0.4404 and the noise
  with weight √(1 − ᾱ) = 0.8978.
- The noise seeds are new fixed seeds next to those of section 17.5 (40 000 and 40 001), below
  every training and evaluation seed.

Regression. The out-of-distribution split of section 16.1: 4 096 examples with x sampled
uniformly in [4, 9) and y = ±√x, from its own generator of the data seed 0. Initial candidates
N(0, s²) with s = 1.4565, the training-split value, as in distribution.

- The range adjoins the training range [0.25, 4) on the side away from x = 0, where the branches
  meet and r, normalized by √x, is ill-conditioned.
- The targets reach |y| = 3, 1.5 times the largest training |y| (2), with branch separation
  2√x from 4 to 6, so the shift is in both x and the scale of y.

Text. The out-of-distribution block of section 18.1: 55 770 characters of Pride and Prejudice,
in the 435 non-overlapping windows of 128 used by the final runs, initial logits N(0, 1). The
final runs' BPC on it at 4 steps (section 18.9) uses other initial logits (`ood_initial`), so
it differs from the G3 value at 4 steps within noise, as on the test split (section 19.5).

### 20.4 Models

Included: the models whose inference is the thinking loop, so that error after 1 and after 64
steps is defined, as in section 19.2: the regression EBTs (b) `ebt_mse`, (c1) `ebt_contrastive`,
(c2) `ebt_combined` and (c3) `ebt_regularized`; the MNIST EBT; the text EBT.

Not included:

- the baselines (a), the one-pass MNIST denoiser and the causal Transformer, and (a2) the MDN: they
  predict in one forward pass and have no step count;
- (c4) IBC: its inference is the derivative-free optimizer, with no gradient step count. Its
  candidates are also confined to the bounds of the training targets widened by 5%, about
  ±2.20 in the original units, so for x above about 4.84 neither branch can be reached;
- (c5) IRED: its inference is the fixed annealed schedule of section 16.8 (10 landscapes with 5
  steps each), not a count of thinking steps, and its prediction is clamped to ±2√ᾱ₀ in
  standardized units, −2.86 to 2.88 in the original units, so the negative branch is out of
  reach above x ≈ 8.19 and the positive branch above x ≈ 8.30.

### 20.5 Results

Measured on 24 November 2026, on the final checkpoints of seeds 0–4. The tables are copied
exactly from the report files `results/text/g5/full/table.md`, `results/mnist/g5/full/table.md`
and `results/regression/g5/full/table.md`; the values with their per-seed lists are in `g5.json`
in the same folders. I = 1 − e₆₄ / e₁ per checkpoint and Δ = I_OOD − I_ID per seed (section
20.2); mean over seeds ± half-width of the 95% t-interval. The outcome texts are those the report
wrote. The rule compares 1 step with 64 steps, the comparison pre-registered in section 20.2; the
error at the trained step count is reported for reference only.

Text (`results/text/g5/full/table.md`):

| model | out of distribution | I_ID | I_OOD | Δ | outcome |
|---|---|---|---|---|---|
| ebt | split | -0.0095 ± 0.0060 | -0.0003 ± 0.0074 | 0.0092 ± 0.0101 | Thinking degrades the result in distribution and out of distribution on average; the relative degradation out of distribution is not measurably different. This is not the paper's effect. |

bpc at 1 step, at the trained step count and at 64 steps:

| model | out of distribution | data | 1 | trained | 64 |
|---|---|---|---|---|---|
| ebt | split | in distribution | 2.3208 ± 0.0218 | 2.3318 ± 0.0222 (4 steps) | 2.3344 ± 0.0215 |
| ebt | split | out of distribution | 2.6126 ± 0.0548 | 2.6121 ± 0.0610 (4 steps) | 2.6131 ± 0.0611 |

MNIST (`results/mnist/g5/full/table.md`):

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

Regression (`results/regression/g5/full/table.md`):

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

Reading of "on average" in the outcome texts (outcome 3 of section 20.2), checked against the
intervals:

- Text: the in-distribution degradation is decided on its interval (I_ID −0.0095 ± 0.0060, below
  0); the out-of-distribution degradation is on its mean only (I_OOD −0.0003 ± 0.0074, an
  interval that contains 0).
- Regression `ebt_combined`: in distribution on its interval (I_ID −0.2539 ± 0.0169); out of
  distribution on its mean only (I_OOD −0.0113 ± 0.0400).
- Regression `ebt_regularized`: in distribution on its interval (I_ID −0.1150 ± 0.0844); out of
  distribution on its mean only (I_OOD −0.9601 ± 2.1183).

The outcomes without "on average", MNIST at noise 0.2 and regression `ebt_mse`, have both the
I_ID and the I_OOD interval below 0. Regression `ebt_contrastive` (outcome 4) has its I_OOD
interval below 0 and its I_ID mean above 0. The answer to G5 and descriptive observations that
the rule does not decide are in questions.md (G5).
