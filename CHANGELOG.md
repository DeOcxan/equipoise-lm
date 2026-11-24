# Changelog

## v1.0

Released on 24 November 2026 as code version 0.1.0 ([docs/reproducing.md,
Installation](docs/reproducing.md#installation)). It contains the autoregressive and EBT families
on Tiny Shakespeare at equal parameters and equal steps, and the EBT study on three tasks
([docs/questions.md, section C](docs/questions.md#c-the-v10-answers)).

### Decisions and revisions made after results had been seen

The list below gives every such decision and revision of v1.0 with its date and reason. What they
mean for the answers is stated in [docs/questions.md, Limitations and threats to
validity](docs/questions.md#limitations-and-threats-to-validity).

#### 19 November 2026: four models added to G2

The first G2 results of (a), (b), (c1) and (c2) had been seen: (b) and (c2) predicted near the mean
of the two branches, and (c1) diverged. Four models were then added (docs/method.md sections 16.4
and 16.8):

- (c3), the contrastive loss with an energy-magnitude penalty [Du-Mordatch], as the remedy for the
  (c1) divergence. The divergence was diagnosed from the numbers of the (c1) search: across its 18
  configurations the mean |ŷ_final − ŷ_0| on validation data was 12–23 at α = 0.1 and 57–80 at
  α = 1 (docs/method.md section 16.4).
- (a2), a mixture density network [Bishop]; (c4), implicit behavioural cloning [IBC]; and (c5),
  IRED [IRED]. Reason: G2 then compares the main ways of obtaining multimodal predictions, rather
  than the EBT against one regressor. Each was implemented from its primary source.

The hypotheses for (a2), (c4) and (c5) were written after a 200-step trial run of each, after their
full-tier search had started and before any full-tier G2 metric was read. The date of the trial
runs is not recorded.

#### 20 to 21 November 2026: item 3 of the text budget rule reworded

Item 3 of the budget rule (docs/method.md section 18.7) was applied to the measurements of 20
November 2026 with the wording of that date. The new wording was adopted after that application had
been seen and before the measurement of 21 November 2026.

- Wording of 20 November 2026: "If no size fits with them, the setting steps down before the size
  does". Its literal reading takes the largest size that fits with S 256 and batch 64 first, and
  tries a smaller setting only if no size fits.
- New wording, as in docs/method.md section 18.7: for each size from the largest,
  the settings step down before the size does.
- Reason: the literal reading lets a preferred setting outweigh the size, and the condition "if no
  size fits with them" contradicts the words "before the size does".

Neither application to the measurements of 20 November 2026 was used, because those measurements
describe the model before corrections 13.1 to 13.4 (below). The records are in
`results/text/measurements-2026-11-20/`. `tests/test_recorded_measurements.py` reproduces from them
the 128 × 4 projections and the application with the new wording, which `tests/test_text_budget.py`
also reproduces from `tests/data/text_budget_2026-11-20.jsonl`. The application with the wording of
20 November 2026, which chose 128 × 4, is recorded here and reproduced by no test:

- With the wording of 20 November 2026: width 128, depth 4, S = 256, batch 64, T = 2 000, split
  layout. The 256 × 6 model needed 11.73 GiB at k = 4 at S 256 and batch 64, above the limit of
  7.68 GiB, and 128 × 4 needed 5.16 GiB. The 128 × 4 projections were 3.07 h for the search and
  3.85 h for the final runs at T = 2 000, and 7.63 h and 9.55 h at T = 5 000.
- With the new wording: width 256, depth 6, S = 256, batch 32, T = 2 000, split
  layout. 256 × 6 at batch 32 needed 7.26 GiB at k = 4, within 7.68 GiB. The projections were
  5.08 h for the search and 6.39 h for the final runs at T = 2 000. 384 × 6 was not measured, so
  this result held only if 384 × 6 met the rule at none of its settings.

The rule as reworded was applied to the measurement of 21 November 2026 of the current model. That
application gives the setting in use (docs/method.md section 18.8).

#### 21 November 2026: G3 and G4 rules revised

docs/method.md section 19 was written before any G3 or G4 result was read. Three rules were changed
after the first regression and MNIST G3 and G4 results had been seen:

- (a) Range rule (docs/method.md section 19.2). Until this date it compared neighbouring step
  counts pair by pair against the seed spread. A slow, steady change made of small steps, each
  within the spread, was then reported as flat. From this date it compares every pair of step
  counts in a range, so a range is flat only if no two of its step counts differ by at least their
  spread.
- (b) Undefined best-of-N fraction (docs/method.md section 19.4). From this date the fraction is
  undefined when the oracle's gain over random selection is at most 1e-5 · |random|. Reason: at the
  level of float32 rounding the fraction divides by a near-zero gap, and its value is meaningless.
  The threshold leaves room above the float32 resolution (about 1.2e-7 relative) for the rounding
  accumulated over the inference steps.
- (c) Non-finite G4 values (docs/method.md section 19.4). Until this date there was no rule for
  energies, errors or candidates that are not finite. From this date every G4 value of such a
  checkpoint is undefined, so that no non-finite value reaches the ranks, the best-of-N or a mean
  over seeds.

Every G3 and G4 result of docs/method.md section 19.5 was computed with the revised rules. The
regression results of the EBTs, (c5) and (a2) were computed on 21 November 2026, and the (c4),
MNIST and text results on 23 November 2026.

#### 23 November 2026: G5 rule refined and outcome texts set

docs/method.md section 20 was written on 23 November 2026, before any out-of-distribution G3 value
was computed. The regression split x ∈ [4, 9) had been generated but never evaluated, and no MNIST
image had been evaluated at a noise level other than 0.1. One out-of-distribution value existed:
the text final runs' BPC on the out-of-distribution block at the trained 4 steps (docs/method.md
section 18.9), with no curve over step counts. With only that value seen, two changes were made on
the same day:

- The rule of docs/method.md section 20.2 changed. Until then it reported a larger
  out-of-distribution improvement whenever the Δ interval lay above 0. From then on a degradation
  in both settings is a separate outcome (outcome 3). Reason: a rule on the Δ interval alone
  reports a model whose thinking makes the error worse in both settings, only less so out of
  distribution, as helping more, which is not the paper's effect. Section 20.2 also cites Figure 7
  as in §4.1.3 of [EBT], measuring thinking longer together with self-verification, and §4.3 as
  comparing the EBT with DiT without an out-of-distribution claim about thinking. It also states
  the starting-point effect.
- The outcome texts of section 20.2 were set, and the rule did not change with them. The paper is
  named for the text task only (outcomes 2 to 5), as its claim is for language (§4.1.3). §4.3
  makes no such claim for images, and the regression task has no counterpart in the paper.
  Outcome 2 with a negative mean of I_ID states the in-distribution degradation that Δ then
  includes. Outcome 6 with a negative mean of I_OOD states that thinking does not help out of
  distribution on average. Outcome 3 says "on average" for a side that is below 0 on its mean
  only. Reason: a text must not say more than the rule decides. The report also stops with the
  file and the field named when a result lacks a field it reads.

### Corrections 13.1 to 13.4

The text models follow the authors' code on four points where the paper says otherwise
(docs/method.md section 13). The model measured on 20 November 2026 did not have them, and the
model measured on 21 November 2026 and used for every text result has all four (docs/method.md
section 18.8). Results computed with a model without them are not reused.

- 13.1, text thinking step: the gradient is taken with respect to p = softmax(ŷ + η), and the step
  is taken from p.
- 13.2, step-index token: the text EBT has one learned step-index token at sequence position 0.
- 13.3, initialization of the text models: Xavier-normal weights for the linear layers, with the
  query, key and value maps as separate D × D blocks; N(0, 0.02²) for the token embedding; N(0, 1)
  for the step-index embedding.
- 13.4, feed-forward width of the text models: a SwiGLU hidden width of 4D.

Bracketed keys such as [EBT] refer to [docs/design.md, References](docs/design.md#references).
