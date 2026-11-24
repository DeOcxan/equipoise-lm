# Numerics

This document states which floating-point type is used for what, why results from different
devices are compared with a tolerance, which tolerances are used, and what is reproducible.

Sources (PyTorch pages for version 2.15, the version pinned in `requirements.txt`):

- [S1] PyTorch, Numerical accuracy:
  <https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html>
- [S2] PyTorch, MPS backend: <https://docs.pytorch.org/docs/2.15/notes/mps.html>
- [S3] PyTorch, Reproducibility: <https://docs.pytorch.org/docs/2.15/notes/randomness.html>
- [S4] PyTorch, torch.testing: <https://docs.pytorch.org/docs/2.15/testing.html>
- [S5] PyTorch, torch.autograd.gradcheck:
  <https://docs.pytorch.org/docs/2.15/generated/torch.autograd.gradcheck.gradcheck.html>
- [S6] PyTorch, Type Info: <https://docs.pytorch.org/docs/2.15/type_info.html>
- [S7] Wikipedia, Single-precision floating-point format:
  <https://en.wikipedia.org/wiki/Single-precision_floating-point_format>

The floating-point formats are defined by IEEE 754-2019, Standard for Floating-Point Arithmetic.

## Machine epsilon of float32

`torch.finfo(dtype).eps` is "the difference between 1.0 and the next smallest representable
float larger than 1.0" [S6].

The IEEE 754 binary32 format, which float32 implements, has a significand of 24 bits, 23 of them
stored [S7]. The spacing of floats just above 1.0 is therefore 2⁻²³.

Measured with the pinned torch: `torch.finfo(torch.float32).eps` = 1.1920928955078125e-07, which
equals 2⁻²³. PyTorch describes float32 as accurate to "about 7 decimal digits", and float64 to
about 16 [S1].

## Type per purpose

| Purpose | Type and device | Reason |
|---|---|---|
| Gradient checks against finite differences (`tests/test_gradients.py`, `tests/test_truncation.py`) | float64 on CPU | PyTorch's own gradient checker states "The default values are designed for input of double precision" [S5]. A central difference with step h has a rounding error of order ε/h. With h = 1e-6 this is about 1e-10 in float64, and in float32 it would be about 0.1, so the tests' 1e-6 relative tolerance is only meaningful in float64. |
| Training and evaluation of the experiments | float32, on CPU or MPS | MPS has no float64: "The MPS backend does not support torch.float64 (torch.double) or torch.complex128 (torch.cdouble) because Metal Shading Language has no double type" [S2]. float32 is used on CPU as well, so that results are comparable across the two devices. |
| Comparisons of MPS results with CPU results (`tests/test_mps.py`) | float32 on both | The comparison is only meaningful at a common precision, and MPS supports float32 only [S2]. |

## Why results from different devices are compared with a tolerance

Floating-point addition and multiplication are not associative, so the order of operations
changes the result. Because of this, PyTorch "is not guaranteed to produce bitwise identical
results for floating point computations that are mathematically identical", and bitwise
identical results "are not guaranteed across PyTorch releases, individual commits, or different
platforms". In particular, CPU and GPU results "can be different even for bitwise-identical
inputs and even after controlling for the sources of randomness" [S1].

The same page gives examples within one device. A batched matrix product is not guaranteed to
be bitwise identical to the product of the individual batch elements, and an operation applied
to a slice is not guaranteed to equal the slice of the operation applied to the whole tensor
[S1].

Two consequences for this project:

1. A result computed on MPS is never compared with a CPU result for exact equality. The
   comparison uses a tolerance.
2. A batched computation is compared with per-example computations with a tolerance, even on
   one device. The energy-contract tests (`tests/test_energy_contract.py`, `tests/test_mlp.py`)
   use 1e-14 in float64 for this reason, not exact equality.

## Tolerance policy

- Cross-device comparisons use `torch.testing.assert_close` with its default tolerances. A value
  passes if |actual − expected| ≤ atol + rtol · |expected|. The defaults are rtol = 1.3e-6 and
  atol = 1e-5 for float32, and rtol = 1e-7 and atol = 1e-7 for float64 [S4].
- Tests in float64 on CPU that check an analytic or finite-difference value state their own
  tolerance in the test. The finite-difference tests use a relative error of 1e-6, and the
  hand-derived values use an absolute error of 1e-12.
- A cross-device test gets a tolerance other than the default only from a measurement of its
  float32 errors. For the text EBT gradients: `python -m tools.mps_text_gradients measure`
  computes the errors of the CPU and MPS float32 results against the same computation in
  float64 on CPU, and `python -m tools.mps_text_gradients decide` (`tools/mps_text_gradients.py:
  decide`, `tests/test_mps_text_gradients.py`) decides whether the MPS error is float32 rounding
  (in each layout, the MPS absolute error and normwise relative error each at most 2.5 times
  those of CPU float32) and derives the absolute tolerance.

The MPS tests of the test MLP energy (`tests/test_mps.py`: float32 parameter gradients of a
three-step training trajectory in S1, S2 and full unroll) use the float32 defaults of
`assert_close`. Their measured differences are not recorded in a file of this repository, so no
values are given here.

Text EBT gradients. The S2 parameter gradients of the small causal text EBT
(`tests/test_mps.py: test_mps_text_energy_s2_gradients_match_cpu`; three thinking steps through
softmax, rotary attention, SwiGLU and two RMSNorms, then a double backward) differ between MPS
and CPU by up to 1.67e-5 in the split layout, more than the float32 default atol of 1e-5, and by
up to 6.68e-6 in the full layout. Both float32 results were compared with the same computation in
float64 on CPU (`python -m tools.mps_text_gradients measure`). Errors against that reference:
absolute, elementwise relative (elements above 1e-3 of their tensor's largest value) and
normwise relative (‖error‖ / ‖reference‖ per tensor), the largest over all parameter tensors;
largest reference gradient magnitude 8.9. The values are those of `mps-decision.txt` (below), to 3
significant figures:

| Layout | CPU float32: abs | rel | normwise | MPS float32: abs | rel | normwise | MPS − CPU float32: abs |
|---|---|---|---|---|---|---|---|
| full | 2.05e-5 | 9.41e-4 | 2.94e-6 | 2.05e-5 | 1.17e-3 | 2.71e-6 | 6.68e-6 |
| split | 1.26e-5 | 1.84e-3 | 3.90e-6 | 1.02e-5 | 1.25e-3 | 2.55e-6 | 1.67e-5 |

The MPS absolute and normwise relative errors are within 2.5 times those of CPU float32 in both
layouts, so the difference is float32 rounding, not an MPS defect. The two float32 results can
differ by up to the sum of their errors, so the test uses atol = 7.2e-5: 3.5 times the largest CPU
float32 absolute error (3.5 × 2.05e-5 = 7.175e-5), rounded up, and at least 1.5 times the largest
measured MPS − CPU difference (1.5 × 1.67e-5 = 2.505e-5), with the float32 default rtol (`python -m
tools.mps_text_gradients decide`, `tools/mps_text_gradients.py: decide`). The diagnosis of 21
November 2026 that gives the numbers above is
`results/text/measurements-2026-11-21/mps-gradients/mps-diag.jsonl`, with the decision it gave in
`mps-decision.txt` next to it. The decision is applied to it again with

```sh
python -m tools.mps_text_gradients decide results/text/measurements-2026-11-21/mps-gradients/mps-diag.jsonl
```

and `tests/test_recorded_measurements.py` checks that it gives float32 rounding and
atol = 7.2e-5.

## Reproducibility

- Seeding. "You can use torch.manual_seed() to seed the RNG for all devices (both CPU and
  CUDA)" [S3]. `src/ebt/runtime/seeding.py: seed_everything` does this. Every other random draw
  in the experiments (data, batch order, initial candidates, thinking noise, replay sampling)
  comes from an explicit `torch.Generator` with its own seed. Python's `random` module and NumPy's
  random generator are not used.
- Limits. "Completely reproducible results are not guaranteed across PyTorch releases,
  individual commits, or different platforms" [S3]. Exact reproducibility is therefore claimed
  only for a fixed torch version on one platform.
- CPU: exact. With the same seed on CPU, a run writes a byte-identical result file
  (`tests/test_training.py: test_same_seed_on_cpu_gives_an_identical_result_file`).
- MPS: not bit-reproducible. No exactness is claimed or tested for MPS runs. The tolerances
  above cover the parameter gradients of short trajectories (three thinking steps) against the
  CPU, not whole training runs of thousands of steps, so a training run repeated on MPS can give
  different numbers. The float32 errors measured above (normwise relative 2.55e-6 to 3.90e-6 for
  the text EBT gradients against float64) are far below the seed-to-seed spread of the reported
  metrics. The
  confidence intervals in the result tables show that spread, the spread to expect between a
  rerun and the committed values.
- Result files record the torch version, the Python version and the platform
  (`src/ebt/results.py: environment`), so that a rerun can use the same ones.
