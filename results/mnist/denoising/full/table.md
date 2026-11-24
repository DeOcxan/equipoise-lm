Test split, 10 000 images, noise level 0.1; mean over 5 seeds ± half-width of the 95% t-interval. PSNR from one MSE over all test pixels (method.md section 17.6). Passes per example: forward / first-order backward / second-order.

| Model | test PSNR (dB) | test Smooth L1 | test MSE | parameters | training passes | inference passes |
|---|---|---|---|---|---|---|
| EBT, S1, 2 inference steps | 27.00 ± 0.03 | 0.00400 ± 0.00003 | 0.00801 ± 0.00006 | 828,033 | 1 / 1 / 1 | 2 / 2 / 0 |
| Feed-forward denoiser | 27.54 ± 0.04 | 0.00353 ± 0.00003 | 0.00707 ± 0.00006 | 829,968 | 1 / 1 / 0 | 1 / 0 / 0 |
| Noisy input | 17.63 (fixed noise, no spread) | | | | | |

EBT step size α at the end of training: 17869.0 ± 1551.6.

Change of the validation Smooth L1 over the last 2000 training steps and
between the last two evaluations (negative: still falling):

| Model | last steps | last two evaluations | seeds still falling |
|---|---|---|---|
| EBT, S1, 2 inference steps | -0.000062 ± 0.000043 | -0.000007 ± 0.000012 | 4 of 5 |
| Feed-forward denoiser | -0.000055 ± 0.000009 | -0.000003 ± 0.000006 | 4 of 5 |

Chosen configurations (search seed 100, validation Smooth L1 after inference):

- Feed-forward denoiser: learning_rate 0.003; validation Smooth L1 0.003566.
- EBT, S1, 2 inference steps: learning_rate 0.003, step_size 30000.0; validation Smooth L1 0.003987.
