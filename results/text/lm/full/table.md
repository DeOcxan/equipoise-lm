Bits per character at each run's best checkpoint (lowest validation BPC); mean over 5 seeds ± half-width of the 95% t-interval. Uniform prediction: log2 65 = 6.022. Passes per window: forward / first-order backward / second-order (method.md section 18.6); training passes per fresh window, the EBT's replayed rows included, as the mean over seeds with the per-seed range where the seeds differ.

| Model | test BPC | out-of-distribution BPC | validation BPC | best step | parameters | training passes | inference passes |
|---|---|---|---|---|---|---|---|
| EBT, S2, 4 inference steps | 2.332 ± 0.022 | 2.612 ± 0.061 | 2.115 ± 0.011 | 1800, 1800, 2000, 2000, 1800 | 6,328,641 | 4.48715 (4.43875 to 4.55700) / 4.48715 (4.43875 to 4.55700) / 1.49975 | 4 / 4 / 0 |
| Causal transformer | 2.266 ± 0.020 | 2.538 ± 0.034 | 2.045 ± 0.008 | 1800, 1800, 1800, 1600, 1800 | 6,328,064 | 1 / 1 / 0 | 1 / 0 / 0 |

Out-of-distribution text: 55,770 characters of Pride and Prejudice after the rule of method.md section 18.1; 4,564 of 688,551 characters of the novel (0.66%) lay outside the vocabulary and were dropped.

Search (seed 100, validation BPC at the best checkpoint), in grid order:

- EBT, S2, 4 inference steps: learning_rate 0.0012, step_size 0.5; 2.1017
- EBT, S2, 4 inference steps: learning_rate 0.0012, step_size 5.0; 2.1185
- EBT, S2, 4 inference steps: learning_rate 0.0012, step_size 50.0; 2.1008 (chosen)
- EBT, S2, 4 inference steps: learning_rate 0.003, step_size 50.0; 2.1252
- Causal transformer: learning_rate 0.0003; 2.0759
- Causal transformer: learning_rate 0.001; 2.0841
- Causal transformer: learning_rate 0.003; 2.0547 (chosen)
- Causal transformer: learning_rate 0.01; 2.2879
