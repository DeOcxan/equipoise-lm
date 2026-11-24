
Measurement of the current model (sections 4.1 and 18.3, corrections 13.1 to 13.4), 2026-11-21, no other training process running. Before the measurements: `memory_pressure` reported 61% free, that is 9.76 GiB of 16 GiB, so the memory limit is 7.32 GiB; load averages { 2.42 2.52 2.91 }. Same procedure as above; the measurements follow the order of item 3 and stop at the first size and setting that meet items 2 and 4. A process that ran out of MPS memory, was killed by the system or was ended by a Metal out-of-memory abort is recorded as out of memory; one that ran longer than 1 800 s is stopped and recorded as over the time limit. Measured and applied with `python -m tools.text_budget` (`tools/text_budget.py`, one process per setting with `tools/text_step_measurement.py`).

| size | S | batch | model | layout | parameters | step k=2 (s) | step k=3 (s) | step k=4 (s) | step (s) | peak k=4 (GiB) | validation eval (s) | generation (s/char) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 384 × 6 | 128 | 32 | EBT | split | 14,211,521 | 1.9872 | 2.3218 | 2.7650 | 2.3580 | 10.13 | 16.20 | 0.1055 |
| 256 × 6 | 256 | 32 | EBT | split | 6,328,641 | 2.2643 | 2.8169 | 3.4150 | 2.8321 | 11.02 | 9.99 | 0.1059 |
| 256 × 6 | 128 | 64 | EBT | split | 6,328,641 | 2.0599 | 2.5765 | 3.1018 | 2.5794 | 10.25 | 8.78 | 0.0700 |
| 256 × 6 | 128 | 32 | EBT | split | 6,328,641 | 1.0127 | 1.2627 | 1.5188 | 1.2648 | 5.21 | 8.90 | 0.0484 |
| 256 × 6 | 128 | 32 | EBT | full | 6,328,641 | 1.6077 | 1.9714 | 2.3355 | 1.9715 | 8.21 | 12.97 | 0.0724 |
| 256 × 6 | 128 | 32 | transformer | - | 6,328,064 | | | | 0.1255 | 1.20 | 0.56 | |

Out of memory: 256x6 S128 B64 ebt full; 256x6 S256 B32 ebt full; 256x6 S256 B64 ebt split; 384x6 S128 B32 ebt full; 384x6 S128 B64 ebt split; 384x6 S256 B32 ebt split; 384x6 S256 B64 ebt split.

Application of section 18.7 (item 3):

- 384 × 6, S 256, batch 64: the EBT ran out of memory (split layout).
- 384 × 6, S 256, batch 32: the EBT ran out of memory (split layout).
- 384 × 6, S 128, batch 64: the EBT ran out of memory (split layout).
- 384 × 6, S 128, batch 32: faster layout split (k = 4: split 2.765 s, full out of memory); 10.13 GiB at k = 4, above 7.32 GiB.
- 256 × 6, S 256, batch 64: the EBT ran out of memory (split layout).
- 256 × 6, S 256, batch 32: faster layout split (k = 4: split 3.415 s, full out of memory); 11.02 GiB at k = 4, above 7.32 GiB.
- 256 × 6, S 128, batch 64: faster layout split (k = 4: split 3.102 s, full out of memory); 10.25 GiB at k = 4, above 7.32 GiB.
- 256 × 6, S 128, batch 32: faster layout split (k = 4: split 1.519 s, full 2.336 s); 5.21 GiB at k = 4, within 7.32 GiB; mean EBT step 1.265 s, baseline 0.126 s; T = 1 000: search 1.65 h, finals 2.10 h; T = 2 000: search 3.19 h, finals 4.03 h; T = 5 000: search 7.83 h, finals 9.83 h.

Chosen: width 256, depth 6 (6 328 641 EBT and 6 328 064 baseline parameters), S = 128, batch 32, T = 2 000, split layout, validation every 200 steps; projected search 3.19 h and final runs 4.03 h, each within the v1.0 budget of 8 h per stage.

Departures from Table D.2: S 128 and batch 32 instead of S 256 and batch 64, forced by the rows above; a size smaller than the authors' xxs (384 × 6), forced by the rows above.
