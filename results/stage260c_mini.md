# Stage 260c open-text gate (paired same-line contrast)

**OPEN_GATE3_NO** slots=88 eval=24 on / 24 off

- gate: on-tape **0.208** | off-tape **0.149** | prose 0.036
- AUC vs prose **0.812**, vs off-tape entities **0.608**
- slot deleted -> gate 0.208 -> **0.206**; shuffled keys AUC 0.812
- paired gap (on − delete) **0.0016** | gate_reads_tape=True
- **features_move=True** | probe |d max|=0.0833 |d margin12|=0.0860 |d cov|=0.0000 | gold top1 0.67 → 0.00
- false fire on prose: 0.024 over 584 positions
