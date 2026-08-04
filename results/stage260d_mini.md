# Stage 260d open-text gate (feature fix + input ablation)

**OPEN_GATE4_NO** slots=88 eval=24 on / 24 off

- gate: on-tape **0.109** | off-tape **0.000** | prose 0.012
- AUC vs prose **0.780**, vs off-tape entities **0.689**
- slot deleted -> gate 0.109 -> **0.071**; shuffled keys AUC 0.780
- paired gap (on - delete) **0.0376**
- feature probe, slot present vs dropped: |d max|=0.1014 |d margin12|=0.1102 |d cov|=0.0000 | gold top1 0.67 -> 0.00
- paired gap by input: h+feat **0.0376** | **feat_only 0.0064** | h_only 0.0000
- AUC vs prose by input: 0.780 / 0.794 / 0.670
- false fire on prose: 0.012 over 582 positions
