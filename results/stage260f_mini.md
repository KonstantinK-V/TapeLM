# Stage 260f open-text gate (headline **feat_only**, random-key control)

**OPEN_GATE6_OK** slots=420 eval=120 on / 120 off

- headline **feat_only**: on-tape **0.232** | off-tape **0.032** | prose 0.025
- AUC vs prose **0.868**, vs off-tape entities **0.844**
- slot deleted -> gate 0.232 -> **0.079**; random-key gate **0.000** (ratio **867.7x**); AUC random 0.746
- paired win-rate **0.672** vs gold top1 **0.738** (n_pairs=61)
- feature probe: |d max|=0.0848 |d margin12|=0.0890 | gold top1 0.74 -> 0.00
- paired win-rate: h+feat **0.639** | feat_only **0.672** | h_only 0.000
- paired logit gap: 3.747 / 2.698 / 0.000 (prob gaps 0.0873 / 0.1535 / 0.0000)
- AUC vs prose: 0.907 / 0.868 / 0.854
- G_h_only_flat: True | G_tape_causal (abs gate): True
- false fire on prose: 0.019 over 3129 positions
