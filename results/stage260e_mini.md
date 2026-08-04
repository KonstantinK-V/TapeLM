# Stage 260e open-text gate (paired win-rate + logit gap)

**OPEN_GATE5_NO** slots=88 eval=24 on / 24 off

- gate: on-tape **0.138** | off-tape **0.005** | prose 0.038
- AUC vs prose **0.792**, vs off-tape entities **0.630**
- slot deleted -> gate 0.138 -> **0.069**; shuffled keys AUC 0.792
- paired gap prob-space **0.0690** | win-rate **0.857** | logit gap **2.476**
- feature probe: |d max|=0.0933 |d margin12|=0.0773 | gold top1 0.86 -> 0.00
- paired win-rate: h+feat **0.857** | feat_only **0.905** | h_only 0.000
- paired logit gap: 2.476 / 2.373 / 0.000 (prob gaps 0.0690 / 0.1012 / 0.0000)
- AUC vs prose: 0.792 / 0.815 / 0.736
- G_h_only_flat: True
- false fire on prose: 0.035 over 681 positions
