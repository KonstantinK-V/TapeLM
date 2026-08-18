# Stage 287 is there a second layer of ink (span 3)

**SECOND_LAYER_REDUNDANT** · trained parameters **0**

| channel | AUC same-vs-different | sigma |
|---|---:|---:|
| lower (today's ctx_fp) | 0.935 | +15.46 |
| upper (adjacent spans) | 0.917 | +14.81 |
| upper given lower | 0.467 | -1.17 |
| null: label permutation | 0.500 | -0.01 |
| null: redundant channel | 0.499 | -0.03 |

- shuffling words: lower moves by 4.47e-08 (must be ~0), upper stays at cos 0.9866 -> only 0.0134 of it depends on word order
- Spearman between the two cosines: 0.9451

## Gates

- G_lower_is_order_blind: **True**
- G_residual_test_can_say_no: **True**
- G_upper_sees_word_order: **True**
- G_channels_are_distinct: **True**
- G_upper_carries_unique_signal: **False**
- G_upper_beats_lower_alone: **False**
