# Stage 287 is there a second layer of ink (span 2)

**MEASUREMENT_BROKEN** · trained parameters **0**

| channel | AUC same-vs-different | sigma |
|---|---:|---:|
| lower (today's ctx_fp) | 0.935 | +15.46 |
| upper (adjacent spans) | 0.920 | +14.93 |
| upper given lower | 0.494 | -0.22 |
| null: lower given lower | 0.163 | -11.96 |

- shuffling words: lower moves by 4.47e-08 (must be ~0), upper stays at cos 0.9871 (must be < 1)
- Spearman between the two cosines: 0.9590

## Gates

- G_lower_is_order_blind: **True**
- G_residual_test_can_say_no: **False**
- G_upper_sees_word_order: **True**
- G_channels_are_distinct: **True**
- G_upper_carries_unique_signal: **False**
- G_upper_beats_lower_alone: **False**
