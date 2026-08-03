# Stage 244 forget cleanliness

**FORGET_CLEAN_OK**

```json
{
  "stage": 244,
  "overall": "FORGET_CLEAN_OK",
  "gates": {
    "G_tape_forget_to_chance": true,
    "G_tape_no_collateral": true,
    "G_gpt_shows_collateral": true,
    "G_gpt_forgot_some": true
  },
  "tape": {
    "tgt_before": 0.875,
    "tgt_after": 0.0,
    "ret_before": 1.0,
    "ret_after": 1.0,
    "next_tok_before": 0.825,
    "next_tok_after": 0.825
  },
  "gpt": {
    "tgt_before": 0.6875,
    "tgt_after": 0.25,
    "ret_before": 0.7083333333333334,
    "ret_after": 0.2916666666666667,
    "next_tok_before": 0.7833333333333333,
    "next_tok_after": 0.55,
    "unlearn_steps": 60,
    "memorize_steps": 300
  },
  "note": "Capability vs parametric GPT; architectural vs GPT+RAG index delete.",
  "timestamp": "2026-07-31T05:06:31.256316+00:00",
  "wall_s": 70.13645386695862
}
```
