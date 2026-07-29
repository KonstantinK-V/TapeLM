# Stage191 night report (2026-07-28T23:13:51.103197+00:00)

**Verdicts:** NIGHT_PARITY_HELD, NIGHT_MEANING_MOVES

## p0
```json
{
  "counts": {
    "next_tok": 300,
    "entity": 150,
    "ood": 100
  },
  "unigram": {
    "next_tok_acc": 0.27,
    "next_tok_n": 300,
    "entity_acc": 0.22666666666666666,
    "entity_n": 150,
    "ood_acc": 0.18,
    "ood_n": 100
  },
  "random": {
    "next_tok_acc": 0.24,
    "next_tok_n": 300,
    "entity_acc": 0.2733333333333333,
    "entity_n": 150,
    "ood_acc": 0.19,
    "ood_n": 100
  },
  "docs": 202029,
  "tokens": 40167271,
  "charset": 977,
  "timestamp": "2026-07-28T21:04:18.300234+00:00"
}
```
## p1
```json
{
  "train": {
    "best_mid": 0.825,
    "best_step": 10000,
    "ce": 4.108392025891041,
    "wall_s": 4265.183955907822
  },
  "exam": {
    "next_tok_acc": 0.8666666666666667,
    "entity_acc": 0.2733333333333333,
    "ood_acc": 0.2,
    "next_tok_n": 300,
    "entity_n": 150,
    "ood_n": 100
  },
  "params_m": 7.56429,
  "timestamp": "2026-07-28T22:17:12.046231+00:00"
}
```
## p2
```json
{
  "best_mid": 0.825,
  "best_step": 7500,
  "exam": {
    "next_tok_acc": 0.8433333333333334,
    "entity_acc": 0.3,
    "ood_acc": 0.2,
    "next_tok_n": 300,
    "entity_n": 150,
    "ood_n": 100
  },
  "params_m": 5.804032,
  "timestamp": "2026-07-28T22:22:59.420879+00:00"
}
```
## p3
```json
{
  "train": {
    "best_mid": 0.8375,
    "best_step": 7500,
    "ce": 4.323549803586784,
    "wall_s": 2837.6578850746155
  },
  "exam": {
    "next_tok_acc": 0.8533333333333334,
    "entity_acc": 0.29333333333333333,
    "ood_acc": 0.24,
    "next_tok_n": 300,
    "entity_n": 150,
    "ood_n": 100
  },
  "g3": {
    "entropy_real": 4.946852380037308,
    "entropy_fake": 4.99720795750618,
    "surprise_real": 0.024830721679609268,
    "surprise_fake": 0.022052408987656237,
    "entropy_ok": true,
    "surprise_ok": false
  },
  "timestamp": "2026-07-28T23:12:05.074827+00:00"
}
```
## p4
```json
{
  "p1_curve": {
    "gateB": {
      "para": 0.9892294454574585,
      "hard": 0.9940304458141327,
      "gap": 0.00480100035667419
    },
    "doclink": 0.0625
  },
  "p3_rarity": {
    "gateB": {
      "para": 0.9985147213935852,
      "hard": 0.999459907412529,
      "gap": 0.0009451860189437777
    },
    "doclink": 0.2125
  },
  "p2_gpt": {
    "gateB": {
      "para": 0.8192921948432922,
      "hard": 0.9372172728180885,
      "gap": 0.11792507797479634
    },
    "doclink": 0.6625
  },
  "old_187_d128_2M": {
    "gateB": {
      "para": 0.9787734341621399,
      "hard": 0.995217353105545,
      "gap": 0.016443918943405134
    },
    "doclink": 0.2875
  },
  "timestamp": "2026-07-28T23:12:52.895806+00:00"
}
```