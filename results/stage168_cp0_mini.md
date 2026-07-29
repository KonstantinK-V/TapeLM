# Stage168 CP0 — infra mini report

**Verdict:** `CP0_READY`

- Rich corpus: 500001 phrases, ~4002173 tokens streamed
- Case tags: {'title': 597310, 'lower': 2669802, 'other': 117957, 'punct': 568435, 'upper': 43931, 'mixed': 4738}
- Top punct: [(',', 204214), ('.', 166459), ('"', 54294), ("'", 36809), ('-', 34510), (')', 23008), ('(', 22990), (';', 8134)]
- Battery n=8000, non-majority gold=4922, majority baseline≈38.5%
- CP3 cue = separate dual-channel embedding (not special token)
- Next: CP1 plain CE baseline (orthography alone)
