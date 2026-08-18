# Stage 286 — results and metrics

The tape is its own exam: leave-one-out cloze on mentions; votes/return demoted to
examiners; UNKNOWN is a candidate. This note collects the measured arms in investigation
order — not a plan.

## Verdict in one page

- **Reading text through the trunk failed as a mind channel on this scale.** Full-width and
  rank-8 reading memorized the train tape (control AUC ~0.99) and collapsed or inverted on
  held-out. `--read-anon` removed the inversion but did not beat features-only (`nohid`).
- **`nohid` (nine count features) is the only reading-free arm that reached `EVIDENCE_OK`**
  under fixed examiners (matched coverage + honesty AUC + survives lie).
- **`--mind relational` is the stage result.** Structure-only graph (ranks, no text/fp
  coordinates): train AUC fell from ~1.0 to ~0.84; matched coverage tied/beat the count
  judges; on **duplicated lie** votes go to **0.000** while the relational mind stays
  positive — a capability counters do not have.
- Still open / failed on relational: contested weighing (almost silent), honesty
  (`unknown_margin_auc` ~0.49), beats-silence. Ink slices did not help the dup axis;
  `ink_coherence` did not carry.

## Arm table (held-out LOO)

| arm | overall | model cov/acc | matched @votes (n) | matched @return (n) | honesty AUC | dup model/votes |
|---|---|---:|---:|---:|---:|---:|
| Old examiners — features-only (nohid) | **EVIDENCE_PARTIAL** | 0.557/0.641 | 0.431 vs 0.431 | 1.000 vs 0.889 | 0.412 | — |
| Old examiners — with reading (hid) | **EVIDENCE_NO** | 0.571/0.325 | 0.379 vs 0.431 | 0.222 vs 0.889 | 0.259 | — |
| Reexam nohid (fixed examiners) | **EVIDENCE_OK** | 0.357/0.840 | 0.517 vs 0.431 | 1.000 vs 0.889 | 0.560 | — |
| Reexam frozen-trunk (full read) | **EVIDENCE_NO** | 0.757/0.377 | 0.414 vs 0.431 (n=58) | 0.556 vs 0.889 (n=9) | 0.416 | — |
| Frozen + read-rank 8 | **EVIDENCE_NO** | 0.729/0.353 | 0.328 vs 0.431 (n=58) | 0.667 vs 0.889 (n=9) | 0.302 | — |
| Frozen + read-rank 8 + read-anon | **EVIDENCE_NO** | 0.443/0.613 | 0.431 vs 0.431 (n=58) | 0.444 vs 0.889 (n=9) | 0.579 | — |
| Relational mind, first dup x3 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.482 | 0.773/0.000 (K=3) |
| Relational + margin gate (dup x3) | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.773/0.000 (K=3) |
| Relational + ink-slices (dup x3) | **EVIDENCE_PARTIAL** | 0.743/0.481 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.515 | 0.333/0.000 (K=3) |
| Relational lie-dup ladder K=1 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | lie 1.000/0.462 |
| Relational lie-dup ladder K=2 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.667/0.333 (K=2) |
| Relational lie-dup ladder K=3 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.773/0.000 (K=3) |
| Relational lie-dup ladder K=5 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.486/0.016 (K=5) |

Honesty column: older arms report `unknown_auc`; from the margin-gate restart onward,
`unknown_margin_auc` (logit margin, not diluted by |cands|).

## Gates by late arm

| arm | learns | abstain | survives_lie | contested | dup_lie | beats_silence |
|---|---|---|---|---|---|---|
| Relational first dup x3 | Y | N | Y | N | Y | N |
| Relational + margin gate | Y | N | Y | N | Y | N |
| Relational + ink-slices | Y | N | Y | N | Y | N |
| Ladder K=1 | Y | N | Y | N | — | N |
| Ladder K=2 | Y | N | Y | N | Y | N |
| Ladder K=3 | Y | N | Y | N | Y | N |
| Ladder K=5 | Y | N | Y | N | Y | N |

Also true on all late arms: `G_arc_enc_frozen`, `G_task_exists`, `G_judges_non_vacuous`.

## Lie-dup ladder (relational, no ink-slices)

Same mind class; K = how many times the forged mention is duplicated into the evidence.

| K | overall | model acc | votes acc | G_survives_duplicated_lie | ink_coherence sep (dup) | margin AUC |
|---:|---|---:|---:|---|---:|---:|
| 1 | EVIDENCE_PARTIAL | 1.000 | 0.462 | — (single lie, not dup exam) | — | 0.486 |
| 2 | EVIDENCE_PARTIAL | 0.667 | 0.333 | Y | 0.024 | 0.486 |
| 3 | EVIDENCE_PARTIAL | 0.773 | 0.000 | Y | 0.058 | 0.486 |
| 5 | EVIDENCE_PARTIAL | 0.486 | 0.016 | Y | -0.011 | 0.486 |

Votes fall toward zero as K grows; the relational mind peaks around K=3 and still beats votes
at K=5 with lower absolute accuracy.

## Reading / capacity / identity (ablation chain)

| step | artifact | held-out | train control AUC | note |
|---|---|---|---:|---|
| nohid reexam | `stage286_decision_reexam_nohid_nohid.json` | **EVIDENCE_OK** | 0.809 | features only |
| frozen full read | `stage286_decision_reexam_frozen_frozen.json` | **EVIDENCE_NO** | 0.986 | collapse; trunk-leak hypothesis (a) dead |
| read-rank 8 | `stage286_decision_r8_frozen_r8.json` | **EVIDENCE_NO** | 0.996 | capacity cut; train fit unchanged -> not (b) |
| read-anon | `stage286_decision_anon_frozen_r8_anon.json` | **EVIDENCE_NO** | 0.997 | inversion gone; no beat of nohid -> (c) identity |

## Relational mind — headline numbers (margin arm, K=3)

Source: `stage286_decision_rel_margin_rel_dup3.json`

- params: **4417** (`mind=relational`)
- train control AUC: **0.845** (was ~0.997 on reading arms)
- matched @votes: **0.431** vs 0.431 (n=58)
- matched @return: **1.000** vs 0.889 (n=9)
- dup x3: model **0.773** vs votes **0.000**
- contested: answer rate **0.044**, acc@votes-coverage contested **0.121**
  (votes 0.121, random floor 0.268)
- `unknown_margin_auc`: **0.486** (z=-0.162)
- `ink_coherence` LOO sep: nan (too few answered); on dup sep **0.058**

## Ink-slices vs plain relational (K=3)

| | plain relational | +ink-slices |
|---|---:|---:|
| overall | EVIDENCE_PARTIAL | EVIDENCE_PARTIAL |
| matched @return | 1.000 | 1.000 |
| dup x3 model/votes | 0.773/0.000 | 0.333/0.000 |
| contested answer rate | 0.044 | 0.600 |
| contested acc@votes cov | 0.121 | 0.121 |
| ink_coherence sep (dup) | 0.058 | -0.024 |

Slices raised contested coverage without accuracy and diluted the dup win. Observer still not
bearing.

## Notes on metrics

- **Matched coverage:** rank by the mind's confidence, cut at the *judge's* coverage, compare
  accuracy there — avoids rewarding silence against a selective judge.
- **Honesty:** `unknown_margin_auc` = AUC of logit margin (UNKNOWN − best candidate); not
  diluted by `|cands|` like mean P(UNKNOWN).
- **Dup-lie:** forge one mention and repeat it K times; votes scale in K by construction;
  relational edge ranks can mark duplicates.
- **ink_coherence:** observer only (correct vs wrong coalition ctx-cos); never in the loss.

## Artifact index

| arm | decision JSON | mini |
|---|---|---|
| Old nohid | [`stage286_decision_nohid.json`](stage286_decision_nohid.json) | [`stage286_mini_nohid.md`](stage286_mini_nohid.md) |
| Old hid | [`stage286_decision_hid.json`](stage286_decision_hid.json) | [`stage286_mini_hid.md`](stage286_mini_hid.md) |
| Reexam nohid | [`stage286_decision_reexam_nohid_nohid.json`](stage286_decision_reexam_nohid_nohid.json) | [`stage286_mini_reexam_nohid_nohid.md`](stage286_mini_reexam_nohid_nohid.md) |
| Reexam frozen | [`stage286_decision_reexam_frozen_frozen.json`](stage286_decision_reexam_frozen_frozen.json) | [`stage286_mini_reexam_frozen_frozen.md`](stage286_mini_reexam_frozen_frozen.md) |
| read-rank 8 | [`stage286_decision_r8_frozen_r8.json`](stage286_decision_r8_frozen_r8.json) | [`stage286_mini_r8_frozen_r8.md`](stage286_mini_r8_frozen_r8.md) |
| read-anon | [`stage286_decision_anon_frozen_r8_anon.json`](stage286_decision_anon_frozen_r8_anon.json) | [`stage286_mini_anon_frozen_r8_anon.md`](stage286_mini_anon_frozen_r8_anon.md) |
| Relational first | [`stage286_decision_rel_rel_dup3.json`](stage286_decision_rel_rel_dup3.json) | [`stage286_mini_rel_rel_dup3.md`](stage286_mini_rel_rel_dup3.md) |
| Relational margin | [`stage286_decision_rel_margin_rel_dup3.json`](stage286_decision_rel_margin_rel_dup3.json) | [`stage286_mini_rel_margin_rel_dup3.md`](stage286_mini_rel_margin_rel_dup3.md) |
| Ink-slices | [`stage286_decision_ink_slices_rel_dup3_slices.json`](stage286_decision_ink_slices_rel_dup3_slices.json) | [`stage286_mini_ink_slices_rel_dup3_slices.md`](stage286_mini_ink_slices_rel_dup3_slices.md) |
| Ladder K=1 | [`stage286_decision_dup1_rel.json`](stage286_decision_dup1_rel.json) | [`stage286_mini_dup1_rel.md`](stage286_mini_dup1_rel.md) |
| Ladder K=2 | [`stage286_decision_dup2_rel_dup2.json`](stage286_decision_dup2_rel_dup2.json) | [`stage286_mini_dup2_rel_dup2.md`](stage286_mini_dup2_rel_dup2.md) |
| Ladder K=3 | [`stage286_decision_dup3_rel_dup3.json`](stage286_decision_dup3_rel_dup3.json) | [`stage286_mini_dup3_rel_dup3.md`](stage286_mini_dup3_rel_dup3.md) |
| Ladder K=5 | [`stage286_decision_dup5_rel_dup5.json`](stage286_decision_dup5_rel_dup5.json) | [`stage286_mini_dup5_rel_dup5.md`](stage286_mini_dup5_rel_dup5.md) |

Code: `_stage286_evidence.py`. Handoff snapshots under `results/_stage286_handoff.md`.
Queue log for the final margin/slices/ladder pass: `results/_queue_286_rel13.out`.
