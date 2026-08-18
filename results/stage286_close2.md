# Stage 286 — results and metrics
The tape is its own exam: leave-one-out cloze on mentions; votes/return demoted to examiners; UNKNOWN is a candidate. This note collects the measured arms in order of the investigation — not a plan.
## Verdict in one page
- **Reading text through the trunk failed as a mind channel on this scale.** Full-width and rank-8 reading memorized the train tape (control AUC ~0.99) and collapsed or inverted on held-out. `--read-anon` removed the inversion but did not beat features-only (`nohid`).
- **`nohid` (nine count features) is the only reading-free arm that reached `EVIDENCE_OK`** under fixed examiners (matched coverage + honesty AUC + survives lie).
- **`--mind relational` is the stage result.** Structure-only graph (ranks, no text/fp coordinates): train AUC fell from ~1.0 → ~0.84; matched coverage tied/beat the count judges; on **duplicated lie** votes go to **0.000** while the relational mind stays positive — a capability counters do not have.
- Still open / failed on relational: contested weighing (almost silent), honesty (`unknown_margin_auc` ~0.49), beats-silence. Ink slices did not help the dup axis; `ink_coherence` did not carry.
## Arm table (held-out LOO)
| arm | overall | model cov/acc | matched @votes (n) | matched @return (n) | honesty AUC | dup×K model/votes |
|---|---|---:|---:|---:|---:|---:|
| Old examiners — features-only (nohid) | **EVIDENCE_PARTIAL** | 0.557/0.641 | 0.431 vs 0.431 (n=—) | 1.000 vs 0.889 (n=—) | 0.412 | — |
| Old examiners — with reading (hid) | **EVIDENCE_NO** | 0.571/0.325 | 0.379 vs 0.431 (n=—) | 0.222 vs 0.889 (n=—) | 0.259 | — |
| Reexam nohid (fixed examiners) | **EVIDENCE_OK** | 0.357/0.840 | 0.517 vs 0.431 (n=—) | 1.000 vs 0.889 (n=—) | 0.560 | — |
| Reexam frozen-trunk (full read) | **EVIDENCE_NO** | 0.757/0.377 | 0.414 vs 0.431 (n=58) | 0.556 vs 0.889 (n=9) | 0.416 | — |
| Frozen + read-rank 8 | **EVIDENCE_NO** | 0.729/0.353 | 0.328 vs 0.431 (n=58) | 0.667 vs 0.889 (n=9) | 0.302 | — |
| Frozen + read-rank 8 + read-anon | **EVIDENCE_NO** | 0.443/0.613 | 0.431 vs 0.431 (n=58) | 0.444 vs 0.889 (n=9) | 0.579 | — |
| Relational mind, first dup×3 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.482 | 0.773/0.000 (K=3) |
| Relational + margin gate (dup×3) | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.773/0.000 (K=3) |
| Relational + ink-slices (dup×3) | **EVIDENCE_PARTIAL** | 0.743/0.481 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.515 | 0.333/0.000 (K=3) |
| Relational lie-dup ladder K=1 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | lie 1.000/0.462 |
| Relational lie-dup ladder K=2 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.667/0.333 (K=2) |
| Relational lie-dup ladder K=3 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.773/0.000 (K=3) |
| Relational lie-dup ladder K=5 | **EVIDENCE_PARTIAL** | 0.386/0.778 | 0.431 vs 0.431 (n=58) | 1.000 vs 0.889 (n=9) | 0.486 | 0.486/0.016 (K=5) |

## Gates by late arm
| arm | gates |
|---|---|
| Relational mind, first dup×3 | learns_evidence=✓, abstains_unknowable=✗, survives_lie=✓, weighs_contested=✗, survives_duplicated_lie=✓, beats_silence=✗, task_exists=✓, judges_non_vacuous=✓ |
| Relational + margin gate (dup×3) | learns_evidence=✓, abstains_unknowable=✗, survives_lie=✓, weighs_contested=✗, survives_duplicated_lie=✓, beats_silence=✗, task_exists=✓, judges_non_vacuous=✓ |
| Relational + ink-slices (dup×3) | learns_evidence=✓, abstains_unknowable=✗, survives_lie=✓, weighs_contested=✗, survives_duplicated_lie=✓, beats_silence=✗, task_exists=✓, judges_non_vacuous=✓ |
| Relational lie-dup ladder K=1 | learns_evidence=✓, abstains_unknowable=✗, survives_lie=✓, weighs_contested=✗, survives_duplicated_lie=✗, beats_silence=✗, task_exists=✓, judges_non_vacuous=✓ |
| Relational lie-dup ladder K=2 | learns_evidence=✓, abstains_unknowable=✗, survives_lie=✓, weighs_contested=✗, survives_duplicated_lie=✓, beats_silence=✗, task_exists=✓, judges_non_vacuous=✓ |
| Relational lie-dup ladder K=3 | learns_evidence=✓, abstains_unknowable=✗, survives_lie=✓, weighs_contested=✗, survives_duplicated_lie=✓, beats_silence=✗, task_exists=✓, judges_non_vacuous=✓ |
| Relational lie-dup ladder K=5 | learns_evidence=✓, abstains_unknowable=✗, survives_lie=✓, weighs_contested=✗, survives_duplicated_lie=✓, beats_silence=✗, task_exists=✓, judges_non_vacuous=✓ |

## Lie-dup ladder (relational, no ink-slices)
Same mind class; K = how many times the forged mention is duplicated into the evidence.

| K | overall | model acc | votes acc | G_survives_duplicated_lie | ink_coherence sep (dup) | margin AUC |
|---:|---|---:|---:|---|---:|---:|
| 1 | EVIDENCE_PARTIAL | 1.000 | 0.462 | — | — | 0.486 |
| 2 | EVIDENCE_PARTIAL | 0.667 | 0.333 | ✓ | 0.024 | 0.486 |
| 3 | EVIDENCE_PARTIAL | 0.773 | 0.000 | ✓ | 0.058 | 0.486 |
| 5 | EVIDENCE_PARTIAL | 0.486 | 0.016 | ✓ | -0.011 | 0.486 |

## Reading / capacity / identity (ablation chain)
| step | what | held-out | train control AUC | note |
|---|---|---|---:|---|
| nohid reexam | `stage286_decision_reexam_nohid_nohid.json` | **EVIDENCE_OK** | 0.809 | EVIDENCE_OK — features only |
| frozen full read | `stage286_decision_reexam_frozen_frozen.json` | **EVIDENCE_NO** | 0.986 | collapse; (a) trunk-leak dead |
| read-rank 8 | `stage286_decision_r8_frozen_r8.json` | **EVIDENCE_NO** | 0.996 | capacity cut; train fit unchanged → not (b) |
| read-anon | `stage286_decision_anon_frozen_r8_anon.json` | **EVIDENCE_NO** | 0.997 | inversion gone; no beat of nohid → (c) |

## Relational mind — headline numbers (margin arm, K=3)
- params: `4417` (mind=`relational`)
- train control AUC: **0.845** (was ~0.997 on reading arms)
- matched @votes: **0.431** vs 0.431 (n=58)
- matched @return: **1.000** vs 0.889 (n=9)
- dup×3: model **0.773** vs votes **0.000**
- contested: answer rate **0.044**, acc@votes-coverage contested **0.121** (votes 0.121, random floor 0.268)
- `unknown_margin_auc`: **0.486** (z=-0.162)
- `ink_coherence` LOO sep: **nan**; on dup sep **0.058**

## Ink-slices vs plain relational (K=3)
| | plain relational | +ink-slices |
|---|---:|---:|
| overall | EVIDENCE_PARTIAL | EVIDENCE_PARTIAL |
| matched @return | 1.000 | 1.000 |
| dup×3 model/votes | 0.773/0.000 | 0.333/0.000 |
| contested answer rate | 0.044 | 0.600 |
| ink_coherence sep (dup) | 0.058 | -0.024 |

## Artifact index
- Old examiners — features-only (nohid): [`stage286_decision_nohid.json`](stage286_decision_nohid.json) · mini [`stage286_mini_nohid.md`](stage286_mini_nohid.md)
- Old examiners — with reading (hid): [`stage286_decision_hid.json`](stage286_decision_hid.json) · mini [`stage286_mini_hid.md`](stage286_mini_hid.md)
- Reexam nohid (fixed examiners): [`stage286_decision_reexam_nohid_nohid.json`](stage286_decision_reexam_nohid_nohid.json) · mini [`stage286_mini_reexam_nohid_nohid.md`](stage286_mini_reexam_nohid_nohid.md)
- Reexam frozen-trunk (full read): [`stage286_decision_reexam_frozen_frozen.json`](stage286_decision_reexam_frozen_frozen.json) · mini [`stage286_mini_reexam_frozen_frozen.md`](stage286_mini_reexam_frozen_frozen.md)
- Frozen + read-rank 8: [`stage286_decision_r8_frozen_r8.json`](stage286_decision_r8_frozen_r8.json) · mini [`stage286_mini_r8_frozen_r8.md`](stage286_mini_r8_frozen_r8.md)
- Frozen + read-rank 8 + read-anon: [`stage286_decision_anon_frozen_r8_anon.json`](stage286_decision_anon_frozen_r8_anon.json) · mini [`stage286_mini_anon_frozen_r8_anon.md`](stage286_mini_anon_frozen_r8_anon.md)
- Relational mind, first dup×3: [`stage286_decision_rel_rel_dup3.json`](stage286_decision_rel_rel_dup3.json) · mini [`stage286_mini_rel_rel_dup3.md`](stage286_mini_rel_rel_dup3.md)
- Relational + margin gate (dup×3): [`stage286_decision_rel_margin_rel_dup3.json`](stage286_decision_rel_margin_rel_dup3.json) · mini [`stage286_mini_rel_margin_rel_dup3.md`](stage286_mini_rel_margin_rel_dup3.md)
- Relational + ink-slices (dup×3): [`stage286_decision_ink_slices_rel_dup3_slices.json`](stage286_decision_ink_slices_rel_dup3_slices.json) · mini [`stage286_mini_ink_slices_rel_dup3_slices.md`](stage286_mini_ink_slices_rel_dup3_slices.md)
- Relational lie-dup ladder K=1: [`stage286_decision_dup1_rel.json`](stage286_decision_dup1_rel.json) · mini [`stage286_mini_dup1_rel.md`](stage286_mini_dup1_rel.md)
- Relational lie-dup ladder K=2: [`stage286_decision_dup2_rel_dup2.json`](stage286_decision_dup2_rel_dup2.json) · mini [`stage286_mini_dup2_rel_dup2.md`](stage286_mini_dup2_rel_dup2.md)
- Relational lie-dup ladder K=3: [`stage286_decision_dup3_rel_dup3.json`](stage286_decision_dup3_rel_dup3.json) · mini [`stage286_mini_dup3_rel_dup3.md`](stage286_mini_dup3_rel_dup3.md)
- Relational lie-dup ladder K=5: [`stage286_decision_dup5_rel_dup5.json`](stage286_decision_dup5_rel_dup5.json) · mini [`stage286_mini_dup5_rel_dup5.md`](stage286_mini_dup5_rel_dup5.md)

## Notes on metrics
- **Matched coverage:** rank items by the mind's confidence, cut at the *judge's* coverage, compare accuracy there. Avoids rewarding silence vs a selective judge.
- **Honesty:** `unknown_margin_auc` = AUC of logit margin(UNKNOWN − best cand); not diluted by `|cands|` like mean P(UNKNOWN).
- **Dup-lie:** forge one mention and repeat it K times; votes scale in K by construction; relational edge ranks can mark duplicates.
- **ink_coherence:** observer only (correct vs wrong coalition ctx-cos); not in the loss.
