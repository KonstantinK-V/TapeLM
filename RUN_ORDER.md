# The queue, in order — what to run, what number decides it, when to stop

Each block names **one** number that decides it and what makes the run worth nothing. These
reports carry twenty fields and one that matters; reading the wrong one has already cost this
project two days.

All timings assume the exact speedups in `_tape_speed.py`, which are on by default. 289a's full
run went from ~20 hours to **411 seconds** with them.

---

## Files

| file | what it is |
|---|---|
| `_tape_speed.py` | the three exact speedups, shared. Not a stage — it holds no science |
| `_check289_fast_grouping.py` | 40-second proof that the speedups change nothing |
| `_stage286_evidence.py` | evidence / lying tape / dup — now speeded up |
| `_stage288_repair.py` | detect and repair a broken tape — now speeded up |
| `_stage289a_presupposition.py` | should this question be answered at all |
| `_stage289_derivation.py` | the answer becomes a derivation (COUNT / COMPARE) |
| `_stage289c_audit.py` | the unprompted audit — imports 289's mind and questions |

Import order is fixed and acyclic: `_tape_speed` ← everything; `289a` ← `289` ← `289c`.
Replace all seven together; `289c` will not run against an old `289`.

## Step 0 — once per machine, and after any edit to `_stage279_write_decision.py`

    python _check289_fast_grouping.py

Must print **IDENTICAL**. If it does not, add `--no-fast-grouping` to everything below (or
`--no-speedups` for 286/288) and tell me — the numbers stay right, the runs just get slow again.

---

## 1. 289a re-run, AUC gate  (~7 min)

    python -u _stage289a_presupposition.py --train-steps 6000 --addresses 1200 \
      --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150auc

**Decides:** `held_out.blind_pair.auc_z` against **1.645**.
Not `model_accuracy` — training draws the four verdicts uniformly and the examiner sees the
tape's natural 76/24, so an argmax against a majority floor charges the mind for a prior shift
the examiner introduced. Already measured on the previous run: TPR 0.700 vs FPR 0.487, z = 3.35.
The AUC gate should confirm that; if it does not, something else moved and the run is suspect.

**Also read:** `blind_pair.prior_corrected_accuracy` — what a user of this mind would get.

## 2. 289a address holdout  (~7 min)

    python -u _stage289a_presupposition.py --train-steps 6000 --addresses 1200 \
      --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 \
      --holdout address --run-tag w150auc

**Decides:** the same `auc_z`, under a split by subject instead of by corpus line. Holding on
both splits makes it a capability; on one only, a corpus artefact. The tag is reused on purpose
— the stage appends `_addrholdout` itself, so nothing is overwritten.

## 3. 289 smoke — read one line, then stop  (~2 min)

    python -u _stage289_derivation.py --smoke

**Read only:** `questions {...}`. All three verbs — count, compare, lookup — must clear
MIN_ANSWERED (5). If `compare` or `lookup` is short, raise `--train-lines`. Never touch the
model to fix a denominator.

## 4. 289 full — as a PAIR, both or neither

    python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 \
      --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150
    python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 \
      --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 \
      --no-derivation

**Decides:** `G_answer_is_derivation` — count and compare each above their random floor and
above the permutation null — and then the number no single run can produce:

    held_out.lookup.model_accuracy   full   vs   --no-derivation

That difference **is** `G_depth_does_not_cost_recall`. The gate printed inside one run only says
the lookup head still functions; it cannot see what depth cost.

**Do not expect** to beat `count_rival`. It reads the value strings and is optimal by
construction — a ceiling, not an opponent. `gap_to_exact_rule` is the claim.

## 5. 289c audit

    python -u _stage289c_audit.py --train-steps 6000 --addresses 1200 \
      --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150

**Decides:** `G_calibrated` (ECE under its own derived binomial noise) **and** `G_ranks_regions`.
Calibration alone is not enough — a map that is right on average but orders the regions wrongly
makes you trust the wrong half of your tape.

**Check for vacuity:** `refusal.vacuous`. True means the mind never declined and
`G_refusal_is_informed` states nothing. That is a real outcome, not a failure.

## 6. The two re-runs that only became affordable now

    python -u _stage286_evidence.py --train-steps 6000 --mind relational --run-tag speedcheck
    python -u _stage288_repair.py   --train-steps 6000 --run-tag min3subset

- 286 carries the arm where **`G_beats_silence` has never passed anywhere**. It was too
  expensive to attack properly; it no longer is.
- `stage288_decision_min3subset.json` was reconstructed **by hand from a log** after a smoke
  overwrote it. Its provenance is weaker than every other result on file. Replace it with a real
  run rather than cite it again.

Both accept `--no-speedups` to reproduce the old path if any number looks different.

---

## Then, per ROADMAP §5

- **290 conjecture writing** — blocked until each slot carries a monotone arrival index.
  Nothing in 290 can be built honestly before that exists.
- **291 selection**
- **292 generation**

## Cross-cutting, still open

- Derive the abstain reward as `2/c − 1`, so `G_beats_silence` means "beats guessing" rather
  than "beats a number someone picked".
- **Twenty tapes is enough** (HANDOFF §9c): `tape_curve` was 0.225 at ten tapes, 0.570 at
  twenty, then flat to 120. `--tape-period 300` at 6000 steps gives exactly twenty. Left at 50
  in the commands above only because the runs are now minutes; reach for 300 the moment a stage
  gets heavy again.
- Every stage still writes its JSON **only at the end**. Resample timings and the saturation
  probe are the only intermediate output; a crash still loses the report.
