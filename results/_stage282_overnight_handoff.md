# Overnight handoff — 282 after m2b

## Queue (already running)
1. **m2b** (`216192`) — 280 with teacher 1-1 fix; key: `held_out.tie.teacher_abstain`
2. Orchestrator `_overnight_282.py` (`216197`): wait m2b → smoke → if not hard-fail → full → `--no-probe`
3. Review file after smoke: `results/_stage282_overnight_review.txt`

## When smoke finishes (agent MUST do this)
1. Read `results/stage280_decision_fp_m2b.json` — confirm `tie.teacher_abstain` rose (was 0.1).
2. Read `results/_stage282_smoke.out`, `results/stage282_decision*.json`, log, mini md if any.
3. If **clear bug** (traceback, empty pack, act space crash, missing gate fields, stall paid as silence, etc.): fix → re-smoke once.
4. If **not total fail** (ran to decision; not hard crash / MIND_INVALID from broken invariants only): launch in order:
   ```
   python _stage282_mind.py --bc-episodes 4000 --rl-episodes 3000 --min-mentions 2
   python _stage282_mind.py --bc-episodes 4000 --rl-episodes 3000 --min-mentions 2 --no-probe
   ```
   (sequential; second is ablation for probe value)
5. Do **not** start fulls on Traceback / no decision file / GPU OOM loop.

## Total fail = stop and leave a note
- Process died without `stage282_decision*.json`
- Import/syntax after our edits
- Tape empty / zero items

## Soft fail = still allow fulls
- `TEACHER_UNUSABLE`, `MIND_NO`, `MIND_PARTIAL` — still informative at full scale
- Gates red but training completed
