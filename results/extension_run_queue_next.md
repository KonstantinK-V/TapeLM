# Queue after typed contract (order)

1. ~~**227** canonical slots + W@read~~ — `CANONICAL_STORAGE_OK`
2. ~~**228c** official decode API~~ — `FP_DECODE_FIX_YES` + `docs/MEMORY_ENGINEERING.md`
3. ~~**229** contradiction / multi-hit~~ — `CONTRADICTION_RAW_MEMORY_OK`
4. **230** resolution policy — `_stage230_slot_resolution.py`
5. **226c** joint e2e (228c in gen loop) — `_stage226c_joint_fp_decode.py`
6. **228** compositional W — branch later
7. **231** temporal W — branch later
8. **233** tool binding — branch later

```bash
python _stage230_slot_resolution.py
python _stage226c_joint_fp_decode.py
python artifact/scripts/sync_decisions.py
```

If **230 + 226c** close on full runs → narrative upgrade from “artifact only” to **memory product contract** (persist W on HF + resolution + decode path documented).

**Status (full run):** `RESOLUTION_POLICY_OK` (230) · `JOINT_FP_DECODE_OK` (226c) — trunk ready for HF `w_registry` + GitHub release narrative (not only demo artifact).
