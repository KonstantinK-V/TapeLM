# Queue after typed contract (order)

1. ~~**227** … **233**~~ — trunk + engineering (231 temporal W, 232 stream+version, 233 tool bind)
2. ~~**234** compositional W (228 algebra)~~ — `_stage234_compositional_W.py`
3. ~~**235** mixed L1 probe~~ — `_stage235_mixed_l1_probe.py`

**Still open (scale, not v1 headline):** full multi-domain **pretrain L1** at 191-scale; compositional W productization only if 234 holds on fixed exam JSON.

```bash
python _stage234_compositional_W.py
python _stage235_mixed_l1_probe.py
python artifact/scripts/sync_decisions.py
```

**Latest full runs (2026-07-30):** `COMPOSITIONAL_W_OK` (234) · `MIXED_L1_PROBE_OK` (235).
