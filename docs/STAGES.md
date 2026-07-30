# Stage index (TapeLM line: 170–212)

Active scripts live in the **repo root** (`_stage*.py`). Legacy pre-170 scripts are under [`legacy/sote/`](../legacy/sote/).

**Architecture context:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · **Visitor map:** [`../artifact/OVERVIEW.md`](../artifact/OVERVIEW.md)

TapeLM line: many stages end in **`_YES` / `_WIN` / `_PARTIAL`** (191–205, 203, 204, 205); a **smaller** set closes variant B and internalization (207–212). See `show_map.py` or [`artifact/decisions/`](../artifact/decisions/).
| Stage | Script | Verdict (see `results/stage*_decision.json`) |
|-------|--------|-----------------------------------------------|
| 170 | `_stage170_curve_dynamics.py` | Curve dynamics smoke |
| 177 | `_stage177_curve_bpe.py` | BPE-on-curve training |
| 180–182 | dual-channel / CE control | Retention + parity path |
| 185–186 | tape read / exam | Calibration harness |
| 187–191 | self-model / **NIGHT P1** | **Parity** encoder |
| 192–193 | fp lexicon + wired | Calibration win |
| 194–195 | fact memory + hop2 | Recall / binding |
| 196 | `_stage196_tapelm.py` | **TapeLM assemble** |
| 197–198 | edit + stream | Edit / stream wins |
| 199–202b | semantic probes | B not confirmed @3050 |
| 203 | internal hops | Structured only |
| 204–205 | noise + unlearn | Capability wins |
| 206 | latent hops budget | Latency only |
| 207 / 207-MAX | curve thinking | **B falsified** |
| 208 | hybrid rare head | No gain |
| 209 | semantic scaling | Not structurally blocked |
| 210–212 | pre-publish frontier | All **THESIS_NO** |
| 213 | arc_enc freeze finetune | **ARC_ENC_FREEZE_** partial |
| 214–220 | extension pipeline | see [`EXTENSION_PIPELINE.md`](EXTENSION_PIPELINE.md) |

Full narrative: [`results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md).
