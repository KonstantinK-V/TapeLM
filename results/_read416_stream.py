import json
from pathlib import Path

d = json.loads(Path("results/_stage416_stream.json").read_text(encoding="utf-8"))
print("seed  ge2_n mind  riv   dlt   one_n ref1  ref_ge2 beat refg")
ob = or_ = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    g, o = h.get("ge2") or {}, h.get("one") or {}
    beat = bool(h.get("gate_mind_beats_rival"))
    rg = bool(h.get("gate_refuse_one_gt_ge2"))
    ob += beat
    or_ += rg
    mh, rh = g.get("mind_hit"), g.get("rival_hit")
    dlt = (mh - rh) if mh is not None and rh is not None else float("nan")
    print(
        f"{s}  {g.get('n')}  {mh:.3f}  {rh:.3f}  {dlt:+.3f}  "
        f"{o.get('n')}  {o.get('refuse'):.3f}  {g.get('refuse'):.3f}  {beat}  {rg}"
    )
print(f"GATE mind>rival: {ob}/3  refuse: {or_}/3")
