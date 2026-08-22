import json
from pathlib import Path

d = json.loads(Path("results/_stage418_densece.json").read_text(encoding="utf-8"))
print("seed  n    mind   rnd    dlt    ref1   ref2   beat  refuse_g")
ob = or_ = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    beat = bool(h.get("gate_mind_beats_random"))
    rg = bool(h.get("gate_refuse_one_gt_ge2"))
    ob += beat
    or_ += rg
    print(
        f"{s}  {h.get('n')}  {h.get('mind_pin'):.3f}  {h.get('random_pin'):.3f}  "
        f"{h.get('mind_minus_random'):+.3f}  {h.get('refuse_df1'):.3f}  "
        f"{h.get('refuse_df2'):.3f}  {beat}  {rg}"
    )
print(f"GATE mind>random: {ob}/3   refuse: {or_}/3")
if ob == 3 and or_ == 3:
    print("VERDICT: CE labels fit pin-Phi — lab holes may probe after freeze")
elif ob == 0:
    print("VERDICT: labels did not fit Phi — not need cut")
else:
    print("VERDICT: mixed — read per seed")
