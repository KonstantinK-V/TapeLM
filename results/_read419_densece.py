import json
from pathlib import Path

d = json.loads(Path("results/_stage419_densece.json").read_text(encoding="utf-8"))
print("seed  n_live  mind_live  always_ref  ref1   ref2   mind   rnd    beat  refuse_g")
ob = or_ = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    beat = bool(h.get("gate_mind_beats_random"))
    rg = bool(h.get("gate_refuse_one_gt_ge2"))
    ob += beat
    or_ += rg
    ml = h.get("mind_live")
    ar = h.get("always_refuse")
    print(
        f"{s}  {h.get('n_live')}  "
        f"{(ml if ml is not None else float('nan')):.3f}  "
        f"{(ar if ar is not None else float('nan')):.3f}  "
        f"{h.get('refuse_df1'):.3f}  {h.get('refuse_df2'):.3f}  "
        f"{h.get('mind_pin'):.3f}  {h.get('random_pin'):.3f}  "
        f"{beat}  {rg}"
    )
print(f"GATE mind>random: {ob}/3   refuse: {or_}/3")
beats_ar = sum(
    1 for s in ("1337", "8642", "2890")
    if (d.get(s) or {}).get("mind_pin", 0) > (d.get(s) or {}).get("always_refuse", 1)
)
live_ok = sum(
    1 for s in ("1337", "8642", "2890")
    if ((d.get(s) or {}).get("mind_live") or 0) > 0.05
)
print(f"mind>always_refuse: {beats_ar}/3   mind_live>0.05: {live_ok}/3")
print("STANDING freeze=419_joint_ce — read mind_live vs always_refuse; not a new lever")
if live_ok == 3 and beats_ar == 3:
    print("VERDICT: standing joint CE — pins learned")
elif live_ok == 0 and beats_ar == 0:
    print("VERDICT: always-refuse trap — mind>random decorative; pins not learned")
else:
    print("VERDICT: mixed — read mind_live / always_refuse per seed")
