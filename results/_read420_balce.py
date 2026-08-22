import json
from pathlib import Path

p = Path("results/_stage420_balce.json")
if not p.exists():
    print("no results/_stage420_balce.json yet")
    raise SystemExit(1)
d = json.loads(p.read_text(encoding="utf-8"))
print("seed  mind_live  rnd_live  dlt    mind   always_r  w_ref  pins  ge_ar  standing")
gp = ga = st = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    if h.get("loss") not in (None, "ce_417h_w_refuse"):
        print(f"{s}  SKIP foreign loss={h.get('loss')}")
        continue
    pins = bool(h.get("gate_pins"))
    ar = bool(h.get("gate_mind_ge_always_refuse"))
    stand = bool(h.get("standing"))
    gp += pins
    ga += ar
    st += stand
    wr = h.get("w_refuse")
    print(
        f"{s}  {h.get('mind_live'):.3f}  {h.get('random_live'):.3f}  "
        f"{h.get('mind_live_minus_random_live'):+.3f}  "
        f"{h.get('mind_pin'):.3f}  {h.get('always_refuse'):.3f}  "
        f"{(wr if wr is not None else float('nan')):.3f}  "
        f"{pins}  {ar}  {stand}"
    )
print(f"GATE pins: {gp}/3   mind>=always_refuse: {ga}/3   standing: {st}/3")
if st == 3:
    print("VERDICT: freeze 420 — both gates; then raw tape")
elif gp == 0:
    print("VERDICT: Phi does not rank joint pins — stop CE on this y")
elif gp == 3 and ga < 3:
    print("VERDICT: pins move, silencer still wins")
else:
    print("VERDICT: mixed — read per seed")
