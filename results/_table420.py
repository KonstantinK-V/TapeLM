import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

a = load("results/_stage420_balce.json")
b = load("results/_stage420b_batch.json")
print("arm   seed  mind_live  rnd_live     dlt   mind_pin  always_r   pins  ge_ar  standing")
for arm, d in (("420 ", a), ("420b", b)):
    for s in ("1337", "8642", "2890"):
        h = d[s]
        pins = bool(h.get("gate_pins"))
        ge = bool(h.get("gate_mind_ge_always_refuse"))
        st = bool(h.get("standing")) if "standing" in h else (pins and ge)
        print(
            f"{arm}  {s}  {h['mind_live']:8.3f}  {h['random_live']:8.3f}  "
            f"{h['mind_live_minus_random_live']:+7.3f}  {h['mind_pin']:8.3f}  "
            f"{h['always_refuse']:8.3f}  {str(pins):5}  {str(ge):5}  {st}"
        )
print()
print("420  = w_ref + structural feats (overlap/bag/slots/refuse)")
print("420b = batch 50/50 + letter-hash fillers")
print("GATE pins: mind_live - random_live > 0.05")
print("ge_ar:     mind_pin >= always_refuse")
