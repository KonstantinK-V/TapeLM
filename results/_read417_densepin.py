import json
from pathlib import Path

d = json.loads(Path("results/_stage417_densepin.json").read_text(encoding="utf-8"))
print("seed  live   ora    rnd    dlt    ref1   ref2   void gate refuse_ok")
og = orf = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    g = bool(h.get("gate"))
    r = bool(h.get("refuse_ok"))
    v = bool(h.get("void"))
    og += int(g)
    orf += int(r)
    print(
        f"{s}  {h.get('live'):.4f}  {h.get('oracle'):.4f}  {h.get('random'):.4f}  "
        f"{h.get('oracle_minus_random'):+.4f}  {h.get('refuse_df1')}  {h.get('refuse_df2')}  "
        f"{v} {g} {r}"
    )
print(f"GATE ora-rnd: {og}/3   refuse_ok: {orf}/3")
if og == 0:
    print("VERDICT: coin — do not train Phi on these labels")
elif og == 3:
    print("VERDICT: teacher exists — future CE over y (places+REFUSE), not vocab")
else:
    print("VERDICT: mixed — read per seed, do not pool")
