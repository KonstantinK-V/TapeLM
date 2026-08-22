import json
from pathlib import Path

d = json.loads(Path("results/_stage424_308ceil.json").read_text(encoding="utf-8"))
print("seed  q     both   joint  comp_of  VOID  ROOM  ARENA  GO")
nv = nr = na = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, room, arena, go = (
        bool(h.get("void")), bool(h.get("room")),
        bool(h.get("arena")), bool(h.get("go")))
    nv += int(void)
    nr += int(room)
    na += int(arena)
    ng += int(go)
    print(
        f"{s}  {h.get('questions')}  {(h.get('both_offered') or 0):.3f}  "
        f"{(h.get('joint_seen') or 0):.3f}  {(h.get('comp_only_of_offered') or 0):.3f}  "
        f"{void}  {room}  {arena}  {go}"
    )
print(f"VOID {nv}/3  ROOM {nr}/3  ARENA {na}/3  GO {ng}/3")
if nv == 3:
    print("VERDICT: VOID — two-hole not askable; do not call Phi")
elif ng == 3:
    print("VERDICT: GO — counting-blind slice exists; hang could live (not mind yet)")
elif nr == 0:
    print("VERDICT: NO ROOM — pair catalog; Phi hang nothing to add")
elif na == 0:
    print("VERDICT: NO ARENA — counting covers offered pairs")
else:
    print("VERDICT: mixed — object not fed cleanly; do not call Phi")
