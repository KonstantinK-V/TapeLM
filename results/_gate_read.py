"""Extract overnight gate numbers from a stage289 decision JSON."""
from __future__ import annotations
import json, sys
from pathlib import Path

path, what = Path(sys.argv[1]), sys.argv[2]
j = json.loads(path.read_text(encoding="utf-8"))

def walk(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = walk(v, key)
            if r is not None:
                return r
    return None

if what == "unanswerable_rate":
    # prefer held_out.sparse
    sp = (j.get("held_out") or {}).get("sparse") or {}
    v = sp.get("unanswerable_rate")
    if v is None:
        v = walk(j, "unanswerable_rate")
    print("nan" if v is None else v)
elif what == "open_n":
    o = (j.get("held_out") or {}).get("open") or j.get("open") or {}
    print(int(o.get("n") or 0) if isinstance(o, dict) else 0)
elif what == "open_summary":
    ov = j.get("open_verb")
    o = (j.get("held_out") or {}).get("open") or {}
    near = j.get("open_near_source")
    print(f"open_verb={ov} n={o.get('n') if isinstance(o, dict) else None} near={near}")
else:
    raise SystemExit(f"unknown {what}")
