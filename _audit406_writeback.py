"""406: THREE WRITES. Does the mind's pick become a cell the next hop reads?"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

OUT = Path("results/_stage406_writeback.json")
CORPUS = [
    "the XARWIN team won the opening match of the season in a long padded line of text",
    "the XARWIN team lost the closing match of the season in a long padded line here",
    "the OTHER club played a different sport on a different field in a long padded line",
]

def build():
    toks, owner = [], []
    for i, line in enumerate(CORPUS):
        ws = line.split()
        toks.extend(ws); owner.extend([i] * len(ws))
    n = len(toks)
    places, place_of, buckets = [], {}, defaultdict(list)
    for s in range(n):
        left = toks[s - 1] if s > 0 and owner[s] == owner[s - 1] else None
        right = toks[s + 1] if s + 1 < n and owner[s] == owner[s + 1] else None
        buckets[(left, right)].append(s)
    for ps in buckets.values():
        if len(ps) < 2: continue
        pid = len(places); places.append(ps)
        for s in ps: place_of[s] = pid
    return {"toks": toks, "owner": owner, "places": places, "place_of": place_of}

def question(T):
    toks = T["toks"]
    hide = next(s for s, t in enumerate(toks) if t == "XARWIN" and T["owner"][s] == 1)
    pid = T["place_of"][hide]
    homes = defaultdict(list)
    for s, t in enumerate(toks): homes[t].append(s)
    return {"hide": hide, "pid": pid, "truth": toks[hide], "homes": dict(homes),
            "slots": list(T["places"][pid]), "query_row": list(T["places"][pid]).index(hide),
            "corpus_n": len(toks), "marked": set(), "working": []}

def fill_conjecture(q, said):
    w = dict(q)
    w["tmp_vals"] = [said if i == q["query_row"] else None for i in range(len(q["slots"]))]
    w["working"] = list(q["working"])
    w["marked"] = set(q["marked"])
    return w

def write_back(q, said):
    w = dict(q)
    w["working"] = list(q["working"]) + [said]
    w["marked"] = set(q["marked"]) | {("work", len(w["working"]) - 1)}
    w["here"] = ("work", len(w["working"]) - 1)
    w["tmp_vals"] = None
    return w

def hop2_cells(T, q, said):
    work = [i for i, v in enumerate(q["working"]) if v == said]
    marked = [i for i in work if ("work", i) in q["marked"]]
    return {"corpus_homes": list(q["homes"].get(said, [])), "working": work,
            "marked_working": marked}

def measure(write):
    T = build(); q0 = question(T); corpus_n = q0["corpus_n"]; rows = {}
    for said in (q0["truth"], "OTHER"):
        q = write_back(q0, said) if write else fill_conjecture(q0, said)
        h = hop2_cells(T, q, said)
        rows[said] = {
            "corpus_untouched": q0["corpus_n"] == corpus_n and len(T["toks"]) == corpus_n,
            "working_cells": len(q["working"]), "marked_working": len(h["marked_working"]),
            "hop2_sees_marked": int(len(h["marked_working"]) > 0),
            "hop2_corpus_homes": len(h["corpus_homes"]),
            "tmp_fill": q.get("tmp_vals") is not None,
        }
    return {"write_back": bool(write), "truth": rows[q0["truth"]], "wrong": rows["OTHER"],
            "hop2_sees_write": rows[q0["truth"]]["hop2_sees_marked"],
            "wrong_walks_elsewhere": (rows["OTHER"]["hop2_corpus_homes"] > 0 and
                rows[q0["truth"]]["hop2_corpus_homes"] != rows["OTHER"]["hop2_corpus_homes"])}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-back", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rep = measure(args.write_back)
    print(f"arm        {'WRITE-BACK' if args.write_back else 'TODAY (conjecture fill)'}")
    print(f"hop2 sees a marked cell the mind wrote   {rep['hop2_sees_write']}")
    print(f"corpus untouched                         {rep['truth']['corpus_untouched']}")
    print(f"tmp fill (throwaway query row)           {rep['truth']['tmp_fill']}")
    print(f"working cells after pick                 {rep['truth']['working_cells']}")
    print(f"wrong pick walks a different cluster     {rep['wrong_walks_elsewhere']}")
    print("\nTODAY: throwaway graph." if not args.write_back else
          "\nWRITE-BACK: marked cell; hop 2 stands on it; wrong pick walks OTHER.")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev["write_back" if args.write_back else "today"] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
