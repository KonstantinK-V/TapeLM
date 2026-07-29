"""
Stage 195 — hop2 chains: retest of old SOTE's death place (collision-bound @ d64).

Task: question context = paragraph P1 around entity B (B masked). Gold answer C
co-occurs with B in ANOTHER paragraph P2 (and never appears in P1). Distractors
never co-occur with B anywhere. Direct hop1 retrieval cannot solve this — the
chain is required: ctx(P1) → B → B's other contexts → C.

Two mechanisms tested:
  CHAIN   : B̂ = retrieve(q1); score(c) = max cos(key[B̂ slots ∉ P1], key[c slots])
  BINDING : edge memory e=norm(fp(A)⊙fp(B)) per co-occurring pair (old SOTE edge_fp);
            score(c) = max cos(norm(fp(B̂)⊙fp(c)), E)   ← the d64-collision victim, now d256

Gates:
  G1 chain acc >= 0.50 (chance 0.25)
  G2 direct hop1 scoring <= 0.35 (no shortcut — else items are broken)
  report: binding acc, oracle-B chain acc (upper bound)

  python _stage195_fp_hop2.py
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, FpBank

RES = Path("results")
DATA = Path("data")
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage195_decision.json"
MINI = RES / "stage195_mini.md"
LOG = RES / "_stage195_log.txt"

SEED = 195
CORPUS_CHARS = 150_000_000
EXAM_TAIL_CHARS = 3_000_000
N_ITEMS = 120
N_CAND = 4
MAX_PAIRS_PER_PARA = 30


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage195 start {datetime.now(timezone.utc).isoformat()}")
    log("hop2 chains + edge-binding retest (old SOTE failure, now d256)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    tail = text[-EXAM_TAIL_CHARS:]
    paras = [p.strip() for p in tail.split("\n") if 120 < len(p.strip()) < 1000][:1200]
    del text
    log(f"paras={len(paras)} ({time.time()-t0:.0f}s)")

    # ---- memory with para ids ----
    keys, vals, pids = [], [], []
    ents_of_para: list[list[str]] = []
    for pi, p in enumerate(paras):
        seen = []
        for m in ENT_RE.finditer(p):
            ent = m.group(1)
            lo, hi = max(0, m.start() - CTX_WIN), min(len(p), m.end() + CTX_WIN)
            k = bank.ctx_fp(p[lo:hi], exclude=ent)
            if k is not None:
                keys.append(k)
                vals.append(ent)
                pids.append(pi)
                seen.append(ent)
        ents_of_para.append(list(dict.fromkeys(seen)))
    K = torch.stack(keys, 0)
    pids_a = np.array(pids)
    log(f"memory slots={len(vals)} ({time.time()-t0:.0f}s)")

    by_ent: dict[str, list[int]] = {}
    for i, v in enumerate(vals):
        by_ent.setdefault(v, []).append(i)
    paras_of_ent: dict[str, set[int]] = {}
    for i, v in enumerate(vals):
        paras_of_ent.setdefault(v, set()).add(pids[i])
    co_partners: dict[str, set[str]] = {}
    for es in ents_of_para:
        for a in es:
            co_partners.setdefault(a, set()).update(e for e in es if e != a)
    all_ents = list(by_ent.keys())

    # ---- build hop2 items ----
    items = []
    cands_B = [e for e in all_ents if len(paras_of_ent[e]) >= 2]
    rng.shuffle(cands_B)
    for B in cands_B:
        if len(items) >= N_ITEMS:
            break
        ps = sorted(paras_of_ent[B])
        rng.shuffle(ps)
        ok = False
        for p1 in ps:
            for p2 in ps:
                if p1 == p2:
                    continue
                ents1 = set(ents_of_para[p1])
                golds = [c for c in ents_of_para[p2] if c != B and c not in ents1 and c in by_ent]
                if not golds:
                    continue
                C = golds[rng.randint(0, len(golds) - 1)]
                # TOPIC-MATCHED distractors: entities from paragraphs ADJACENT to p2
                # (same article/topic region) that never co-occur with B anywhere.
                pool = []
                for radius in (6, 15, 40):
                    for pn in range(max(0, p2 - radius), min(len(paras), p2 + radius + 1)):
                        if pn in (p1, p2):
                            continue
                        for d in ents_of_para[pn]:
                            if (
                                d not in (B, C)
                                and d not in co_partners.get(B, set())
                                and d not in ents1
                                and d not in ents_of_para[p2]
                                and d not in pool
                            ):
                                pool.append(d)
                    if len(pool) >= N_CAND - 1:
                        break
                rng.shuffle(pool)
                distractors = pool[: N_CAND - 1]
                if len(distractors) < N_CAND - 1:
                    continue
                # question context: window around B in P1, B excluded
                m = None
                for mm in ENT_RE.finditer(paras[p1]):
                    if mm.group(1) == B:
                        m = mm
                        break
                if m is None:
                    continue
                lo, hi = max(0, m.start() - CTX_WIN), min(len(paras[p1]), m.end() + CTX_WIN)
                cands = [C] + distractors
                order = list(range(N_CAND))
                rng.shuffle(order)
                items.append(
                    {
                        "ctx_text": paras[p1][lo:hi],
                        "B": B,
                        "p1": p1,
                        "cands": [cands[k] for k in order],
                        "gold_idx": order.index(0),
                    }
                )
                ok = True
                break
            if ok:
                break
    log(f"hop2 items={len(items)}")

    # ---- edge-binding memory (old SOTE edge_fp) ----
    edges = []
    for es in ents_of_para:
        pairs = 0
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                if pairs >= MAX_PAIRS_PER_PARA:
                    break
                fa = bank.fp([es[i]])[0]
                fb = bank.fp([es[j]])[0]
                edges.append(F.normalize(fa * fb, dim=-1))
                pairs += 1
    E = torch.stack(edges, 0)
    log(f"edge memory={len(edges)} pairs")

    # ---- scoring ----
    n = len(items)
    acc_chain = acc_direct = acc_bind = acc_oracle = 0
    hop1_hits = 0
    for it in items:
        q1 = bank.ctx_fp(it["ctx_text"], exclude=it["B"])
        if q1 is None:
            n -= 1
            continue
        sims = K @ q1
        B_hat = vals[int(sims.argmax())]
        hop1_hits += int(B_hat == it["B"])

        def chain_score(B_use):
            sc = []
            b_idx = [i for i in by_ent.get(B_use, []) if pids[i] != it["p1"]]
            for c in it["cands"]:
                c_idx = by_ent.get(c, [])
                if not b_idx or not c_idx:
                    sc.append(-1.0)
                    continue
                sub = K[b_idx] @ K[c_idx].T
                sc.append(float(sub.max()))
            return sc

        # chain with predicted B̂ / oracle B
        acc_chain += int(int(np.argmax(chain_score(B_hat))) == it["gold_idx"])
        acc_oracle += int(int(np.argmax(chain_score(it["B"]))) == it["gold_idx"])

        # direct hop1 shortcut check
        sc_d = []
        for c in it["cands"]:
            c_idx = by_ent.get(c, [])
            sc_d.append(float((K[c_idx] @ q1).max()) if c_idx else -1.0)
        acc_direct += int(int(np.argmax(sc_d)) == it["gold_idx"])

        # binding
        fB = bank.fp([B_hat])[0]
        sc_b = []
        for c in it["cands"]:
            e = F.normalize(fB * bank.fp([c])[0], dim=-1)
            sc_b.append(float((E @ e).max()))
        acc_bind += int(int(np.argmax(sc_b)) == it["gold_idx"])

    res = {
        "n": n,
        "hop1_B_acc": hop1_hits / max(1, n),
        "chain_acc": acc_chain / max(1, n),
        "chain_acc_oracle_B": acc_oracle / max(1, n),
        "direct_shortcut_acc": acc_direct / max(1, n),
        "binding_acc": acc_bind / max(1, n),
        "chance": 1 / N_CAND,
    }
    log(json.dumps(res, indent=2))

    g1 = res["chain_acc"] >= 0.50
    g2 = res["direct_shortcut_acc"] <= 0.35
    if g1 and g2:
        overall = "HOP2_CHAIN_YES"
    elif res["chain_acc"] >= 0.35 and g2:
        overall = "HOP2_CHAIN_WEAK"
    elif not g2:
        overall = "HOP2_ITEMS_LEAKY"
    else:
        overall = "HOP2_CHAIN_NO"
    if res["binding_acc"] >= 0.50 and g2:
        overall += "+BINDING_YES"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "fp_hop2_195",
        "overall": overall,
        "results": res,
        "slots": len(vals),
        "edges": len(edges),
        "note": "zero training; chain = two cosine hops over slot memory; binding = old SOTE edge_fp at d256",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage195 — hop2 chains + edge binding (d256 retest)",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- chain acc={res['chain_acc']:.3f} (oracle-B {res['chain_acc_oracle_B']:.3f}), chance 0.25",
                f"- direct shortcut={res['direct_shortcut_acc']:.3f} (must be low)",
                f"- binding (old edge_fp)={res['binding_acc']:.3f}",
                f"- hop1 B retrieval={res['hop1_B_acc']:.3f}, n={res['n']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[195] {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
