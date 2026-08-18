import json
import math
from pathlib import Path

TAGS = ["audit1200", "290b_dense", "audit2400", "292_s4001", "292_s7919"]


def r4(x):
    if isinstance(x, float):
        if math.isnan(x):
            return None
        return round(x, 4)
    return x


def slim_paired(p):
    if not isinstance(p, dict) or not p:
        return None
    return {
        "z": r4(p.get("mcnemar_z")),
        "m_only": p.get("model_only_right"),
        "r_only": p.get("rival_only_right"),
        "discordant": p.get("discordant"),
        "underpowered": p.get("underpowered"),
        "max_z": r4(p.get("max_achievable_z")),
    }


def slim_lookup(blk):
    if not isinstance(blk, dict):
        return None
    return {
        "n": blk.get("n"),
        "phi": r4(blk.get("model_accuracy")),
        "rival": r4(blk.get("rival_accuracy")),
        "floor": r4(blk.get("random_floor")),
        "rival_cos": r4(blk.get("rival_cos_accuracy")),
    }


def slim_sparse(sp):
    if not isinstance(sp, dict):
        return None
    return {
        "n": sp.get("n"),
        "n_answerable": sp.get("n_answerable"),
        "phi": r4(sp.get("acc_answerable")),
        "own_row": r4(sp.get("rival_own_row_accuracy")),
        "n_no_own_row": sp.get("n_no_own_row"),
        "acc_no_own_row": r4(sp.get("acc_no_own_row")),
        "false_refusal": r4(sp.get("false_refusal")),
    }


def slim_open(op):
    if not isinstance(op, dict):
        return None
    return {
        "n": op.get("n"),
        "phi": r4(op.get("accuracy")),
        "ret": r4(op.get("corpus_retrieval_accuracy")),
        "floor": r4(op.get("random_floor")),
        "vs_ret": slim_paired(op.get("paired_vs_corpus_retrieval")),
    }


def slim_ladder(lad):
    if not isinstance(lad, dict):
        return None
    keep = {}
    for k in ("consistency", "n_pairs", "underpowered", "mean_phi"):
        if k in lad:
            v = lad[k]
            if isinstance(v, dict):
                keep[k] = {kk: r4(vv) for kk, vv in v.items()}
            else:
                keep[k] = r4(v) if isinstance(v, float) else v
    # sometimes mean_phi flattened as true/near/middle/far
    for k in ("true", "near", "middle", "far"):
        if k in lad and isinstance(lad[k], (int, float)):
            keep.setdefault("mean_phi", {})[k] = r4(lad[k])
    return keep or None


def slim_audit(a):
    if not isinstance(a, dict):
        return None
    out = {}
    for k_key, routes in a.items():
        if not isinstance(routes, dict):
            continue
        out[k_key] = {
            route: {
                "q": body.get("questions"),
                "ans": body.get("answerable"),
                "hit": r4(body.get("hit_rate")),
                "rows": r4(body.get("mean_rows")),
            }
            for route, body in routes.items()
            if isinstance(body, dict)
        }
    return out


def slim_tape_side(t):
    if not isinstance(t, dict):
        return None
    return {
        "addresses": t.get("addresses"),
        "slots": t.get("slots"),
        "mentions_per_address": r4(t.get("mentions_per_address")),
        "lookup_q": t.get("lookup_questions"),
    }


def slim_side(ex):
    if not isinstance(ex, dict):
        return None
    out = {
        "lookup": slim_lookup(ex.get("lookup")),
        "vs_counts": slim_paired(ex.get("lookup_paired_vs_rival")),
        "vs_cos": slim_paired(ex.get("lookup_paired_vs_rival_cos")),
        "sparse": slim_sparse(ex.get("sparse")),
        "open": slim_open(ex.get("open")),
        "ladder": slim_ladder(ex.get("ladder")),
    }
    return {k: v for k, v in out.items() if v is not None}


def slim_run(tag):
    p = Path(f"results/stage289_decision_{tag}.json")
    if not p.exists():
        return {"run_tag": tag, "missing": True}
    d = json.loads(p.read_text(encoding="utf-8"))
    es = d.get("early_stop") or {}
    gates = d.get("gates") or {}
    tape = d.get("tape_shape") or {}
    out = {
        "run_tag": d.get("run_tag") or tag,
        "seed": d.get("seed"),
        "overall": d.get("overall"),
        "fp": d.get("fp"),
        "neighbours": d.get("neighbours"),
        "import_k": d.get("import_k"),
        "open_verb": d.get("open_verb"),
        "open_near_source": d.get("open_near_source"),
        "params": d.get("params"),
        "train_steps": d.get("train_steps"),
        "wall_s": r4(d.get("wall_s")) if isinstance(d.get("wall_s"), (int, float)) else d.get("wall_s"),
        "early_stop": {
            "best_step": es.get("best_step"),
            "best_probe_loss": r4(es.get("best_probe_loss")),
            "total_steps": es.get("total_steps"),
        },
        "gates_fail": {k: v for k, v in gates.items() if v is False} or None,
        "tape": {
            "held": slim_tape_side(tape.get("held_out")),
            "train": slim_tape_side(tape.get("train")),
        },
        "held": slim_side(d.get("held_out")),
        "train": slim_side(d.get("train_control")),
        "neighbourhood_audit": slim_audit(d.get("neighbourhood_audit")),
    }
    return {k: v for k, v in out.items() if v is not None}


def score_row(r):
    if r.get("missing"):
        return {"id": r["run_tag"], "missing": True}
    h = r.get("held") or {}
    t = r.get("train") or {}
    row = {
        "id": r["run_tag"],
        "seed": r.get("seed"),
        "overall": r.get("overall"),
        "best": (r.get("early_stop") or {}).get("best_step"),
        "addrs_held": ((r.get("tape") or {}).get("held") or {}).get("addresses"),
    }
    sp = h.get("sparse")
    if sp:
        row["sparse_phi"] = sp.get("phi")
        row["own_row"] = sp.get("own_row")
        row["sparse_n"] = sp.get("n")
        row["no_own"] = sp.get("n_no_own_row")
        # delta: neighbourhood beyond own row
        if sp.get("phi") is not None and sp.get("own_row") is not None:
            row["phi_minus_own"] = r4(sp["phi"] - sp["own_row"])
    op_h = h.get("open")
    op_t = t.get("open")
    if op_h:
        row["held_open"] = op_h.get("phi")
        row["held_ret"] = op_h.get("ret")
        row["held_z"] = (op_h.get("vs_ret") or {}).get("z")
        row["held_n"] = op_h.get("n")
    if op_t:
        row["train_open"] = op_t.get("phi")
        row["train_ret"] = op_t.get("ret")
        row["train_z"] = (op_t.get("vs_ret") or {}).get("z")
        row["train_n"] = op_t.get("n")
    audit = r.get("neighbourhood_audit") or {}
    k3 = audit.get("k=3") or {}
    if k3:
        row["audit_k3_hit"] = {
            route: (body or {}).get("hit")
            for route, body in k3.items()
            if route in ("anchor", "rel", "all")
        }
    return row


def slim_baseline():
    p = Path("results/stage289_decision_292_open.json")
    if not p.exists():
        return None
    r = slim_run("292_open")
    return score_row(r)


def main():
    runs = [slim_run(t) for t in TAGS]
    scoreboard = [score_row(r) for r in runs]
    doc = {
        "series": "289 night: audit1200 → 290b_dense → audit2400 → 292 seeds",
        "now": "idle — night + seed queues finished (QUEUE_DONE seed replicas)",
        "read": (
            "scoreboard first. "
            "290b: sparse_phi vs own_row (phi_minus_own). "
            "audits: audit_k3_hit and neighbourhood_audit route×k. "
            "292 seeds: held_z vs train_z — held≫train did not replicate on 4001."
        ),
        "baseline_292_open": slim_baseline(),
        "scoreboard": scoreboard,
        "runs": runs,
    }
    outp = Path("results/stage289_night_290_292_compact.json")
    outp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", outp, "bytes", outp.stat().st_size)
    print(json.dumps({"now": doc["now"], "scoreboard": scoreboard, "baseline": doc["baseline_292_open"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
