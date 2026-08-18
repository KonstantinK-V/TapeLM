"""
Stage 270 — A tape that lies: the first exam a lookup cannot pass.

Everything measured so far has been retrieval. The tape holds one truth per key, the question is
which key, and a good index wins — which is why zero-train word votes keep beating a trained
query (266: 0.199 vs 0.062). No stage so far has posed a question where finding the slot is not
the same as knowing the answer.

Here several slots speak about the same subject and they disagree. Three witnesses say the value
is X, one says Y, and every witness is keyed just as well as the others — same subject fp, same
sentence shape, only the value differs. Top-1 retrieval therefore lands on the liar about as often
as chance allows, and the only route to the answer is to read several slots and weigh them.

That is the first place in this project where memory alone is provably insufficient, and it is
what "mind separate from memory" has to mean: the tape can be wrong, and something else judges.
No RAG setup asks this — there the retrieved passage is true by definition.

270 does not train anything. It asks whether the machinery already built can survive a tape that
lies, and it establishes the numbers a trained mind would have to beat:

    A  lookup      value of the single highest-similarity slot        must FAIL
    B  majority    unweighted vote over the witnesses retrieved       the honest target
    C  idf_weight  witnesses weighted by key similarity               does sharpening help or hurt
    D  glue        265 span-lock decode, untouched                    what today's pipeline does

Gates are written so the stage can fail informatively. G_lookup_fails is a VALIDITY gate: if
single-slot lookup already answers contradicted subjects, the exam has a leak — most likely the
liar's key is distinguishable — and every other number is meaningless.

  python _stage270_lying_tape.py --smoke
  python _stage270_lying_tape.py --witnesses 4 --liars 1
  python _stage270_lying_tape.py --liars 2          # 3-vs-2, near the aggregation limit
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
import _stage265_span_lock as s265
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import SlotBias, TapeView, ctx_query, load_glue

RES = Path("results")
DECISION = RES / "stage270_decision.json"
MINI = RES / "stage270_mini.md"
LOG = RES / "_stage270_log.txt"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_268 = Path("checkpoints/stage268_mind_learns_tape.pt")
CKPT_265 = Path("checkpoints/stage265_span_lock.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 270

CUE = "{S} was appointed director of"

# Witnesses must not be byte-identical, or their keys collapse and "several slots" is a fiction.
# They must also not encode WHICH witness is right — no hedging, no "reportedly", no ordering
# words. Every one of these is a flat assertion of equal standing.
WITNESS_TMPL = (
    "{S} was appointed director of {V} in the regional chronicle of 1987 .",
    "The county register lists {S} as appointed director of {V} that year .",
    "According to the parish record , {S} was appointed director of {V} .",
    "{S} , appointed director of {V} , appears in the 1987 civil roll .",
    "A ledger entry names {S} as the appointed director of {V} .",
)


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def fp_version() -> str:
    fn = getattr(L, "canonical_fp_version", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    return CKPT_P1.name


def build_lying_tape(
    *, bank_can, tok, pad_id, device, rng, values_pool, lines, n_subj, n_wit, n_liars, n_dist,
):
    """Each contradicted subject gets n_wit witnesses: n_wit-n_liars agree, n_liars dissent.

    A control population of plain single-witness subjects rides along, so "the machinery broke on
    a lying tape" can be told apart from "the machinery broke".
    """
    avail = [w for w in values_pool if len(w) >= 5]
    rng.shuffle(avail)
    subs = [w for w in gen_fakes(set(avail), rng, n_subj * 2 + 80) if len(w) >= 5]
    subs = list(dict.fromkeys(subs))
    need_vals = n_subj * (1 + n_liars) + n_subj  # truths + lies + clean subjects
    if len(subs) < n_subj * 2 or len(avail) < need_vals:
        raise RuntimeError(f"pool too small: subs={len(subs)} vals={len(avail)} need={need_vals}")

    keys, vals, texts = [], [], []
    items, clean_items = [], []
    vi = 0

    def add_slot(S, value, tmpl_i):
        sent = WITNESS_TMPL[tmpl_i % len(WITNESS_TMPL)].format(S=S, V=value)
        kf = bank_can.fp([S])[0]
        c = bank_can.ctx_fp(sent, exclude=value)
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(value)
        texts.append(sent)
        return len(vals) - 1

    # contradicted subjects
    for i in range(n_subj):
        S = subs[i]
        truth = avail[vi]; vi += 1
        # One shared lie, repeated by every liar. The smoke gave each liar its own value, so the
        # tally read 2/1/1 and the truth won by default — the exam never posed the question.
        lie = avail[vi]; vi += 1
        lies = [lie] * n_liars
        n_true = n_wit - n_liars
        slots, slot_val = [], {}
        order = [truth] * n_true + lies
        rng.shuffle(order)  # the liar is not always last
        for j, v in enumerate(order):
            sid = add_slot(S, v, j)
            slots.append(sid)
            slot_val[sid] = v
        items.append({
            "S": S, "truth": truth, "lies": lies, "slots": slots, "slot_val": slot_val,
            "n_true": n_true, "n_liars": n_liars,
        })

    # clean control subjects: one witness, no contradiction
    for i in range(n_subj):
        S = subs[n_subj + i]
        v = avail[vi]; vi += 1
        sid = add_slot(S, v, 0)
        clean_items.append({"S": S, "truth": v, "lies": [], "slots": [sid],
                            "slot_val": {sid: v}, "n_true": 1, "n_liars": 0})

    # wiki noise
    used = set(vals)
    from _inprint_glue import ANCHOR_RE
    for ln in lines:
        if len(vals) >= n_subj * (n_wit + 1) + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used:
                continue
            lo, hi = max(0, m.start() - 120), min(len(ln), m.end() + 120)
            c = bank_can.ctx_fp(ln[lo:hi], exclude=ent)
            if c is None:
                continue
            anch = [w for w in ANCHOR_RE.findall(ln[lo: m.start()]) if w != ent]
            if not anch:
                continue
            keys.append(F.normalize(bank_can.fp([anch[-1]])[0] + c, dim=-1))
            vals.append(ent)
            texts.append(ln[lo:hi])
            used.add(ent)
            if len(vals) >= n_subj * (n_wit + 1) + n_dist:
                break

    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    return tape, items, clean_items, texts


@torch.no_grad()
def retrieve(glue, bank_can, tok, tape, S, pad_id, k):
    cue_ids = [i for i in tok.encode(CUE.format(S=S)).ids if i != pad_id]
    q = ctx_query(glue, bank_can, tok, cue_ids, anchor_ids=cue_ids)
    if q is None:
        return None, None
    sims = (tape.K @ q).masked_fill(~tape.alive, -1e4)
    v, idx = torch.topk(sims, min(k, int(tape.alive.sum())))
    return v, idx


def arm_scores(glue, bank_can, tok, tape, items, pad_id, k):
    """A lookup / B majority / C similarity-weighted, plus how often the liar outranks the truth."""
    a_hit, b_hit, c_hit, liar_top, wit_recall = [], [], [], [], []
    rows = []
    for it in items:
        sims, idx = retrieve(glue, bank_can, tok, tape, it["S"], pad_id, k)
        if idx is None:
            a_hit.append(0); b_hit.append(0); c_hit.append(0)
            liar_top.append(0); wit_recall.append(0.0)
            continue
        ids = idx.tolist()
        svals = sims.tolist()
        own = [(j, s) for j, s in zip(ids, svals) if j in it["slot_val"]]
        wit_recall.append(len(own) / max(1, len(it["slots"])))

        top_val = tape.values[ids[0]]
        a_hit.append(int(top_val == it["truth"]))
        liar_top.append(int(top_val in it["lies"]))

        cnt = Counter(tape.values[j] for j, _ in own)
        b_hit.append(int(bool(cnt) and cnt.most_common(1)[0][0] == it["truth"]))

        w: dict[str, float] = defaultdict(float)
        for j, s in own:
            w[tape.values[j]] += max(0.0, s)
        c_hit.append(int(bool(w) and max(w.items(), key=lambda kv: kv[1])[0] == it["truth"]))

        rows.append({
            "S": it["S"], "truth": it["truth"], "top1": top_val,
            "witnesses_retrieved": len(own),
            "counts": dict(cnt),
        })
    n = max(1, len(items))
    return {
        "lookup_top1": sum(a_hit) / n,
        "majority": sum(b_hit) / n,
        "sim_weighted": sum(c_hit) / n,
        "liar_is_top1": sum(liar_top) / n,
        "witness_recall": float(np.mean(wit_recall)) if wit_recall else float("nan"),
        "n": len(items),
        "rows": rows[:8],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--witnesses", type=int, default=4)
    ap.add_argument("--liars", type=int, default=1,
                    help="witnesses repeating ONE shared lie; --liars 2 of 4 is a tie")
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--topk", type=int, default=0, help="0 = witnesses + 4")
    ap.add_argument("--glue-ckpt", type=str, default="", help="268 or 265 checkpoint; empty = auto")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    n_subj = args.subjects or (12 if args.smoke else 60)
    n_wit = args.witnesses
    n_liars = args.liars
    n_dist = args.distractor_slots or (200 if args.smoke else 1200)
    k = args.topk or (n_wit + 4)
    max_new = 6 if args.smoke else 12
    max_lines = 600 if args.smoke else 6000
    if n_liars >= n_wit - n_liars:
        log(f"  WARNING: {n_liars} liars vs {n_wit - n_liars} truths — majority is not defined")

    log(f"Stage270 lying-tape start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"subjects={n_subj} witnesses={n_wit} liars={n_liars} topk={k} dist={n_dist}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)

    ck = Path(args.glue_ckpt) if args.glue_ckpt else (CKPT_268 if CKPT_268.is_file() else CKPT_265)
    glue = None
    if ck.is_file():
        # 268: {"glue": state_dict}; 265: {"W_q_glue"|"W_q", "gate", "log_tau"}; 256: flat W_q.
        try:
            glue = load_glue(model, device, ck)
        except (KeyError, RuntimeError):
            glue = None
        if glue is None:
            st = torch.load(ck, map_location=device, weights_only=False)
            glue = SlotBias(2 * (model.head.in_features // 2), device)
            with torch.no_grad():
                if "glue" in st and isinstance(st["glue"], dict):
                    glue.load_state_dict(st["glue"], strict=False)
                    log(f"  loaded glue from {ck.name} via nested state_dict")
                elif "W_q_glue" in st:
                    glue.W_q.load_state_dict(st["W_q_glue"])
                    glue.gate.load_state_dict(st["gate"])
                    glue.log_tau.copy_(st["log_tau"].to(device))
                    log(f"  loaded glue from {ck.name} via W_q_glue")
                elif "W_q" in st:
                    glue.W_q.load_state_dict(st["W_q"])
                    glue.gate.load_state_dict(st["gate"])
                    glue.log_tau.copy_(st["log_tau"].to(device))
                    log(f"  loaded glue from {ck.name} via W_q")
                else:
                    glue = None
                    log(f"  {ck.name} has no readable glue — arm D would be meaningless")
            if glue is not None:
                glue.eval()
    if glue is None:
        glue = SlotBias(2 * (model.head.in_features // 2), device)
        glue.eval()
        log(f"  no glue checkpoint at {ck} — running with an UNTRAINED glue (W_q ~ identity)")
    else:
        log(f"  glue={ck.name} trunk={trunk_ckpt.name} fp_version={fp_version()}")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_000_000 if args.smoke else 6_000_000)
    values_pool = list(dict.fromkeys(
        m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5
    ))
    rng.shuffle(values_pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:max_lines]

    tape, items, clean_items, _ = build_lying_tape(
        bank_can=bank_can, tok=tok, pad_id=pad_id, device=device, rng=rng,
        values_pool=values_pool, lines=lines, n_subj=n_subj, n_wit=n_wit,
        n_liars=n_liars, n_dist=n_dist,
    )
    log(f"  tape slots={len(tape.values)} contradicted={len(items)} clean={len(clean_items)}")

    contra = arm_scores(glue, bank_can, tok, tape, items, pad_id, k)
    clean = arm_scores(glue, bank_can, tok, tape, clean_items, pad_id, k)
    log(f"  contradicted: lookup={contra['lookup_top1']:.3f} majority={contra['majority']:.3f} "
        f"sim_w={contra['sim_weighted']:.3f} liar_top1={contra['liar_is_top1']:.3f} "
        f"witness_recall={contra['witness_recall']:.3f}")
    log(f"  clean       : lookup={clean['lookup_top1']:.3f}")

    # D: what today's pipeline actually emits, span-lock and all
    facts = [{"S": it["S"], "value": it["truth"], "kind": "wiki"} for it in items]
    glue_exam = s265.exam(
        glue, model, char_table, tok, bank_can, tape, facts, pad_id, V, device, k, max_new,
        locked=True,
    )
    facts_clean = [{"S": it["S"], "value": it["truth"], "kind": "wiki"} for it in clean_items]
    glue_clean = s265.exam(
        glue, model, char_table, tok, bank_can, tape, facts_clean, pad_id, V, device, k, max_new,
        locked=True,
    )
    log(f"  glue span-lock: contradicted EM={glue_exam['em']:.3f} clean EM={glue_clean['em']:.3f}")

    # causal: drop every liar slot, lookup must recover — otherwise the exam is broken elsewhere
    tape_nolie = tape.copy()
    n_dropped = 0
    for it in items:
        for lie in set(it["lies"]):
            n_dropped += tape_nolie.drop_value(lie)
    recovered = arm_scores(glue, bank_can, tok, tape_nolie, items, pad_id, k)
    log(f"  liars removed ({n_dropped} slots): lookup={recovered['lookup_top1']:.3f}")

    chance = 1.0 / max(1, n_wit)
    g_lookup_fails = contra["lookup_top1"] <= 0.60          # validity: lookup must not suffice
    g_liar_causal = recovered["lookup_top1"] >= contra["lookup_top1"] + 0.20
    g_clean_ok = clean["lookup_top1"] >= 0.70
    g_witnesses_reachable = contra["witness_recall"] >= 0.75
    g_majority_works = contra["majority"] >= 0.80
    g_aggregation_beats_lookup = contra["majority"] >= contra["lookup_top1"] + 0.20
    g_glue_aggregates = glue_exam["em"] >= contra["majority"] - 0.10

    valid = g_clean_ok and g_witnesses_reachable and g_liar_causal
    if not valid:
        overall = "LYING_TAPE_INVALID"
    elif not g_lookup_fails:
        overall = "LOOKUP_SUFFICES"       # the exam does not need a mind after all
    elif g_majority_works and g_aggregation_beats_lookup and g_glue_aggregates:
        overall = "GLUE_ALREADY_AGGREGATES"
    elif g_majority_works and g_aggregation_beats_lookup:
        overall = "AGGREGATION_NEEDED"    # the target 271 has to learn
    else:
        overall = "LYING_TAPE_NO"

    out = {
        "stage": 270,
        "overall": overall,
        "smoke": args.smoke,
        "seed": SEED,
        "trunk": trunk_ckpt.name,
        "glue_ckpt": ck.name if ck.is_file() else None,
        "fp_version": fp_version(),
        "trained_parameters": 0,
        "n_subjects": n_subj,
        "witnesses_per_subject": n_wit,
        "liars_per_subject": n_liars,
        "topk": k,
        "tape_slots": len(tape.values),
        "chance_per_witness": chance,
        "gates": {
            "G_clean_ok": g_clean_ok,
            "G_witnesses_reachable": g_witnesses_reachable,
            "G_liar_causal": g_liar_causal,
            "G_lookup_fails": g_lookup_fails,
            "G_majority_works": g_majority_works,
            "G_aggregation_beats_lookup": g_aggregation_beats_lookup,
            "G_glue_aggregates": g_glue_aggregates,
        },
        "contradicted": {kk: vv for kk, vv in contra.items() if kk != "rows"},
        "clean": {kk: vv for kk, vv in clean.items() if kk != "rows"},
        "liars_removed": {kk: vv for kk, vv in recovered.items() if kk != "rows"},
        "glue_span_lock": {
            "em_contradicted": glue_exam["em"],
            "em_clean": glue_clean["em"],
            "verbatim": glue_exam["verbatim"],
            "open_recall": glue_exam["open_recall"],
        },
        "samples": contra["rows"],
        "note": (
            "Several slots speak about one subject and disagree; every witness is keyed alike, so "
            "top-1 retrieval lands on the liar as often as the geometry allows and only reading "
            "several slots answers the question. G_lookup_fails is a validity gate, not a result: "
            "if lookup already answers, the liar's key is distinguishable and the exam leaks. "
            "Nothing is trained here — the point is to establish the number an aggregating mind "
            "would have to beat, and to see whether the 265/268 pipeline aggregates by accident."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 270 lying tape\n\n**{overall}** · {n_wit} witnesses, {n_liars} lying · "
        f"{n_subj} subjects · {len(tape.values)} slots · trained params **0**"
        f"{' · SMOKE' if args.smoke else ''}\n\n"
        f"| arm | contradicted | clean |\n|---|---:|---:|\n"
        f"| A lookup (top-1 slot) | **{contra['lookup_top1']:.3f}** | {clean['lookup_top1']:.3f} |\n"
        f"| B majority over witnesses | **{contra['majority']:.3f}** | — |\n"
        f"| C similarity-weighted | {contra['sim_weighted']:.3f} | — |\n"
        f"| D glue span-lock | {glue_exam['em']:.3f} | {glue_clean['em']:.3f} |\n\n"
        f"- liar is top-1: **{contra['liar_is_top1']:.3f}**, witness recall {contra['witness_recall']:.3f}\n"
        f"- liars removed → lookup {recovered['lookup_top1']:.3f} (was {contra['lookup_top1']:.3f})\n\n"
        f"## Gates (read G_lookup_fails first — it is validity, not result)\n\n"
        + "".join(f"- {kk}: **{vv}**\n" for kk, vv in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
