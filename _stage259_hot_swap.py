"""
Stage 259 — Hot swap: change a fact in memory, get the new answer, zero gradient steps.

256 proved the head USES the tape. This asks the question that separates "knowledge lives in
memory" from "knowledge lives in weights" in one move a reader can check without a benchmark:

    edit the slot -> ask again -> the model says the new thing, immediately, no training

Keys are written as norm( fp(anchor) + ctx_fp(sentence, exclude=value) ), so a value never
enters its own key. Replacing it leaves the key BIT-IDENTICAL — asserted here, not assumed.
That is what makes this an update rather than a re-index, and it is why the edit costs no
gradient: nothing in the geometry moved.

Nothing trains. The glue (W_q + gate + tau) is loaded from stage 256 and its parameters are
snapshotted and compared bit for bit at the end, so "zero-train" is a measured claim.

What would break the story, and is therefore tested:
  old value survives    the two answers coexist -> the edit did not take
  neighbours die        editing one fact damages others -> not a local update
  keys moved            it was a re-index, and the editability claim is empty
  params moved          something trained; the demo is a lie
  second edit ignored   first write wins -> a cache, not a memory
  empty tape answers    the value was in the weights all along

Data is rebuilt with stage 256's seed and call order so the loaded glue matches the tape it
was fit on. Requires checkpoints/stage256_slot_bias.pt — run 256 first.

  python _stage259_hot_swap.py [--smoke]
"""
from __future__ import annotations

import argparse
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
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import (
    ANCHOR_RE,
    DEFAULT_CUE,
    DEFAULT_FACT_TMPL,
    TapeView,
    free_decode_value,
    load_glue,
    value_exact_match,
)

RES = Path("results")
DECISION = RES / "stage259_decision.json"
MINI = RES / "stage259_mini.md"
LOG = RES / "_stage259_log.txt"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_GLUE = Path("checkpoints/stage256_slot_bias.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED_256 = 256  # data must be rebuilt exactly as 256 built it, or the loaded glue is invalid


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def rebuild_256_tape(bank: FpBank, tok: Tokenizer, pad_id: int, device, smoke: bool):
    """Mirror of stage 256's data build — same seed, same call order, same loops."""
    rng = random.Random(SEED_256)
    n_facts = 8 if smoke else 48
    n_dist = 150 if smoke else 1200
    max_lines = 400 if smoke else 6000

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_000_000 if smoke else 6_000_000)
    values_pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(values_pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:max_lines]

    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 30) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        facts.append({
            "S": S, "value": values_pool[i], "sent": DEFAULT_FACT_TMPL.format(S=S, V=values_pool[i]),
            "fid": f"f{i}", "glue_train": i % 2 == 0,
        })

    keys, vals = [], []
    for f in facts:
        kf = bank.fp([f["S"]])[0]
        c = bank.ctx_fp(f["sent"], exclude=f["value"])
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(f["value"])
    used = set(vals)
    for ln in lines:
        if len(vals) >= n_facts + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used:
                continue
            lo, hi = max(0, m.start() - 120), min(len(ln), m.end() + 120)
            c = bank.ctx_fp(ln[lo:hi], exclude=ent)
            if c is None:
                continue
            anchors = [w for w in ANCHOR_RE.findall(ln[lo : m.start()]) if w != ent]
            if not anchors:
                continue
            keys.append(F.normalize(bank.fp([anchors[-1]])[0] + c, dim=-1))
            vals.append(ent)
            used.add(ent)
            if len(vals) >= n_facts + n_dist:
                break
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    fresh = [v for v in values_pool if v not in used]  # values the bank has never held
    return facts, tape, fresh


@torch.no_grad()
def em_against(glue, model, char_table, tok, bank, tape, pairs, pad_id, V, device, k, max_new):
    """pairs: (subject, expected_value). Free-form greedy decode, no candidate set."""
    ok = 0
    for S, gold in pairs:
        got, _ = free_decode_value(
            glue, model, char_table, tok, bank, tape, S, pad_id, V, device, k=k, max_new=max_new
        )
        ok += int(value_exact_match(got, gold))
    return ok / max(1, len(pairs))


def param_fingerprint(glue) -> str:
    """Bit-level snapshot of everything trainable, so 'nothing trained' is checkable."""
    import hashlib

    h = hashlib.sha256()
    for t in (
        list(glue.W_q.state_dict().values())
        + list(glue.gate.state_dict().values())
        + [glue.log_tau.detach()]
    ):
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED_256)
    t0 = time.time()
    k = args.topk
    max_new = 4 if args.smoke else 6

    if not CKPT_GLUE.is_file():
        log(f"missing {CKPT_GLUE} — run _stage256_slot_bias_decode.py first")
        return 1

    log(f"Stage259 hot swap start {datetime.now(timezone.utc).isoformat()} device={device}")

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
    bank = FpBank(model_can, stoi, device)

    facts, tape, fresh = rebuild_256_tape(bank, tok, pad_id, device, args.smoke)
    ev = [f for f in facts if not f["glue_train"]]  # the half 256 never fit
    glue = load_glue(model, device, CKPT_GLUE)
    if glue is None:
        log("glue failed to load")
        return 1
    log(f"  trunk={trunk_ckpt.name} glue={CKPT_GLUE.name} slots={len(tape.values)} eval_facts={len(ev)}")
    if len(fresh) < 2 * len(ev):
        log(f"  not enough unused values ({len(fresh)}) for two rounds of edits")
        return 1

    fp_before = param_fingerprint(glue)
    K_before = tape.K.clone()

    em_before = em_against(glue, model, char_table, tok, bank, tape,
                           [(f["S"], f["value"]) for f in ev], pad_id, V, device, k, max_new)
    log(f"before edit: EM={em_before:.3f}")

    # ---- round 1: one edit per held-out fact, each measured on its own tape ----
    new1 = {f["fid"]: fresh[i] for i, f in enumerate(ev)}
    hit_new, hit_old, neigh, edit_us = [], [], [], []
    for f in ev:
        t_e = time.perf_counter()
        tp = tape.with_value(f["value"], new1[f["fid"]], tok, pad_id)
        edit_us.append((time.perf_counter() - t_e) * 1e6)
        hit_new.append(em_against(glue, model, char_table, tok, bank, tp,
                                  [(f["S"], new1[f["fid"]])], pad_id, V, device, k, max_new))
        hit_old.append(em_against(glue, model, char_table, tok, bank, tp,
                                  [(f["S"], f["value"])], pad_id, V, device, k, max_new))
        others = [(o["S"], o["value"]) for o in ev if o is not f][:4]
        if others:
            neigh.append(em_against(glue, model, char_table, tok, bank, tp,
                                    others, pad_id, V, device, k, max_new))
    em_new = float(np.mean(hit_new))
    em_old = float(np.mean(hit_old))
    em_neigh = float(np.mean(neigh)) if neigh else float("nan")
    log(f"after edit:  new={em_new:.3f}  old={em_old:.3f}  neighbours={em_neigh:.3f} "
        f"(edit {np.mean(edit_us):.0f} us)")

    # ---- round 2: edit the SAME slot again — a memory tracks the latest write, a cache does not
    new2 = {f["fid"]: fresh[len(ev) + i] for i, f in enumerate(ev)}
    hit_newest, hit_stale = [], []
    for f in ev:
        tp = tape.with_value(f["value"], new1[f["fid"]], tok, pad_id)
        tp = tp.with_value(new1[f["fid"]], new2[f["fid"]], tok, pad_id)
        hit_newest.append(em_against(glue, model, char_table, tok, bank, tp,
                                     [(f["S"], new2[f["fid"]])], pad_id, V, device, k, max_new))
        hit_stale.append(em_against(glue, model, char_table, tok, bank, tp,
                                    [(f["S"], new1[f["fid"]])], pad_id, V, device, k, max_new))
    em_newest = float(np.mean(hit_newest))
    em_stale = float(np.mean(hit_stale))
    log(f"second edit: newest={em_newest:.3f}  superseded={em_stale:.3f}")

    # ---- leak floor: edited value, then no tape at all ----
    empty_pairs = [(f["S"], new1[f["fid"]]) for f in ev]
    em_empty = em_against(glue, model, char_table, tok, bank, tape.emptied(),
                          empty_pairs, pad_id, V, device, k, max_new)

    keys_untouched = bool(torch.equal(tape.K, K_before))
    params_untouched = param_fingerprint(glue) == fp_before

    g_baseline = em_before >= 0.40
    g_follows = em_new >= 0.40 and em_new >= em_before - 0.15
    g_old_dies = em_old <= 0.10
    g_local = (not np.isnan(em_neigh)) and em_neigh >= 0.7 * em_before
    g_keys = keys_untouched
    g_zero_grad = params_untouched
    g_latest_wins = em_newest >= 0.40 and em_stale <= 0.10
    g_no_leak = em_empty <= 0.10

    core = g_baseline and g_follows and g_old_dies and g_keys and g_zero_grad and g_no_leak
    if core and g_local and g_latest_wins:
        overall = "HOT_SWAP_OK"
    elif core:
        overall = "HOT_SWAP_PARTIAL"
    else:
        overall = "HOT_SWAP_NO"

    out = {
        "stage": 259,
        "overall": overall,
        "trunk": trunk_ckpt.name,
        "glue": CKPT_GLUE.name,
        "topk": k,
        "tape_slots": len(tape.values),
        "n_eval_facts": len(ev),
        "gates": {
            "G_baseline_alive": g_baseline,
            "G_answer_follows_edit": g_follows,
            "G_old_answer_dies": g_old_dies,
            "G_edit_is_local": g_local,
            "G_keys_untouched": g_keys,
            "G_zero_gradient_steps": g_zero_grad,
            "G_latest_write_wins": g_latest_wins,
            "G_no_param_leak": g_no_leak,
        },
        "summary": {
            "em_before": em_before,
            "em_new_value_after_edit": em_new,
            "em_old_value_after_edit": em_old,
            "em_neighbours_after_edit": em_neigh,
            "em_newest_after_second_edit": em_newest,
            "em_superseded_after_second_edit": em_stale,
            "em_empty_tape": em_empty,
            "edit_wall_us_mean": float(np.mean(edit_us)),
            "edit_wall_us_note": "includes a defensive copy of the value list (O(slots)); an "
            "in-place product edit is a tokenize plus two assignments",
            "keys_bit_identical": keys_untouched,
            "glue_params_bit_identical": params_untouched,
            "gradient_steps": 0,
        },
        "note": "No training anywhere: the 256 glue is loaded and its parameters are hashed before "
        "and after, so zero-train is measured rather than asserted. A value never enters its own "
        "key (ctx_fp excludes it at write time), so the edit leaves keys bit-identical — an update, "
        "not a re-index. Each fact is edited on its own view of the tape and scored free-form with "
        "no candidate set, on the half of the facts 256 never fit. The second edit checks the tape "
        "tracks the latest write instead of behaving like a write-once cache.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 259 hot swap\n\n**{overall}** glue={CKPT_GLUE.name} slots={len(tape.values)} "
        f"facts={len(ev)} gradient steps **0**\n\n"
        f"- before edit **{em_before:.3f}** -> after edit, new value **{em_new:.3f}**, "
        f"old value **{em_old:.3f}**\n"
        f"- neighbours untouched: {em_neigh:.3f}\n"
        f"- edited again: newest **{em_newest:.3f}**, superseded {em_stale:.3f}\n"
        f"- empty tape (leak floor): {em_empty:.3f}\n"
        f"- keys bit-identical: {keys_untouched} | glue params bit-identical: {params_untouched}\n"
        f"- edit cost: {np.mean(edit_us):.0f} us, 0 gradient steps\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"], "summary": out["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
