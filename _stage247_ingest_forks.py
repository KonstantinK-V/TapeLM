"""
Stage 247 — Fork map: where do unknown facts go?

One domain stream with planted novel facts. Three ingest policies:

  P_ce     : everything into CE (bindings enter weights)
  P_slots  : everything novel into slots; CE only on binding-stripped filler
  P_hop    : hop-similarity gate → slots if cos(fp(fact), hop_query) high;
             else skip slot; CE on binding-stripped filler (same as P_slots CE)

Then score each policy on:
  M_acquire   fact recall after ingest
  M_edit      overwrite one fact; target updates; retained collateral
  M_cf        after short code-domain CE on the *same* weights (P_ce) or
              query-shift+W (P_slots/P_hop); retained fact recall
  M_under    next_tok on exam v3 (understanding / language proxy)

Not a shipping trunk stage — map of forks for TapeLM evolution.

  python _stage247_ingest_forks.py [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _tapelm_ext import DomainAdapter

RES = Path("results")
DECISION = RES / "stage247_decision.json"
MINI = RES / "stage247_mini.md"
LOG = RES / "_stage247_log.txt"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
EXAM = Path("data/stage191_exam_v3.jsonl")
SEED = 247


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def mask_bindings(text: str, facts) -> str:
    out = text
    for f in facts:
        out = out.replace(f["sent"], f"The chronicle continues without naming the director.")
        out = out.replace(f["S"], "Someone")
        out = out.replace(f["value"], "somewhere")
    return out


def build_stream(paras, facts, rng: random.Random) -> str:
    """Interleave filler paragraphs with fact sentences."""
    chunks = []
    pi = 0
    for i, f in enumerate(facts):
        if pi < len(paras):
            chunks.append(paras[pi][:280])
            pi += 1
        chunks.append(f["sent"])
        if i % 2 == 0 and pi < len(paras):
            chunks.append(paras[pi][:200])
            pi += 1
    while pi < min(len(paras), len(facts) + 8):
        chunks.append(paras[pi][:240])
        pi += 1
    rng.shuffle(chunks)  # mild mix; facts still present
    # actually keep facts discoverable — don't shuffle facts away; rebuild ordered with inserts
    chunks = []
    for i, f in enumerate(facts):
        if i < len(paras):
            chunks.append(paras[i][:260])
        chunks.append(f["sent"])
    return " ".join(chunks)


def ce_train(model, flat, off, char_table, pad_id, device, steps, seed, tag, frozen_arc=True):
    m = copy.deepcopy(model)
    if frozen_arc:
        s213.set_train_mode(m, "upper")
    else:
        m.train()
        for p in m.parameters():
            p.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    rng = random.Random(seed)
    for step in range(1, steps + 1):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        pad = ids == pad_id
        logits, _, pred_loss = m.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(20, steps // 4) == 0:
            log(f"  {tag} step {step}: ce={float(ce):.3f}")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def gpt_ce_train(gm, flat, off, pad_id, device, steps, seed, tag):
    g = copy.deepcopy(gm)
    opt = torch.optim.AdamW(g.parameters(), lr=3e-4, weight_decay=0.01)
    rng = random.Random(seed)
    g.train()
    for step in range(1, steps + 1):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        loss = g(input_ids=ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(20, steps // 4) == 0:
            log(f"  {tag} step {step}: loss={float(loss):.3f}")
    g.eval()
    return g


def write_slots(bank, facts):
    return L.write_tape_bank(bank, facts)


def slot_recall(facts, all_values, bank, K, V, seed, W=None):
    return L.tape_recall(facts, all_values, bank, K, V, seed, W_bwd=W)


def gpt_recall(gm, tok, pad_id, facts, all_values, device, seed):
    return L.gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, seed)


def curve_fact_recall(model, stoi, device, facts, all_values, seed, K=None, V=None, bank=None, W=None):
    if bank is None:
        bank = FpBank(model, stoi, device)
    if K is None:
        K, V = write_slots(bank, facts)
    return slot_recall(facts, all_values, bank, K, V, seed, W=W), K, V, bank


def next_tok_acc(model, char_table, pad_id, items, device):
    if not items:
        return float("nan")
    ok = 0
    for it in items:
        sc = [span_logprob_x(model, char_table, pad_id, it["ctx_ids"], c, device) for c in it["cand_ids"]]
        ok += int(int(np.argmax(sc)) == it["gold_idx"])
    return ok / len(items)


def hop_gate(bank, facts, hop_query: str, thresh: float):
    """Admit fact if subject/ctx fp is close to hop query fp."""
    q = bank.ctx_fp(hop_query)
    if q is None:
        q = bank.fp(["organization"])[0]
    kept = []
    scores = []
    for f in facts:
        k = bank.fp([f["S"]])[0]
        c = bank.ctx_fp(f["sent"], exclude=f["value"])
        key = F.normalize(k + c, dim=-1) if c is not None else k
        s = float((key * q).sum())
        scores.append(s)
        if s >= thresh:
            kept.append(f)
    return kept, scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    n_facts = 10 if args.smoke else 24
    ce_steps = 80 if args.smoke else 600
    cf_steps = 60 if args.smoke else 400
    arc_steps = 40 if args.smoke else 300
    w_steps = 40 if args.smoke else 400
    n_next = 30 if args.smoke else 80
    max_lines = 200 if args.smoke else 4000
    core_n = 40 if args.smoke else 200
    hop_thresh = 0.15  # relative: keep top half by similarity

    log(f"Stage247 start {datetime.now(timezone.utc).isoformat()} device={device}")
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    Vtok = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, Vtok).to(device)

    model0 = SelfModelXL(n_char, Vtok).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model0.eval()
    for p in model0.parameters():
        p.requires_grad_(False)
    bank0 = FpBank(model0, stoi, device)
    gm0 = copy.deepcopy(load_gpt(device))
    gm0.eval()

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(3_000_000 if args.smoke else 12_000_000)
    values_pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5))
    rng.shuffle(values_pool)
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 180]
    core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:core_n]
    F_can = s221.fp_matrix(bank0, core)

    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 30) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[i]
        facts.append(
            {
                "S": S,
                "value": Vv,
                "sent": f"{S} was appointed director of {Vv} in 1987 .",
                "fid": i,
            }
        )
    all_values = [f["value"] for f in facts] + values_pool[n_facts : n_facts + 60]
    # hop theme: organization / director chain
    hop_query = "In the report the organization appointed a new director linked to governance."
    _, hop_scores = hop_gate(bank0, facts, hop_query, thresh=-1.0)
    order = sorted(range(len(facts)), key=lambda i: hop_scores[i], reverse=True)
    n_keep = max(2, len(facts) // 2)
    hop_set = {facts[i]["fid"] for i in order[:n_keep]}
    facts_hop = [f for f in facts if f["fid"] in hop_set]
    facts_nonhop = [f for f in facts if f["fid"] not in hop_set]
    log(f"facts={len(facts)} hop_admit={len(facts_hop)} hop_score_mean={float(np.mean(hop_scores)):.3f}")

    stream = build_stream(paras, facts, rng)
    stream_masked = mask_bindings(stream, facts)
    flat_full, off_full = s213.build_flat_from_text(stream, tok, pad_id, max_lines=max_lines, min_line_len=16)
    flat_mask, off_mask = s213.build_flat_from_text(stream_masked, tok, pad_id, max_lines=max_lines, min_line_len=16)

    items = []
    if EXAM.exists():
        with EXAM.open(encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                if it.get("type") == "next_tok":
                    items.append(it)
                if len(items) >= n_next:
                    break

    code = s227.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(code, tok, pad_id, max_lines=max_lines, min_line_len=20)

    results = {}

    # ---------- P_ce: all into CE (use GPT as parametric carrier — clear CF story) ----------
    log("P_ce: GPT CE on full stream (bindings in weights)")
    gm_ce = gpt_ce_train(gm0, flat_full, off_full, pad_id, device, ce_steps, SEED + 3, "P_ce")
    acq_ce = gpt_recall(gm_ce, tok, pad_id, facts, all_values, device, SEED)
    # edit: gradient ascent on one target fact (205-lite) then measure
    tgt = facts[: max(2, n_facts // 5)]
    ret = facts[len(tgt) :]
    tgt_ids = [[i for i in tok.encode(f["sent"]).ids if i != pad_id] for f in tgt]
    g_edit = copy.deepcopy(gm_ce)
    opt = torch.optim.AdamW(g_edit.parameters(), lr=5e-5)
    g_edit.train()
    for step in range(1, (20 if args.smoke else 40) + 1):
        x = L.ft_batch(random.Random(SEED + step), tgt_ids, [], 4, 64, device, mix_real=False)
        loss = -g_edit(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    g_edit.eval()
    edit_tgt = gpt_recall(g_edit, tok, pad_id, tgt, all_values, device, SEED)
    edit_ret = gpt_recall(g_edit, tok, pad_id, ret, all_values, device, SEED)
    # CF: continue CE on code
    gm_cf = gpt_ce_train(gm_ce, flat_c, off_c, pad_id, device, cf_steps, SEED + 4, "P_ce_cf")
    cf_ce = gpt_recall(gm_cf, tok, pad_id, facts, all_values, device, SEED)
    under_ce = L.gpt_next_tok(gm_cf, items, device)
    results["P_ce"] = {
        "acquire": acq_ce,
        "edit_target_after_unlearn": edit_tgt,
        "edit_retained": edit_ret,
        "edit_collateral": abs(edit_ret - gpt_recall(gm_ce, tok, pad_id, ret, all_values, device, SEED)),
        "cf_retain": cf_ce,
        "cf_drop": acq_ce - cf_ce,
        "understand_next_tok": under_ce,
        "carrier": "gpt_weights",
    }
    log(f"  P_ce acq={acq_ce:.3f} cf={cf_ce:.3f} edit_ret={edit_ret:.3f} under={under_ce:.3f}")

    # ---------- P_slots: all novel → slots; CE on masked filler (curve upper) ----------
    log("P_slots: write all facts to slots; CE on masked stream")
    K_all, V_all = write_slots(bank0, facts)
    m_slots = ce_train(model0, flat_mask, off_mask, char_table, pad_id, device, ce_steps, SEED + 5, "P_slots", True)
    bank_s = FpBank(m_slots, stoi, device)
    # acquire via slots (canonical keys; query on frozen can bank)
    acq_s = slot_recall(facts, all_values, bank0, K_all, V_all, SEED)
    # edit: rewrite one slot value
    K_ed = K_all.clone()
    V_ed = list(V_all)
    edit_i = 0
    new_val = values_pool[n_facts + 3]
    old_val = V_ed[edit_i]
    V_ed[edit_i] = new_val
    # rebuild that key with new sentence
    f_ed = dict(facts[edit_i])
    f_ed["value"] = new_val
    f_ed["sent"] = f"{f_ed['S']} was appointed director of {new_val} in 1987 ."
    k = bank0.fp([f_ed["S"]])[0]
    c = bank0.ctx_fp(f_ed["sent"], exclude=new_val)
    K_ed[edit_i] = F.normalize(k + c, dim=-1) if c is not None else k
    facts_ed_tgt = [f_ed]
    facts_ed_old = [facts[edit_i]]
    facts_ret = facts[1:]
    edit_new = slot_recall(facts_ed_tgt, all_values + [new_val], bank0, K_ed, V_ed, SEED)
    edit_old = slot_recall(facts_ed_old, all_values, bank0, K_ed, V_ed, SEED)
    edit_ret_s = slot_recall(facts_ret, all_values, bank0, K_ed, V_ed, SEED)
    # CF: code shift + W, slots unchanged
    model_b = s221.finetune_arc_enc(m_slots, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 6)
    bank_b = FpBank(model_b, stoi, device)
    W, align = s221.train_remap(
        DomainAdapter(256).to(device), s221.fp_matrix(bank_b, core), F_can, rng, w_steps, device
    )
    cf_s = slot_recall(facts, all_values, bank_b, K_all, V_all, SEED, W=W)
    under_s = next_tok_acc(m_slots, char_table, pad_id, items, device)
    results["P_slots"] = {
        "acquire": acq_s,
        "edit_new_ok": edit_new,
        "edit_old_gone": 1.0 - edit_old,
        "edit_retained": edit_ret_s,
        "edit_collateral": abs(edit_ret_s - slot_recall(facts_ret, all_values, bank0, K_all, V_all, SEED)),
        "cf_retain": cf_s,
        "cf_drop": acq_s - cf_s,
        "understand_next_tok": under_s,
        "W_align": align,
        "carrier": "slots+masked_CE",
    }
    log(f"  P_slots acq={acq_s:.3f} cf={cf_s:.3f} edit_new={edit_new:.3f} under={under_s:.3f}")

    # ---------- P_hop: hop-sim → slots; CE masked ----------
    log("P_hop: hop-similar facts → slots only; CE masked")
    K_h, V_h = write_slots(bank0, facts_hop) if facts_hop else (torch.zeros(1, 256, device=device), [])
    m_hop = ce_train(model0, flat_mask, off_mask, char_table, pad_id, device, ce_steps, SEED + 7, "P_hop", True)
    acq_hop_in = slot_recall(facts_hop, all_values, bank0, K_h, V_h, SEED) if facts_hop else 0.0
    # non-hop facts should NOT be in slots → low recall if only hop bank
    acq_hop_out = (
        slot_recall(facts_nonhop, all_values, bank0, K_h, V_h, SEED) if facts_nonhop and facts_hop else 0.0
    )
    # edit on hop bank
    if len(facts_hop) >= 2:
        K_he = K_h.clone()
        V_he = list(V_h)
        V_he[0] = values_pool[n_facts + 5]
        f_h = dict(facts_hop[0])
        f_h["value"] = V_he[0]
        f_h["sent"] = f"{f_h['S']} was appointed director of {V_he[0]} in 1987 ."
        k = bank0.fp([f_h["S"]])[0]
        c = bank0.ctx_fp(f_h["sent"], exclude=V_he[0])
        K_he[0] = F.normalize(k + c, dim=-1) if c is not None else k
        edit_new_h = slot_recall([f_h], all_values + [V_he[0]], bank0, K_he, V_he, SEED)
        edit_ret_h = slot_recall(facts_hop[1:], all_values, bank0, K_he, V_he, SEED)
        coll_h = abs(edit_ret_h - slot_recall(facts_hop[1:], all_values, bank0, K_h, V_h, SEED))
    else:
        edit_new_h, edit_ret_h, coll_h = float("nan"), float("nan"), float("nan")
    model_hb = s221.finetune_arc_enc(m_hop, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 8)
    bank_hb = FpBank(model_hb, stoi, device)
    Wh, align_h = s221.train_remap(
        DomainAdapter(256).to(device), s221.fp_matrix(bank_hb, core), F_can, rng, w_steps, device
    )
    cf_h = slot_recall(facts_hop, all_values, bank_hb, K_h, V_h, SEED, W=Wh) if facts_hop else 0.0
    under_h = next_tok_acc(m_hop, char_table, pad_id, items, device)
    results["P_hop"] = {
        "acquire_admitted": acq_hop_in,
        "acquire_rejected_should_be_low": acq_hop_out,
        "edit_new_ok": edit_new_h,
        "edit_retained": edit_ret_h,
        "edit_collateral": coll_h,
        "cf_retain": cf_h,
        "cf_drop": acq_hop_in - cf_h,
        "understand_next_tok": under_h,
        "n_admitted": len(facts_hop),
        "n_rejected": len(facts_nonhop),
        "W_align": align_h,
        "carrier": "hop_gated_slots+masked_CE",
    }
    log(
        f"  P_hop in={acq_hop_in:.3f} out={acq_hop_out:.3f} cf={cf_h:.3f} "
        f"edit_new={edit_new_h} under={under_h:.3f}"
    )

    # ---------- Fork recommendation ----------
    # Prefer slots/hop if: better CF retain than CE, lower edit collateral, under not much worse
    ce, sl, hp = results["P_ce"], results["P_slots"], results["P_hop"]
    g_slots_cf = sl["cf_retain"] >= ce["cf_retain"] + 0.15
    g_slots_edit = sl["edit_collateral"] <= 0.05
    g_hop_select = hp["acquire_admitted"] >= 0.70 and hp["acquire_rejected_should_be_low"] <= hp["acquire_admitted"] - 0.20
    g_under_ok = (
        sl["understand_next_tok"] + 0.05 >= ce.get("understand_next_tok", 0)
        or sl["understand_next_tok"] >= 0.55
    )
    if g_slots_cf and g_slots_edit and g_hop_select:
        overall = "INGEST_FORK_SLOTS_AND_HOP"
    elif g_slots_cf and g_slots_edit:
        overall = "INGEST_FORK_SLOTS_BEATS_CE"
    elif g_hop_select:
        overall = "INGEST_FORK_HOP_SELECTIVE"
    else:
        overall = "INGEST_FORK_MIXED"

    out = {
        "stage": 247,
        "overall": overall,
        "gates": {
            "G_slots_cf_beats_ce_0p15": g_slots_cf,
            "G_slots_edit_low_collateral": g_slots_edit,
            "G_hop_admits_and_rejects": g_hop_select,
            "G_slots_under_not_ruined": g_under_ok,
        },
        "policies": results,
        "n_facts": n_facts,
        "ce_steps": ce_steps,
        "cf_steps": cf_steps,
        "note": (
            "Fork map only. P_ce=parametric GPT; P_slots/P_hop=TapeLM slots + masked CE. "
            "Evolution hint: keep bindings out of CE; use hop-sim as admission, not as CE curriculum."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 247 ingest forks\n\n**{overall}**\n\n"
        f"| policy | acquire | cf_retain | edit_collateral | under |\n"
        f"|--------|---------|-----------|-----------------|-------|\n"
        f"| P_ce | {ce['acquire']:.2f} | {ce['cf_retain']:.2f} | {ce['edit_collateral']:.2f} | {ce['understand_next_tok']:.2f} |\n"
        f"| P_slots | {sl['acquire']:.2f} | {sl['cf_retain']:.2f} | {sl['edit_collateral']:.2f} | {sl['understand_next_tok']:.2f} |\n"
        f"| P_hop | {hp['acquire_admitted']:.2f} (out {hp['acquire_rejected_should_be_low']:.2f}) | "
        f"{hp['cf_retain']:.2f} | {hp['edit_collateral']} | {hp['understand_next_tok']:.2f} |\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
