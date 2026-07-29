"""
Stage 205 — W3: targeted unlearning, provenance, contradiction audit.

Claim under test: because facts live in explicit fp slots, TapeLM can (a) delete one fact in O(1)
with provably zero collateral, (b) attribute every answer to the slot/source it came from, and
(c) flag contradictions instead of silently answering. A parametric GPT can do none of these
without gradients — and gradient unlearning damages what it should not touch.

Controls:
  - GPT is FIRST fine-tuned to actually memorize the same facts (otherwise "unlearning" is vacuous),
    then unlearned by gradient ascent on the target facts with EARLY STOP (minimal, fairest damage).
  - Collateral measured identically for both: retained-fact recall + next_tok on exam v3 items.
  - Honest note recorded: GPT+RAG could also delete from its index and give provenance, so vs RAG
    this axis is architectural; vs parametric GPT it is capability.

Gates:
  G_forget      curve target recall drops to <= chance+0.05 after slot delete
  G_no_collat   curve retained recall delta <= 0.02 AND curve next_tok delta == 0
  G_gpt_collat  GPT unlearning shows collateral (retained recall or next_tok drop > 0.02)
  G_prov        curve provenance attribution >= 0.90
  G_conflict    conflict detection >= 0.80 with false-positive <= 0.20

  python _stage205_unlearn_provenance.py
"""
from __future__ import annotations

import copy
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
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt

RES = Path("results")
DATA = Path("data")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
EXAM_V3 = DATA / "stage191_exam_v3.jsonl"
DECISION = RES / "stage205_decision.json"
MINI = RES / "stage205_mini.md"
LOG = RES / "_stage205_log.txt"

SEED = 205
CORPUS_CHARS = 20_000_000
N_FACTS = 60
N_TARGET = 20
N_CONFLICT = 20
N_FILLER = 300
N_NEXTTOK = 120
FT_STEPS = 800
FT_BATCH = 8
FT_LEN = 64
FT_LR = 3e-4
UNLEARN_LR = 5e-5
UNLEARN_MAX = 60
CHANCE = 0.25


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
    log(f"Stage205 start {datetime.now(timezone.utc).isoformat()}")
    log("W3: targeted unlearning / provenance / contradiction audit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bank = FpBank(model, stoi, device)
    log(f"curve loaded, frozen ({time.time()-t0:.0f}s)")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 300]
    values_pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5))
    rng.shuffle(values_pool)

    subs = [w for w in gen_fakes(set(values_pool), rng, N_FACTS + N_CONFLICT + 60) if len(w) >= 5]
    facts = []
    for i in range(N_FACTS):
        S, Vv = subs[i], values_pool[i]
        facts.append({"S": S, "value": Vv, "sent": f"{S} was appointed director of {Vv} in 1987 .", "fid": i})
    target = facts[:N_TARGET]
    retained = facts[N_TARGET:]
    tset = {f["S"] for f in target}

    # conflicting facts: same subject, two different values (for the audit test)
    conflicts = []
    for j in range(N_CONFLICT):
        S = subs[N_FACTS + j]
        v1, v2 = values_pool[N_FACTS + 2 * j], values_pool[N_FACTS + 2 * j + 1]
        conflicts.append({"S": S, "v1": v1, "v2": v2})
    log(f"facts={len(facts)} (target={len(target)} retained={len(retained)}) conflicts={len(conflicts)} ({time.time()-t0:.0f}s)")

    # ---------------- curve memory with provenance ----------------
    def slot_of(S, sent, value, fid, kind):
        k = bank.fp([S])[0]
        c = bank.ctx_fp(sent, exclude=value)
        key = F.normalize(k + c, dim=-1) if c is not None else k
        return {"key": key, "value": value, "src": fid, "kind": kind, "subject": S}

    slots = [slot_of(f["S"], f["sent"], f["value"], f["fid"], "fact") for f in facts]
    for j, cf in enumerate(conflicts):
        for tag, v in (("v1", cf["v1"]), ("v2", cf["v2"])):
            sent = f"{cf['S']} was appointed director of {v} in 1987 ."
            slots.append(slot_of(cf["S"], sent, v, 10_000 + j, f"conflict_{tag}"))
    for i, para in enumerate(paras[:N_FILLER]):
        m = ENT_RE.search(para)
        if not m:
            continue
        ent = m.group(1)
        c = bank.ctx_fp(para[: 400], exclude=ent)
        if c is None:
            continue
        slots.append(
            {"key": F.normalize(bank.fp([ent])[0] + c, dim=-1), "value": ent, "src": 20_000 + i, "kind": "filler", "subject": ent}
        )
    log(f"curve slots={len(slots)} ({time.time()-t0:.0f}s)")

    all_values = list(dict.fromkeys(s["value"] for s in slots))

    def curve_recall(fs, live):
        Kmat = torch.stack([s["key"] for s in live], 0)
        qrng = random.Random(SEED + 3)
        ok = 0
        for f in fs:
            q = bank.fp([f["S"]])[0]
            sims = (Kmat @ q).tolist()
            best = {}
            for s, sc in zip(live, sims):
                best[s["value"]] = max(best.get(s["value"], -9.9), sc)
            others = [x for x in all_values if x != f["value"]]
            qrng.shuffle(others)
            cands = [f["value"]] + others[:3]
            order = list(range(4))
            qrng.shuffle(order)
            shuf = [cands[i] for i in order]
            ok += int(int(np.argmax([best.get(c, -9.9) for c in shuf])) == order.index(0))
        return ok / max(1, len(fs))

    def curve_provenance(fs, live):
        Kmat = torch.stack([s["key"] for s in live], 0)
        ok = 0
        for f in fs:
            q = bank.fp([f["S"]])[0]
            top = int((Kmat @ q).argmax())
            ok += int(live[top]["src"] == f["fid"])
        return ok / max(1, len(fs))

    # ---------------- next_tok parity slice (collateral probe) ----------------
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
    nt = [it for it in items if it["type"] == "next_tok"][:N_NEXTTOK]

    def curve_next_tok():
        ok = 0
        for it in nt:
            sc = [span_logprob_x(model, char_table, pad_id, it["ctx_ids"], c, device) for c in it["cand_ids"]]
            ok += int(int(np.argmax(sc)) == it["gold_idx"])
        return ok / len(nt)

    def gpt_next_tok(gm):
        ok = 0
        for it in nt:
            sc = [gpt_span(gm, device, it["ctx_ids"], c) for c in it["cand_ids"]]
            ok += int(int(np.argmax(sc)) == it["gold_idx"])
        return ok / len(nt)

    # ---------------- curve: unlearn by slot delete ----------------
    live_all = slots
    c_tgt_before = curve_recall(target, live_all)
    c_ret_before = curve_recall(retained, live_all)
    c_nt_before = curve_next_tok()
    log(f"curve BEFORE: target={c_tgt_before:.3f} retained={c_ret_before:.3f} next_tok={c_nt_before:.3f} ({time.time()-t0:.0f}s)")

    td = time.time()
    live_after = [s for s in live_all if not (s["kind"] == "fact" and s["subject"] in tset)]
    del_secs = time.time() - td
    c_tgt_after = curve_recall(target, live_after)
    c_ret_after = curve_recall(retained, live_after)
    c_nt_after = curve_next_tok()
    prov = curve_provenance(retained, live_after)
    log(
        f"curve AFTER delete ({del_secs*1000:.1f} ms, {len(live_all)-len(live_after)} slots): "
        f"target={c_tgt_after:.3f} retained={c_ret_after:.3f} next_tok={c_nt_after:.3f} provenance={prov:.3f} ({time.time()-t0:.0f}s)"
    )

    # ---------------- curve: contradiction audit ----------------
    def conflict_flags(live, subjects, gap_thresh=0.02):
        Kmat = torch.stack([s["key"] for s in live], 0)
        out = []
        for S in subjects:
            q = bank.fp([S])[0]
            sims = (Kmat @ q).tolist()
            best = {}
            for s, sc in zip(live, sims):
                best[s["value"]] = max(best.get(s["value"], -9.9), sc)
            top = sorted(best.items(), key=lambda kv: -kv[1])[:2]
            out.append(len(top) == 2 and (top[0][1] - top[1][1]) < gap_thresh)
        return out

    det = conflict_flags(live_after, [c["S"] for c in conflicts])
    fp_flags = conflict_flags(live_after, [f["S"] for f in retained])
    conf_det = sum(det) / max(1, len(det))
    conf_fp = sum(fp_flags) / max(1, len(fp_flags))
    log(f"conflict audit: detection={conf_det:.3f} false_positive={conf_fp:.3f} ({time.time()-t0:.0f}s)")

    # ---------------- GPT parametric: memorize then gradient-unlearn ----------------
    gm = load_gpt(device)
    gm = copy.deepcopy(gm)
    gm.train()

    fact_ids = []
    for f in facts:
        fact_ids.append([i for i in tok.encode(f["sent"]).ids if i != pad_id])
    real_ids = [i for i in tok.encode(" ".join(paras[300:600])[:200_000]).ids if i != pad_id]

    def ft_batch(brng, only=None):
        rows = []
        for _ in range(FT_BATCH):
            if only is not None or brng.random() < 0.5:
                pool = only if only is not None else fact_ids
                seq = []
                while len(seq) < FT_LEN:
                    seq += pool[brng.randrange(len(pool))]
                rows.append(seq[:FT_LEN])
            else:
                s = brng.randrange(max(1, len(real_ids) - FT_LEN - 1))
                rows.append(real_ids[s : s + FT_LEN])
        return torch.tensor(rows, device=device)

    def gpt_fact_recall(fs):
        qrng = random.Random(SEED + 3)
        ok = 0
        for f in fs:
            ctx = [i for i in tok.encode(f"{f['S']} was appointed director of").ids if i != pad_id]
            others = [x for x in all_values if x != f["value"]]
            qrng.shuffle(others)
            cands = [f["value"]] + others[:3]
            order = list(range(4))
            qrng.shuffle(order)
            shuf = [cands[i] for i in order]
            sc = [gpt_span(gm, device, ctx, [i for i in tok.encode(" " + c).ids if i != pad_id]) for c in shuf]
            ok += int(int(np.argmax(sc)) == order.index(0))
        return ok / max(1, len(fs))

    opt = torch.optim.AdamW(gm.parameters(), lr=FT_LR, weight_decay=0.01)
    brng = random.Random(SEED + 11)
    for step in range(1, FT_STEPS + 1):
        x = ft_batch(brng)
        loss = gm(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            log(f"  gpt memorize step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
    gm.eval()
    g_tgt_before = gpt_fact_recall(target)
    g_ret_before = gpt_fact_recall(retained)
    g_nt_before = gpt_next_tok(gm)
    log(f"gpt AFTER memorize: target={g_tgt_before:.3f} retained={g_ret_before:.3f} next_tok={g_nt_before:.3f} ({time.time()-t0:.0f}s)")

    # gradient-ascent unlearning on target facts only, early stop at chance (minimal damage)
    tgt_ids = [[i for i in tok.encode(f["sent"]).ids if i != pad_id] for f in target]
    uopt = torch.optim.AdamW(gm.parameters(), lr=UNLEARN_LR)
    urng = random.Random(SEED + 13)
    gm.train()
    used = 0
    tu = time.time()
    for step in range(1, UNLEARN_MAX + 1):
        x = ft_batch(urng, only=tgt_ids)
        loss = -gm(input_ids=x, labels=x).loss  # ascent
        uopt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gm.parameters(), 1.0)
        uopt.step()
        used = step
        if step % 10 == 0:
            gm.eval()
            cur = gpt_fact_recall(target)
            log(f"  gpt unlearn step {step}: target={cur:.3f} ({time.time()-t0:.0f}s)")
            gm.train()
            if cur <= CHANCE + 0.05:
                break
    unl_secs = time.time() - tu
    gm.eval()
    g_tgt_after = gpt_fact_recall(target)
    g_ret_after = gpt_fact_recall(retained)
    g_nt_after = gpt_next_tok(gm)
    log(
        f"gpt AFTER unlearn ({used} grad steps, {unl_secs:.1f}s): target={g_tgt_after:.3f} "
        f"retained={g_ret_after:.3f} next_tok={g_nt_after:.3f} ({time.time()-t0:.0f}s)"
    )

    g_forget = c_tgt_after <= CHANCE + 0.05
    g_no_collat = abs(c_ret_after - c_ret_before) <= 0.02 and abs(c_nt_after - c_nt_before) < 1e-9
    g_gpt_collat = (g_ret_before - g_ret_after) > 0.02 or (g_nt_before - g_nt_after) > 0.02
    g_prov = prov >= 0.90
    g_conflict = conf_det >= 0.80 and conf_fp <= 0.20
    if g_forget and g_no_collat and g_prov and g_conflict and g_gpt_collat:
        overall = "UNLEARN_PROVENANCE_WIN"
    elif g_forget and g_no_collat and (g_prov or g_conflict):
        overall = "UNLEARN_PARTIAL"
    else:
        overall = "UNLEARN_NO"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "unlearn_provenance_audit_205",
        "overall": overall,
        "curve": {
            "target_recall": {"before": c_tgt_before, "after": c_tgt_after},
            "retained_recall": {"before": c_ret_before, "after": c_ret_after},
            "next_tok": {"before": c_nt_before, "after": c_nt_after},
            "provenance_attribution": prov,
            "delete_seconds": del_secs,
            "slots_deleted": len(live_all) - len(live_after),
        },
        "gpt_parametric": {
            "target_recall": {"after_memorize": g_tgt_before, "after_unlearn": g_tgt_after},
            "retained_recall": {"after_memorize": g_ret_before, "after_unlearn": g_ret_after},
            "next_tok": {"after_memorize": g_nt_before, "after_unlearn": g_nt_after},
            "unlearn_grad_steps": used,
            "unlearn_seconds": unl_secs,
        },
        "conflict_audit": {"detection": conf_det, "false_positive": conf_fp},
        "gates": {
            "g_forget": g_forget,
            "g_no_collateral": g_no_collat,
            "g_gpt_collateral": g_gpt_collat,
            "g_provenance": g_prov,
            "g_conflict": g_conflict,
        },
        "chance": CHANCE,
        "note": "vs parametric GPT this is capability (no gradient-free deletion, no attribution); "
        "vs GPT+RAG it is architectural — a RAG index can also delete and attribute",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage205 — targeted unlearning / provenance / audit",
                "",
                f"**Overall:** `{overall}`",
                "",
                "| metric | curve before | curve after | GPT after memorize | GPT after unlearn |",
                "|--------|--------------|-------------|--------------------|-------------------|",
                f"| target fact recall | {c_tgt_before:.3f} | **{c_tgt_after:.3f}** | {g_tgt_before:.3f} | {g_tgt_after:.3f} |",
                f"| retained fact recall | {c_ret_before:.3f} | **{c_ret_after:.3f}** | {g_ret_before:.3f} | {g_ret_after:.3f} |",
                f"| next_tok (collateral) | {c_nt_before:.3f} | **{c_nt_after:.3f}** | {g_nt_before:.3f} | {g_nt_after:.3f} |",
                "",
                f"- curve delete: {len(live_all)-len(live_after)} slots in {del_secs*1000:.1f} ms, no gradient",
                f"- GPT unlearn: {used} gradient steps, {unl_secs:.1f} s",
                f"- provenance attribution (curve): {prov:.3f}",
                f"- conflict audit: detection {conf_det:.3f}, false-positive {conf_fp:.3f}",
                f"- gates: forget={g_forget} no_collateral={g_no_collat} gpt_collateral={g_gpt_collat} prov={g_prov} conflict={g_conflict}",
            ]
        ),
        encoding="utf-8",
    )
    log(
        f"[205] {overall} | curve tgt {c_tgt_before:.2f}->{c_tgt_after:.2f} ret {c_ret_before:.2f}->{c_ret_after:.2f} "
        f"nt {c_nt_before:.3f}->{c_nt_after:.3f} | gpt tgt {g_tgt_before:.2f}->{g_tgt_after:.2f} ret {g_ret_before:.2f}->{g_ret_after:.2f} nt {g_nt_before:.3f}->{g_nt_after:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
