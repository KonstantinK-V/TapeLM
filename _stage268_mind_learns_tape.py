"""
Stage 268 — Unfreeze the mind: learn the tape *procedure*, not a fixed tape.

265 proved span-lock decode. After 256 the trunk stayed frozen as a *measurement*
discipline (slot delete kills the answer → fact not in weights). That discipline
quietly became architecture. The product invariant is weaker and sharper:

    no fact written *after deployment* enters the weights.

It does not require weights never change. Continual (253–255) already trained
upper layers with arc_enc frozen. 268 restores that mode on the 265 exam:

  - set_train_mode(m, "upper"): fast/slow/head learn; arc_enc frozen (hash check)
  - tape rebuilt every ~200 steps (new subjects, values, keys) — nothing factual
    survives across rebuilds, so memorizing a bank cannot explain novel-tape EM
  - decode = 265 span-lock, unchanged

Gates (priority):
  G_novel_tape        — EM on a never-seen tape ≥ EM on last train tape − 0.05
  G_arc_enc_frozen    — arc_enc hash unchanged
  G_beats_frozen_mind — trained upper beats init-upper with the same glue on novel tape
  G_no_param_leak     — empty tape EM ≤ 0.10
  G_slot_delete       — target dies, others live
  G_lang_intact       — hold CE does not rise

Verdict: MIND_LEARNS_TAPE_OK / _PARTIAL / _NO.

  python _stage268_mind_learns_tape.py --smoke
  python _stage268_mind_learns_tape.py          # night: 8000 steps, ~40 tapes
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
import _stage265_span_lock as s265
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import (
    ANCHOR_RE,
    DEFAULT_CUE,
    DEFAULT_FACT_TMPL,
    SlotBias,
    TapeView,
    hidden_and_logits,
)

RES = Path("results")
DECISION = RES / "stage268_decision.json"
MINI = RES / "stage268_mini.md"
LOG = RES / "_stage268_log.txt"
DECISION_265 = RES / "stage265_decision.json"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage268_mind_learns_tape.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 268

CUE = DEFAULT_CUE
FACT_TMPL = DEFAULT_FACT_TMPL
PLACEHOLDER = s265.PLACEHOLDER


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


def arc_enc_hash(model: SelfModelXL) -> str:
    h = hashlib.sha256()
    for _, t in sorted(model.arc_enc.state_dict().items()):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def published_265_em() -> float | None:
    if not DECISION_265.is_file():
        return None
    try:
        d = json.loads(DECISION_265.read_text(encoding="utf-8"))
        arms = d.get("arms") or {}
        b = arms.get("B_soft_locked") or {}
        return float(b.get("em_text", b.get("em", float("nan"))))
    except Exception:
        return None


def build_tape(
    *,
    bank_can: FpBank,
    tok,
    pad_id: int,
    device,
    rng: random.Random,
    values_pool: list[str],
    lines: list[str],
    used: set[str],
    n_facts: int,
    n_nonsense: int,
    n_dist: int,
) -> dict:
    """Fresh planted facts + distractors. Subjects/values drawn from unused pool."""
    available = [w for w in values_pool if w not in used and len(w) >= 5]
    if len(available) < n_facts + n_dist // 4:
        # wrap: allow reuse of distractor pool but keep planted values novel when possible
        available = [w for w in values_pool if len(w) >= 5]
    rng.shuffle(available)

    subs = [w for w in gen_fakes(set(used) | set(available), rng, n_facts + n_nonsense + 80) if len(w) >= 5]
    subs = [w for w in dict.fromkeys(subs) if w not in used]
    fake_vals = [
        w for w in gen_fakes(set(used) | set(available) | set(subs), rng, n_nonsense + 40)
        if len(w) >= 6 and w not in subs and w not in used
    ]
    fake_vals = list(dict.fromkeys(fake_vals))[:n_nonsense]
    if len(subs) < n_facts + len(fake_vals) or len(available) < n_facts:
        raise RuntimeError(
            f"tape pool exhausted: subs={len(subs)} avail={len(available)} "
            f"need facts={n_facts} nonsense={n_nonsense}"
        )

    facts = []
    for i in range(n_facts):
        facts.append({
            "S": subs[i],
            "value": available[i],
            "sent": FACT_TMPL.format(S=subs[i], V=available[i]),
            "glue_train": i % 2 == 0,
            "kind": "wiki",
        })
        used.add(available[i])
        used.add(subs[i])
    for j, fv in enumerate(fake_vals):
        S = subs[n_facts + j]
        facts.append({
            "S": S, "value": fv, "sent": FACT_TMPL.format(S=S, V=fv),
            "glue_train": False, "kind": "nonsense",
        })
        used.add(fv)
        used.add(S)

    fit_facts = [f for f in facts if f["glue_train"]]
    eval_wiki = [f for f in facts if not f["glue_train"] and f["kind"] == "wiki"]
    eval_non = [f for f in facts if f["kind"] == "nonsense"]
    eval_facts = eval_wiki + eval_non

    keys, vals = [], []
    pair_q, pair_slot = [], []
    for f in facts:
        kf = bank_can.fp([f["S"]])[0]
        c = bank_can.ctx_fp(f["sent"], exclude=f["value"])
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(f["value"])
    used_vals = set(vals)
    for ln in lines:
        if len(vals) >= len(facts) + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used_vals:
                continue
            lo, hi = max(0, m.start() - 120), min(len(ln), m.end() + 120)
            c = bank_can.ctx_fp(ln[lo:hi], exclude=ent)
            if c is None:
                continue
            anchors = [w for w in ANCHOR_RE.findall(ln[lo: m.start()]) if w != ent]
            if not anchors:
                continue
            keys.append(F.normalize(bank_can.fp([anchors[-1]])[0] + c, dim=-1))
            cq = bank_can.ctx_fp(ln[lo: m.start()])
            if cq is not None:
                pair_q.append(F.normalize(bank_can.fp([anchors[-1]])[0] + cq, dim=-1))
                pair_slot.append(len(vals))
            vals.append(ent)
            used_vals.add(ent)
            used.add(ent)
            if len(vals) >= len(facts) + n_dist:
                break

    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    nce_q = torch.stack(pair_q).to(device).float() if pair_q else None
    nce_slot = torch.tensor(pair_slot, device=device) if pair_slot else None
    return {
        "tape": tape,
        "fit_facts": fit_facts,
        "eval_facts": eval_facts,
        "eval_wiki": eval_wiki,
        "eval_non": eval_non,
        "nce_q": nce_q,
        "nce_slot": nce_slot,
        "n_slots": len(vals),
    }


def train_step(
    glue, model, char_table, tok, bank, pack, flat, off, train_docs, pad_id, V, device,
    *, k, gate_l1, nce_w, nce_tau, rng,
):
    tape = pack["tape"]
    fit = pack["fit_facts"]
    batch = [fit[rng.randrange(len(fit))] for _ in range(min(4, len(fit)))]
    l_fact, g_fact = s265.fact_batch(
        glue, model, char_table, tok, bank, tape, batch, pad_id, V, device, k, open_only=True,
    )
    ids = s251.sample_windows_docs(flat, off, 1, rng, pad_id, train_docs).to(device)
    l_prose, g_prose = s265.prose_batch(
        glue, model, char_table, tok, bank, tape, ids, pad_id, V, device, k, gate_l1,
    )
    l_nce = None
    nce_q, nce_slot = pack["nce_q"], pack["nce_slot"]
    if nce_q is not None and nce_w > 0:
        K_all = tape.K.float()
        sel = torch.randint(0, nce_q.size(0), (min(64, nce_q.size(0)),), device=device)
        gold = F.one_hot(nce_slot[sel], K_all.size(0)).bool()
        l_nce = nce_w * s265.nce_loss(glue, nce_q[sel], gold, K_all, nce_tau)
    parts = [x for x in (l_fact, l_prose, l_nce) if x is not None]
    if not parts:
        return None
    loss = parts[0]
    for p in parts[1:]:
        loss = loss + p
    return {
        "loss": loss,
        "loss_fact": float(l_fact) if l_fact is not None else None,
        "loss_prose": float(l_prose) if l_prose is not None else None,
        "gate_fact": g_fact,
        "gate_prose": g_prose,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=0, help="rebuild tape every N steps")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gate-l1", type=float, default=0.02)
    ap.add_argument("--nce-w", type=float, default=1.0)
    ap.add_argument("--nce-tau", type=float, default=0.05)
    ap.add_argument("--facts", type=int, default=0)
    ap.add_argument("--nonsense-facts", type=int, default=0)
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--lr-glue", type=float, default=3e-3)
    ap.add_argument("--lr-upper", type=float, default=3e-5)
    ap.add_argument("--open-thresh", type=float, default=0.5)
    ap.add_argument("--reopen-margin", type=float, default=0.1)
    ap.add_argument("--max-opens", type=int, default=1)
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    steps = args.steps or (400 if args.smoke else 8000)
    tape_period = args.tape_period or (100 if args.smoke else 200)
    n_facts = args.facts or (8 if args.smoke else 48)
    n_nonsense = args.nonsense_facts or (4 if args.smoke else 16)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_new = 6 if args.smoke else 12
    n_hold = 4 if args.smoke else 12
    max_lines = 400 if args.smoke else 6000
    k = args.topk
    n_tapes_plan = max(1, steps // tape_period)

    log(
        f"Stage268 mind-learns-tape start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"steps={steps} tape_period={tape_period} (~{n_tapes_plan} tapes) facts={n_facts} "
        f"distractors={n_dist} lr_glue={args.lr_glue} lr_upper={args.lr_upper}"
    )

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)["model"])
    s213.set_train_mode(model, "upper")
    arc_hash0 = arc_enc_hash(model)
    log(f"  trunk={trunk_ckpt.name} upper=TRAIN arc_enc=FROZEN hash0={arc_hash0[:16]}…")

    # Frozen snapshot of init upper (same glue later → isolates mind learning)
    model_init = SelfModelXL(n_char, V).to(device)
    model_init.load_state_dict(copy.deepcopy(model.state_dict()))
    model_init.eval()
    for p in model_init.parameters():
        p.requires_grad_(False)

    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)
    log(f"  fp_version={fp_version()} (canonical keys always from frozen P1)")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_000_000 if args.smoke else 8_000_000)
    values_pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(values_pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:max_lines]
    log(f"  entity pool={len(values_pool)} wiki_lines={len(lines)}")

    prose = "\n".join(lines + [PLACEHOLDER] * 32)
    flat, off = s213.build_flat_from_text(prose, tok, pad_id, max_lines=max_lines + 64, min_line_len=20)
    n_docs = len(off) - 1
    hold_docs = list(range(max(1, n_docs - max(2, n_docs // 20)), n_docs))
    train_docs = list(range(0, hold_docs[0]))
    hold_batches = s252.make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)
    base_hold = s252.fixed_hold_ce(model_init, hold_batches, char_table, pad_id, device)
    log(f"  hold CE base (init upper)={base_hold:.4f}")

    d_hidden = 2 * (model.head.in_features // 2)
    glue = SlotBias(d_hidden, device)
    opt_glue = torch.optim.AdamW(glue.trainable(), lr=args.lr_glue, weight_decay=0.01)
    opt_upper = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr_upper, weight_decay=0.01,
    )

    used: set[str] = set()
    pack = None
    n_tapes = 0
    curve = []
    last_train_em = float("nan")

    def run_locked(mdl, gl, tp, facts_):
        return s265.exam(
            gl, mdl, char_table, tok, bank_can, tp, facts_, pad_id, V, device, k, max_new,
            locked=True, open_thresh=args.open_thresh, reopen_margin=args.reopen_margin,
            max_opens=args.max_opens,
        )

    for step in range(1, steps + 1):
        if pack is None or (step - 1) % tape_period == 0:
            pack = build_tape(
                bank_can=bank_can, tok=tok, pad_id=pad_id, device=device, rng=rng,
                values_pool=values_pool, lines=lines, used=used,
                n_facts=n_facts, n_nonsense=n_nonsense, n_dist=n_dist,
            )
            n_tapes += 1
            log(
                f"  tape#{n_tapes} @step {step}: slots={pack['n_slots']} "
                f"fit={len(pack['fit_facts'])} eval={len(pack['eval_facts'])} "
                f"used_pool={len(used)}"
            )

        s213.set_train_mode(model, "upper")
        out = train_step(
            glue, model, char_table, tok, bank_can, pack, flat, off, train_docs,
            pad_id, V, device, k=k, gate_l1=args.gate_l1, nce_w=args.nce_w,
            nce_tau=args.nce_tau, rng=rng,
        )
        if out is None:
            continue
        opt_glue.zero_grad(set_to_none=True)
        opt_upper.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(glue.trainable()) + [p for p in model.parameters() if p.requires_grad], 1.0)
        opt_glue.step()
        opt_upper.step()

        if step % max(1, steps // 10) == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                tr = run_locked(model, glue, pack["tape"], pack["eval_facts"])
            last_train_em = float(tr["em"])
            curve.append({
                "step": step, "tape": n_tapes, "em_train_tape": last_train_em,
                "loss_fact": out["loss_fact"], "loss_prose": out["loss_prose"],
                "gate_fact": out["gate_fact"], "gate_prose": out["gate_prose"],
            })
            log(
                f"  step {step}/{steps} tape#{n_tapes} em_train={last_train_em:.3f} "
                f"fact={out['loss_fact']} prose={out['loss_prose']} ({time.time()-t0:.0f}s)"
            )
            s213.set_train_mode(model, "upper")

    glue.eval()
    model.eval()
    arc_hash1 = arc_enc_hash(model)
    g_arc = arc_hash0 == arc_hash1
    log(f"  arc_enc hash match={g_arc} ({arc_hash0[:12]}… vs {arc_hash1[:12]}…)")

    # ---- last train tape (procedure on familiar bank) ----
    em_train = run_locked(model, glue, pack["tape"], pack["eval_facts"])
    log(f"  last-train-tape EM={em_train['em']:.3f} verbatim={em_train['verbatim']:.3f}")

    # ---- novel tape: entities never used in any train rebuild ----
    pack_novel = build_tape(
        bank_can=bank_can, tok=tok, pad_id=pad_id, device=device, rng=random.Random(SEED + 99),
        values_pool=values_pool, lines=lines, used=used,
        n_facts=n_facts, n_nonsense=n_nonsense, n_dist=n_dist,
    )
    em_novel = run_locked(model, glue, pack_novel["tape"], pack_novel["eval_facts"])
    em_novel_frozen = run_locked(model_init, glue, pack_novel["tape"], pack_novel["eval_facts"])
    log(
        f"  novel-tape EM live={em_novel['em']:.3f} frozen_upper={em_novel_frozen['em']:.3f} "
        f"slots={pack_novel['n_slots']}"
    )

    # ---- sanitation on novel tape ----
    empty = run_locked(model, glue, pack_novel["tape"].emptied(), pack_novel["eval_facts"])
    shuf = run_locked(model, glue, pack_novel["tape"].shuffled(SEED + 1), pack_novel["eval_facts"])
    per_tgt, per_ret = [], []
    for f in pack_novel["eval_facts"]:
        td = pack_novel["tape"].copy()
        td.drop_value(f["value"])
        per_tgt.append(run_locked(model, glue, td, [f])["em"])
        others = [o for o in pack_novel["eval_facts"] if o is not f]
        if others:
            per_ret.append(run_locked(model, glue, td, others)["em"])
    em_tgt = float(np.mean(per_tgt)) if per_tgt else float("nan")
    em_ret = float(np.mean(per_ret)) if per_ret else float("nan")

    hold_after = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)
    log(f"  hold CE after={hold_after:.4f} (base={base_hold:.4f}) empty_em={empty['em']:.3f}")

    em265 = published_265_em()
    g_novel = (
        not math.isnan(em_novel["em"]) and not math.isnan(em_train["em"])
        and em_novel["em"] >= em_train["em"] - 0.05
    )
    g_beats = em_novel["em"] >= em_novel_frozen["em"] + 0.05
    g_leak = empty["em"] <= 0.10
    g_slot = (
        em_novel["em"] >= 0.4 and em_tgt <= 0.1
        and (math.isnan(em_ret) or em_ret >= 0.7 * em_novel["em"])
    )
    g_lang = hold_after <= base_hold + 0.05
    g_shuf = shuf["em"] <= max(0.10, em_novel["em"] - 0.40)

    if g_novel and g_arc and g_beats and g_leak and g_slot and g_lang:
        overall = "MIND_LEARNS_TAPE_OK"
    elif g_novel and g_arc and (g_beats or g_leak):
        overall = "MIND_LEARNS_TAPE_PARTIAL"
    else:
        overall = "MIND_LEARNS_TAPE_NO"

    torch.save(
        {
            "model": model.state_dict(), "glue": glue.state_dict(),
            "stage": 268, "steps": steps, "n_tapes": n_tapes,
            "arc_enc_hash": arc_hash1,
        },
        CKPT_OUT,
    )

    out = {
        "stage": 268,
        "overall": overall,
        "smoke": args.smoke,
        "seed": SEED,
        "trunk": trunk_ckpt.name,
        "fp_version": fp_version(),
        "steps": steps,
        "tape_period": tape_period,
        "n_tapes": n_tapes,
        "distractor_slots": n_dist,
        "lr_glue": args.lr_glue,
        "lr_upper": args.lr_upper,
        "gates": {
            "G_novel_tape": g_novel,
            "G_arc_enc_frozen": g_arc,
            "G_beats_frozen_mind": g_beats,
            "G_no_param_leak": g_leak,
            "G_slot_delete": g_slot,
            "G_lang_intact": g_lang,
            "G_tape_causal": g_shuf,
        },
        "headline": {
            "em_last_train_tape": em_train["em"],
            "em_novel_tape": em_novel["em"],
            "em_novel_frozen_upper": em_novel_frozen["em"],
            "delta_novel_minus_train": em_novel["em"] - em_train["em"],
            "delta_live_minus_frozen": em_novel["em"] - em_novel_frozen["em"],
            "em_265_published": em265,
        },
        "controls": {
            "em_empty_tape": empty["em"],
            "em_shuffled_tape": shuf["em"],
            "em_target_after_delete": em_tgt,
            "em_retained_after_delete": em_ret,
            "hold_ce_base": base_hold,
            "hold_ce_after": hold_after,
            "arc_enc_hash_before": arc_hash0,
            "arc_enc_hash_after": arc_hash1,
        },
        "train_tape": {
            "em": em_train["em"], "em_span": em_train["em_span"], "em_text": em_train["em_text"],
            "verbatim": em_train["verbatim"], "open_recall": em_train["open_recall"],
            "n_eval": len(pack["eval_facts"]), "n_slots": pack["n_slots"],
        },
        "novel_tape": {
            "em": em_novel["em"], "em_span": em_novel["em_span"], "em_text": em_novel["em_text"],
            "verbatim": em_novel["verbatim"], "open_recall": em_novel["open_recall"],
            "n_eval": len(pack_novel["eval_facts"]), "n_slots": pack_novel["n_slots"],
            "frozen_upper_em": em_novel_frozen["em"],
        },
        "curve": curve,
        "note": (
            "Upper trunk learns; arc_enc frozen (hash). Tape rebuilt every tape_period steps so "
            "no planted fact survives across rebuilds. G_novel_tape is the claim: procedure "
            "transfers to a bank never seen in training. G_beats_frozen_mind compares the same "
            "glue on novel tape with live vs init upper. Decode is 265 span-lock."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")

    mini = (
        f"# Stage 268 mind learns tape\n\n"
        f"**{overall}** · steps={steps} · tapes={n_tapes} · bank≈{n_dist}+facts"
        f"{' · SMOKE' if args.smoke else ''}\n\n"
        f"| exam | EM | frozen-upper EM |\n|------|---:|----------------:|\n"
        f"| last train tape | {em_train['em']:.3f} | — |\n"
        f"| novel tape | **{em_novel['em']:.3f}** | {em_novel_frozen['em']:.3f} |\n\n"
        f"## Gates (read G_novel_tape first)\n\n"
        f"- G_novel_tape: **{g_novel}** (novel {em_novel['em']:.3f} vs train {em_train['em']:.3f})\n"
        f"- G_arc_enc_frozen: **{g_arc}**\n"
        f"- G_beats_frozen_mind: **{g_beats}**\n"
        f"- G_no_param_leak: **{g_leak}** (empty={empty['em']:.3f})\n"
        f"- G_slot_delete: **{g_slot}**\n"
        f"- G_lang_intact: **{g_lang}** (hold {base_hold:.3f}→{hold_after:.3f})\n"
    )
    MINI.write_text(mini, encoding="utf-8")
    log(json.dumps({"overall": overall, "gates": out["gates"], "headline": out["headline"]}, indent=2))
    log(f"wrote {DECISION} wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
