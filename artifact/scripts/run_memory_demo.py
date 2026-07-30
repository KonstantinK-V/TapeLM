#!/usr/bin/env python3
"""
TapeLM memory demo — product path (no stage numbers in output).

Shows one encoder, canonical slots, optional family W, 4-way retrieve,
fp decode, and contradiction resolution.

  python artifact/scripts/run_memory_demo.py
  python artifact/scripts/run_memory_demo.py --smoke
  python artifact/scripts/run_memory_demo.py --skip-cross-domain
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage225_family_fork as s225
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import (
    DomainAdapter,
    W_REGISTRY_DIR,
    fp_decode_pick_retrieved_4way,
    load_w_registry,
    resolve_slot_contradiction,
    slot_retrieve_4way,
    subject_slot_hits,
)

CKPT = REPO / "checkpoints/stage191_p1_curve.pt"
WIKI = REPO / "data/_wikitext103_train.txt"
SEED = 88001


def banner(title: str) -> None:
    print(f"\n{'-' * 60}\n{title}\n{'-' * 60}")


def load_p1(device: torch.device):
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)
    return model, bank, tok, pad_id, char_table, stoi, device


def wiki_entities(n: int, rng: random.Random) -> list[str]:
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        words = list(
            dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(2_000_000)) if len(m.group(1)) >= 5)
        )
    return words[:n]


def demo_canonical_recall(bank: FpBank, rng: random.Random, n_facts: int) -> tuple[torch.Tensor, list[str], list[str], list[str]]:
    banner("1 - Canonical memory (write -> 4-way recall)")
    words = wiki_entities(n_facts + 8, rng)
    subs = gen_fakes(set(words), rng, n_facts + 4)[:n_facts]
    vals = words[:n_facts]
    K, V = s221.build_fact_bank(bank, subs, vals, rng)
    print(f"  Wrote {len(V)} episodic slots (canonical fp keys).")
    ok = 0
    for S, gold in zip(subs, vals):
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        ctx = f"In the report {S} was linked to the organization."
        q = bank.ctx_fp(ctx, exclude=gold)
        if q is None:
            continue
        hit = slot_retrieve_4way(K, V, q, cands)
        ok += int(hit == gold)
        print(f"  Query subject {S!r} -> retrieved {hit!r}  ({'OK' if hit == gold else 'miss'})")
    rate = ok / max(1, n_facts)
    print(f"  4-way recall: {ok}/{n_facts} ({rate:.0%})")
    return K, V, subs, vals


def demo_contradiction(bank: FpBank, rng: random.Random) -> None:
    banner("2 - Contradictions (multi-hit -> resolution policy)")
    words = wiki_entities(20, rng)
    S = gen_fakes(set(words), rng, 1)[0]
    a, b = words[0], words[1]
    ctx_a = f"Official records state {S} was director of {a} in 1987 ."
    ctx_b = f"Later revision claims {S} was director of {b} in 1999 ."
    ka = bank.fp([S])[0]
    ca, cb = bank.ctx_fp(ctx_a, exclude=a), bank.ctx_fp(ctx_b, exclude=b)
    if ca is None or cb is None:
        print("  (skip: ctx_fp unavailable for sample)")
        return
    keys = torch.stack([F.normalize(ka + ca, dim=-1), F.normalize(ka + cb, dim=-1)])
    vals = [a, b]
    meta = [{"provenance": "official", "year": 1987}, {"provenance": "revision", "year": 1999}]
    idxs = [0, 1]

    for label, tmpl, gold in (
        ("neutral", f"In the report {S} was linked to the organization.", b),
        ("1987 cue", f"Per the 1987 official records, {S} was linked to the organization.", a),
        ("1999 cue", f"Per the 1999 revision, {S} was linked to the organization.", b),
    ):
        q = bank.ctx_fp(tmpl, exclude=None)
        if q is None:
            continue
        hits = subject_slot_hits(keys, vals, q, idxs, meta)
        pick_argmax = resolve_slot_contradiction(hits, tmpl, policy="argmax")
        pick = resolve_slot_contradiction(hits, tmpl, policy="composite")
        print(f"  [{label}] composite->{pick!r}  argmax->{pick_argmax!r}  gold={gold!r}  {'OK' if pick == gold else '-'}")


def demo_fp_decode_cross_domain(
    model0,
    bank_can: FpBank,
    K_can,
    V,
    subs,
    vals,
    tok,
    pad_id,
    char_table,
    stoi,
    device,
    rng: random.Random,
    smoke: bool,
) -> None:
    banner("3 - Cross-domain read (prose slots, code query + fp decode)")
    print("  Note: simulates encoder drift for query-side fp (stage-221 protocol).")
    print("  Canonical slot keys stay in frozen P1 geometry; W_bwd qmap + fp decode at read.")
    W_bwd = None
    try:
        adapters, _ = load_w_registry(REPO / W_REGISTRY_DIR, device)
        W_bwd = adapters.get("code_bwd")
        print("  Loaded family W from checkpoints/w_registry/ (qmap: code -> canonical).")
    except FileNotFoundError:
        print("  No w_registry - training smoke W (~1 min GPU)...")
        print("  Persist for reuse: python artifact/scripts/export_w_registry.py --smoke")
        with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read(500_000)
        core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:80]
        F0 = s221.fp_matrix(bank_can, core)
        text_code = s225.ensure_code(rng, smoke)
        flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=400, min_line_len=20)
        model_c = s221.finetune_arc_enc(model0, flat_c, off_c, char_table, pad_id, device, 80, SEED + 1)
        bank_c_shift = FpBank(model_c, stoi, device)
        F_c = s221.fp_matrix(bank_c_shift, core)
        W_bwd, align = s221.train_remap(DomainAdapter(256).to(device), F_c, F0, rng, 100, device)
        print(f"  Smoke W_bwd align={align:.3f}")

    if W_bwd is None:
        print("  Cross-domain demo skipped.")
        return

    text_code = s225.ensure_code(rng, True)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=400, min_line_len=20)
    model_c = s221.finetune_arc_enc(model0, flat_c, off_c, char_table, pad_id, device, 80, SEED + 2)
    bank_c = FpBank(model_c, stoi, device)

    ok_fp = n = 0
    for S, gold in zip(subs[: min(4, len(subs))], vals):
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        ctx = f"In the report {S} was linked to the organization."
        try:
            _, pick = fp_decode_pick_retrieved_4way(bank_can, K_can, V, W_bwd, bank_c, ctx, gold, cands)
        except ValueError:
            continue
        ok_fp += int(pick == gold)
        n += 1
        print(f"  Code-domain query {S!r} -> fp decode picks {pick!r}  ({'OK' if pick == gold else 'miss'})")
    if n:
        print(f"  fp decode accuracy on sample: {ok_fp}/{n} ({ok_fp / n:.0%})")


def main() -> int:
    ap = argparse.ArgumentParser(description="TapeLM memory product demo")
    ap.add_argument("--smoke", action="store_true", help="Fast run (default)")
    ap.add_argument("--full", action="store_true", help="More facts in step 1")
    ap.add_argument("--skip-cross-domain", action="store_true")
    args = ap.parse_args()
    n_facts = 8 if args.full else 4

    if not CKPT.is_file():
        print("Missing checkpoints/stage191_p1_curve.pt")
        print("  python artifact/scripts/download_checkpoints.py")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    print("TapeLM - memory demo (one encoder, canonical slots, W, decode, resolve)")
    print(f"Device: {device}")

    model, bank, tok, pad_id, char_table, stoi, device = load_p1(device)
    K, V, subs, vals = demo_canonical_recall(bank, rng, n_facts)
    demo_contradiction(bank, rng)
    if not args.skip_cross_domain:
        demo_fp_decode_cross_domain(model, bank, K, V, subs, vals, tok, pad_id, char_table, stoi, device, rng, args.smoke)

    banner("Done")
    print("  Docs: docs/MEMORY_ENGINEERING.md  artifact/OVERVIEW.md")
    print("  Full scorecard: python artifact/scripts/run_demo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
