"""
Stage 244 — Forget-A cleanliness: slot delete vs GPT gradient unlearn.

Subset of 205 framing: delete half the facts from TapeLM slots (O(1)); GPT gradient-ascent
unlearn with early stop. Compare target forget + retained collateral + next_tok.

  python _stage244_forget_clean.py [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime, timezone

import torch

import _stage24x_lib as L
from _stage196_tapelm import load_gpt

SEED = 244
DECISION = L.RES / "stage244_decision.json"
MINI = L.RES / "stage244_mini.md"
LOG = L.RES / "_stage244_log.txt"
CHANCE = 0.25


def recall_subset(gm_or_none, mode, facts_sub, all_values, tok, pad_id, device, seed, bank=None, K=None, V=None):
    if mode == "tape":
        return L.tape_recall(facts_sub, all_values, bank, K, V, seed)
    return L.gpt_fact_recall(gm_or_none, tok, pad_id, facts_sub, all_values, device, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    LOG.write_text("", encoding="utf-8")
    log = L.make_logger(LOG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = __import__("random").Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    n_facts = 16 if args.smoke else 40
    n_tgt = 6 if args.smoke else 16
    ft_steps = 200 if args.smoke else 1600
    unl_max = 30 if args.smoke else 60
    n_next = 40 if args.smoke else 120
    n_batch, ft_len, ft_lr = 8, 64, 3e-4
    unl_lr = 5e-5
    mem_target = 0.72

    log(f"Stage244 start {datetime.now(timezone.utc).isoformat()}")
    _, _, _, _, tok, _, pad_id, char_table, model, bank = L.load_p1(device)
    _, values_pool, _, paras = L.wiki_bits(args.smoke, 60 if args.smoke else 400, rng)
    facts, all_values = L.make_facts(n_facts, values_pool, rng)
    target = facts[:n_tgt]
    retained = facts[n_tgt:]
    items = L.load_next_tok_items(n_next)

    K, vals = L.write_tape_bank(bank, facts)
    # map value -> indices for delete
    tape_tgt0 = L.tape_recall(target, all_values, bank, K, vals, SEED)
    tape_ret0 = L.tape_recall(retained, all_values, bank, K, vals, SEED)
    tape_nt0 = L.curve_next_tok(model, char_table, pad_id, items, device)

    # delete target slots
    keep = [i for i, f in enumerate(facts) if f not in target]
    # facts are dicts — use fid
    tgt_fids = {f["fid"] for f in target}
    keep_idx = [i for i, f in enumerate(facts) if f["fid"] not in tgt_fids]
    K_del = K[keep_idx]
    V_del = [vals[i] for i in keep_idx]
    tape_tgt1 = L.tape_recall(target, all_values, bank, K_del, V_del, SEED)
    tape_ret1 = L.tape_recall(retained, all_values, bank, K_del, V_del, SEED)
    tape_nt1 = L.curve_next_tok(model, char_table, pad_id, items, device)
    log(f"tape delete: tgt {tape_tgt0:.3f}->{tape_tgt1:.3f} ret {tape_ret0:.3f}->{tape_ret1:.3f}")

    gm = copy.deepcopy(load_gpt(device))
    used, fact_ids, _ = L.memorize_gpt(
        gm, tok, pad_id, facts, all_values, paras, device, SEED, ft_steps, n_batch, ft_len, ft_lr, mem_target,
        40 if args.smoke else 100, log,
    )
    gpt_tgt0 = L.gpt_fact_recall(gm, tok, pad_id, target, all_values, device, SEED)
    gpt_ret0 = L.gpt_fact_recall(gm, tok, pad_id, retained, all_values, device, SEED)
    gpt_nt0 = L.gpt_next_tok(gm, items, device)

    tgt_ids = [[i for i in tok.encode(f["sent"]).ids if i != pad_id] for f in target]
    uopt = torch.optim.AdamW(gm.parameters(), lr=unl_lr)
    urng = __import__("random").Random(SEED + 13)
    gm.train()
    used_u = 0
    for step in range(1, unl_max + 1):
        x = L.ft_batch(urng, tgt_ids, [], n_batch, ft_len, device, mix_real=False)
        loss = -gm(input_ids=x, labels=x).loss
        uopt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gm.parameters(), 1.0)
        uopt.step()
        used_u = step
        if step % 10 == 0:
            gm.eval()
            cur = L.gpt_fact_recall(gm, tok, pad_id, target, all_values, device, SEED)
            log(f"  gpt unlearn {step}: tgt={cur:.3f}")
            if cur <= CHANCE + 0.05:
                break
            gm.train()
    gm.eval()
    gpt_tgt1 = L.gpt_fact_recall(gm, tok, pad_id, target, all_values, device, SEED)
    gpt_ret1 = L.gpt_fact_recall(gm, tok, pad_id, retained, all_values, device, SEED)
    gpt_nt1 = L.gpt_next_tok(gm, items, device)
    log(f"gpt unlearn ({used_u}): tgt {gpt_tgt0:.3f}->{gpt_tgt1:.3f} ret {gpt_ret0:.3f}->{gpt_ret1:.3f}")

    g_forget = tape_tgt1 <= CHANCE + 0.05
    g_no_collat = abs(tape_ret1 - tape_ret0) <= 0.02 and abs(tape_nt1 - tape_nt0) < 1e-9
    g_gpt_collat = abs(gpt_ret1 - gpt_ret0) > 0.02 or abs(gpt_nt1 - gpt_nt0) > 0.02
    g_gpt_forget = gpt_tgt1 <= gpt_tgt0 - 0.15 or gpt_tgt1 <= CHANCE + 0.15
    if g_forget and g_no_collat and g_gpt_collat:
        overall = "FORGET_CLEAN_OK"
    elif g_forget and g_no_collat:
        overall = "FORGET_CLEAN_PARTIAL"
    else:
        overall = "FORGET_CLEAN_NO"

    out = {
        "stage": 244,
        "overall": overall,
        "gates": {
            "G_tape_forget_to_chance": g_forget,
            "G_tape_no_collateral": g_no_collat,
            "G_gpt_shows_collateral": g_gpt_collat,
            "G_gpt_forgot_some": g_gpt_forget,
        },
        "tape": {
            "tgt_before": tape_tgt0,
            "tgt_after": tape_tgt1,
            "ret_before": tape_ret0,
            "ret_after": tape_ret1,
            "next_tok_before": tape_nt0,
            "next_tok_after": tape_nt1,
        },
        "gpt": {
            "tgt_before": gpt_tgt0,
            "tgt_after": gpt_tgt1,
            "ret_before": gpt_ret0,
            "ret_after": gpt_ret1,
            "next_tok_before": gpt_nt0,
            "next_tok_after": gpt_nt1,
            "unlearn_steps": used_u,
            "memorize_steps": used,
        },
        "note": "Capability vs parametric GPT; architectural vs GPT+RAG index delete.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    L.dump(DECISION, MINI, out, "Stage 244 forget cleanliness")
    log(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
