"""
Stage 212 — instance / content-invariant channel on the frozen tape (final pre-publish stage).

Question: can a narrow read-only channel over the SAME frozen tape encode WHICH OCCURRENCE
of a surface form we are in (episode identity), and thereby resolve collisions that a
surface-keyed memory cannot? (old hop2/joint debt: dense H1 collisions, soft-disambig failed)

Setup (no CE touched, P1 frozen, channel read-only):
  h(crop)   = [fast_last ; slow_last] from frozen P1 over a text crop
  inst      = normalize(g(h)), g = 2-layer MLP trained CONTRASTIVELY:
                positive = the OTHER (disjoint) half of the same occurrence window
                hard neg = halves of OTHER occurrences of the SAME surface form

T1 collision (4-way, chance 0.25): one surface form S with 4 distinct occurrences, each
  carrying a distinct value label. Store key from first half, query from the DISJOINT second
  half. Candidates = the 4 sibling values → surface key alone is blind by construction.
  Baselines: fp_only (surface key), ctx_blend (197 M3 subject+ctx), soft_rerank (weak ctx),
             inst_random (untrained g — does learning matter?)
T2 para/hard invariance on 179 pairs (held out from corpus training).
T3/G5 next_tok unchanged + anti-CF assert.

  python _stage212_instance_channel.py
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, WORD_RE, FpBank

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
EXAM = Path("data/stage191_exam_v3.jsonl")
DECISION = RES / "stage212_decision.json"
MINI = RES / "stage212_mini.md"
LOG = RES / "_stage212_log.txt"

SEED = 212
CORPUS_CHARS = 150_000_000
MID_START = 70_000_000
MID_CHARS = 6_000_000
OCC_PER_SURFACE = 4
N_SURFACES = 320
WIN_CHARS = 220
STEPS = 3000
BATCH_SURF = 16
LR = 1e-3
D_INST = 128
TEMP = 0.07
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


class InstanceHead(nn.Module):
    """Narrow read-only channel: tape state -> instance code."""

    def __init__(self, d_in: int, d_out: int = D_INST):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in // 2), nn.GELU(), nn.Linear(d_in // 2, d_out))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)


@torch.no_grad()
def tape_states(model, char_table, tok, pad_id, texts: list[str], device, batch: int = 32) -> torch.Tensor:
    """[N, 2d] = [fast_last ; slow_last] over each text crop (frozen P1)."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        rows, lens = [], []
        for t in chunk:
            ids = [j for j in tok.encode(t).ids if j != pad_id][:MAX_ARCS]
            if not ids:
                ids = [pad_id]
            lens.append(len(ids))
            rows.append(ids + [pad_id] * (MAX_ARCS - len(ids)))
        x = torch.tensor(rows, dtype=torch.long, device=device)
        pad = x == pad_id
        arcs = model._arcs(char_table[x], x)
        fast = model.fast(arcs, pad_mask=pad)
        slow, _, _ = model.slow(arcs, pad)
        idx = torch.tensor([max(0, l - 1) for l in lens], device=device)
        ar = torch.arange(len(chunk), device=device)
        out.append(torch.cat([fast[ar, idx], slow[ar, idx]], dim=-1))
    return torch.cat(out, 0)


def collect_occurrences(paras: list[str], rng: random.Random):
    """surface -> list of OCC_PER_SURFACE windows, each split into two disjoint halves."""
    occ = defaultdict(list)
    for p in paras:
        for m in ENT_RE.finditer(p):
            S = m.group(1)
            if len(S) < 4:
                continue
            lo = max(0, m.start() - WIN_CHARS // 2)
            hi = min(len(p), m.end() + WIN_CHARS // 2)
            win = p[lo:hi]
            if len(WORD_RE.findall(win)) < 12:
                continue
            occ[S].append(win)
    items = []
    for S, wins in occ.items():
        uniq = list(dict.fromkeys(wins))
        if len(uniq) < OCC_PER_SURFACE:
            continue
        rng.shuffle(uniq)
        halves = []
        for w in uniq[:OCC_PER_SURFACE]:
            cut = len(w) // 2
            a, b = w[:cut].strip(), w[cut:].strip()
            if len(WORD_RE.findall(a)) < 4 or len(WORD_RE.findall(b)) < 4:
                halves = []
                break
            halves.append((a, b))
        if len(halves) == OCC_PER_SURFACE:
            items.append((S, halves))
        if len(items) >= N_SURFACES:
            break
    return items


def train_instance(head: InstanceHead, groups, device, rng):
    """InfoNCE: positive = disjoint other half of same occurrence; hard negs = same surface."""
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()
    running = None
    for step in range(1, STEPS + 1):
        picks = [groups[rng.randint(0, len(groups) - 1)] for _ in range(BATCH_SURF)]
        Ha, Hb = [], []
        for g in picks:
            for a_state, b_state in g:
                Ha.append(a_state)
                Hb.append(b_state)
        A = head(torch.stack(Ha))
        B = head(torch.stack(Hb))
        logits = A @ B.T / TEMP
        target = torch.arange(A.size(0), device=device)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log(f"  inst step {step} loss~{float(loss):.3f}")
        running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
    head.eval()
    return running


@torch.no_grad()
def eval_collision(test_items, states_a, states_b, bank, head, mode: str, rng, w_ctx: float = 1.0):
    """4-way among the 4 sibling values of the SAME surface (chance 0.25)."""
    ok = n = 0
    for si, (S, halves) in enumerate(test_items):
        sfp = bank.fp([S])[0]
        keys = []
        for oi in range(OCC_PER_SURFACE):
            if mode in ("inst", "inst_random"):
                keys.append(head(states_a[si][oi].unsqueeze(0))[0])
            elif mode == "ctx_blend":
                c = bank.ctx_fp(halves[oi][0])
                keys.append(F.normalize(sfp + c, dim=-1) if c is not None else sfp)
            elif mode == "soft_rerank":
                c = bank.ctx_fp(halves[oi][0])
                keys.append(F.normalize(sfp + 0.25 * c, dim=-1) if c is not None else sfp)
            else:  # fp_only
                keys.append(sfp)
        for oi in range(OCC_PER_SURFACE):
            if mode in ("inst", "inst_random"):
                q = head(states_b[si][oi].unsqueeze(0))[0]
            elif mode == "ctx_blend":
                c = bank.ctx_fp(halves[oi][1])
                q = F.normalize(sfp + c, dim=-1) if c is not None else sfp
            elif mode == "soft_rerank":
                c = bank.ctx_fp(halves[oi][1])
                q = F.normalize(sfp + 0.25 * c, dim=-1) if c is not None else sfp
            else:
                q = sfp
            scores = [float(k @ q) for k in keys]
            order = list(range(OCC_PER_SURFACE))
            rng.shuffle(order)
            shuffled = [scores[i] for i in order]
            gold = order.index(oi)
            ok += int(int(np.argmax(shuffled)) == gold)
            n += 1
    return ok / max(1, n)


@torch.no_grad()
def eval_para_hard(model, char_table, tok, pad_id, head, device):
    def code(sent):
        h = tape_states(model, char_table, tok, pad_id, [sent], device)
        return head(h)[0]

    para = [float(code(a) @ code(b)) for a, b in s179.PARAPHRASE_PAIRS]
    hard = [float(code(a) @ code(b)) for a, b in s179.HARD_PAIRS]
    mp, mh = float(np.mean(para)), float(np.mean(hard))
    return {"para": mp, "hard": mh, "gap_hard_minus_para": mh - mp, "inversion_para_gt_hard": mp > mh}


@torch.no_grad()
def next_tok_acc(model, char_table, pad_id, device, n=100):
    if not EXAM.exists():
        return None
    items = []
    with EXAM.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if it.get("type") == "next_tok":
                items.append(it)
            if len(items) >= n:
                break
    if not items:
        return None
    ok = 0
    for it in items:
        sc = [span_logprob_x(model, char_table, pad_id, it["ctx_ids"], cd, device) for cd in it["cand_ids"]]
        ok += int(np.argmax(sc) == it["gold_idx"])
    return ok / len(items)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage212 start {datetime.now(timezone.utc).isoformat()}")
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
    sig_before = sum(float(p.abs().sum()) for p in model.parameters())
    bank = FpBank(model, stoi, device)
    d2 = model.head.in_features  # fast+slow
    log(f"P1 frozen, tape state dim={d2} ({time.time()-t0:.0f}s)")

    nt_before = next_tok_acc(model, char_table, pad_id, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    mid = text[MID_START : MID_START + MID_CHARS]
    del text
    paras = [p.strip() for p in mid.split("\n") if 200 < len(p.strip()) < 1200]
    rng.shuffle(paras)
    items = collect_occurrences(paras, rng)
    log(f"surfaces={len(items)} x{OCC_PER_SURFACE} occurrences ({time.time()-t0:.0f}s)")
    if len(items) < 40:
        log("[212] ABORT not enough colliding surfaces")
        return 1

    all_a = [h[0] for _, halves in items for h in halves]
    all_b = [h[1] for _, halves in items for h in halves]
    Sa = tape_states(model, char_table, tok, pad_id, all_a, device)
    Sb = tape_states(model, char_table, tok, pad_id, all_b, device)
    log(f"tape states {tuple(Sa.shape)} ({time.time()-t0:.0f}s)")

    per_surface_a, per_surface_b = [], []
    for i in range(len(items)):
        sl = slice(i * OCC_PER_SURFACE, (i + 1) * OCC_PER_SURFACE)
        per_surface_a.append(Sa[sl])
        per_surface_b.append(Sb[sl])

    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(0.7 * len(idx))
    tr_idx, te_idx = idx[:cut], idx[cut:]
    groups_train = [[(per_surface_a[i][o], per_surface_b[i][o]) for o in range(OCC_PER_SURFACE)] for i in tr_idx]
    test_items = [items[i] for i in te_idx]
    test_a = [per_surface_a[i] for i in te_idx]
    test_b = [per_surface_b[i] for i in te_idx]
    log(f"train surfaces={len(tr_idx)} test surfaces={len(te_idx)}")

    head = InstanceHead(d2).to(device)
    loss = train_instance(head, groups_train, device, random.Random(SEED))
    log(f"instance head trained loss~{loss:.3f}")

    head_rand = InstanceHead(d2).to(device)
    head_rand.eval()

    ev = lambda mode, h, seed: eval_collision(test_items, test_a, test_b, bank, h, mode, random.Random(seed))
    acc_inst = ev("inst", head, SEED + 1)
    acc_rand = ev("inst_random", head_rand, SEED + 1)
    acc_fp = ev("fp_only", head, SEED + 1)
    acc_blend = ev("ctx_blend", head, SEED + 1)
    acc_soft = ev("soft_rerank", head, SEED + 1)
    log(f"T1 collision: inst={acc_inst:.3f} inst_random={acc_rand:.3f} fp_only={acc_fp:.3f} ctx_blend={acc_blend:.3f} soft_rerank={acc_soft:.3f}")

    ph = eval_para_hard(model, char_table, tok, pad_id, head, device)
    log(f"T2 para={ph['para']:.3f} hard={ph['hard']:.3f} gap={ph['gap_hard_minus_para']:.3f} inversion={ph['inversion_para_gt_hard']}")

    nt_after = next_tok_acc(model, char_table, pad_id, device)
    sig_after = sum(float(p.abs().sum()) for p in model.parameters())
    anticf = abs(sig_before - sig_after) < 1e-3
    d_nt = None if (nt_before is None or nt_after is None) else abs(nt_before - nt_after)

    g1 = acc_inst >= 0.70 and acc_fp <= 0.45
    g2 = ph["inversion_para_gt_hard"] or ph["gap_hard_minus_para"] < 0
    g3 = acc_inst >= acc_rand + 0.10
    g4 = acc_inst >= acc_soft + 0.10
    g5 = anticf and (d_nt is None or d_nt <= 0.005)
    g6 = acc_inst >= acc_blend + 0.05

    if g1 and g3 and g4 and g5 and g6:
        overall = "THESIS_YES"
    elif g1 and g5:
        overall = "ENGINEERING_ONLY"
    else:
        overall = "THESIS_NO"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "instance_channel_212",
        "overall": overall,
        "t1_collision_4way": {
            "instance_learned": acc_inst,
            "instance_random_untrained": acc_rand,
            "fp_only_surface_key": acc_fp,
            "ctx_blend_197": acc_blend,
            "soft_rerank": acc_soft,
            "chance": CHANCE,
        },
        "t2_para_hard": ph,
        "next_tok": {"before": nt_before, "after": nt_after, "delta": d_nt},
        "gates": {
            "g1_collision": g1,
            "g2_invariance": g2,
            "g3_learning_matters": g3,
            "g4_beats_soft_rerank": g4,
            "g5_no_ce_cost": g5,
            "g6_beats_ctx_blend": g6,
        },
        "anticf_frozen": anticf,
        "surfaces": {"total": len(items), "train": len(tr_idx), "test": len(te_idx), "occ_per_surface": OCC_PER_SURFACE},
        "note": "instance channel = read-only 2-layer head on frozen tape state; collision test is 4-way among "
        "siblings of the SAME surface form, store/query crops are DISJOINT halves (no lexical overlap shortcut)",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage212 — instance / content-invariant channel",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- T1 collision (4-way, chance 0.25): instance **{acc_inst:.3f}** | untrained {acc_rand:.3f} | "
                f"fp_only {acc_fp:.3f} | ctx_blend {acc_blend:.3f} | soft_rerank {acc_soft:.3f}",
                f"- T2 para={ph['para']:.3f} hard={ph['hard']:.3f} inversion={ph['inversion_para_gt_hard']}",
                f"- next_tok {nt_before} -> {nt_after} (delta {d_nt}), anti-CF {anticf}",
                f"- gates: {out['gates']}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[212] {overall} | inst={acc_inst:.3f} blend={acc_blend:.3f} fp={acc_fp:.3f} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
