"""
Stage 210 — structured hop composition INSIDE forward, answer via CE tokens.

Extends 203: SoftFollow / HopReader retrieve over frozen fp tape, but inject retrieved
vector as one memory-arc into P1's arc stream and read answer with span_logprob (tokens),
not cosine in fp-space.

Trainable (P1 frozen): reader (SoftFollow or HopReader), inject_proj (zero-init), log_gate.
Anti-CF: encoder untouched; tape K/V non-gradient.

Gates (pre_publish_frontier.md):
  G1 soft_follow_token test k2>=0.70 k3>=0.60
  G2 free_form overfits train>=0.90 test<=0.45
  G3 next_tok delta vs P1 <= 0.01 with inject path (mem_arc=0 on generic text)
  G4 gate=0 bit-identical logits vs P1
  G5 no_memory (gate=0 on hop task) <= 0.35

  python _stage210_softfollow_forward.py
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage203_internal_hops as s203
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_P2 = Path("checkpoints/stage191_p2_gpt.pt")
WIKI = Path("data/_wikitext103_train.txt")
EXAM = Path("data/stage191_exam_v3.jsonl")
DECISION = RES / "stage210_decision.json"
MINI = RES / "stage210_mini.md"
LOG = RES / "_stage210_log.txt"

SEED = 210
CORPUS_CHARS = 40_000_000
MID_START = 90_000_000
N_CHAIN = 240
CHAIN_LEN = 4
N_DISTRACT = 4000
T_STEPS = 3
STEPS_READER = 1500
STEPS_INJECT = 800
BATCH_READER = 128
BATCH_INJECT = 8
LR = 1e-3
QUERY_TMPL = "Chain start {a0} hops {k} answer "


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class InjectPack(nn.Module):
    """Reader + zero-init inject into arc stream; gate=0 => zero mem arc."""

    def __init__(self, reader: nn.Module, d: int):
        super().__init__()
        self.reader = reader
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.proj.weight)
        self.proj.weight.data.mul_(0.02)
        self.log_gate = nn.Parameter(torch.tensor(-2.0))

    def mem_arc(self, q_fp, k, K, Vv, force_zero: bool = False):
        if force_zero:
            z = torch.zeros_like(q_fp)
            return z, 0.0
        retrieved = self.reader(q_fp, k, K, Vv)
        g = torch.sigmoid(self.log_gate)
        return g * self.proj(retrieved), g


def forward_inject_logits(
    model,
    char_table,
    bpe_ids: torch.Tensor,
    pad_id: int,
    mem_arc: torch.Tensor,
    inject_pos: int,
    gate: float,
    always_insert: bool = False,
):
    """bpe_ids [1,L], mem_arc [1,d]; insert mem at inject_pos when gate>0 or always_insert."""
    pad = bpe_ids == pad_id
    if not always_insert and gate <= 1e-6:
        return model.forward_all(char_table[bpe_ids], pad, ids=bpe_ids)[0]
    arcs = model._arcs(char_table[bpe_ids], bpe_ids)
    ins = mem_arc.unsqueeze(1)
    arcs = torch.cat([arcs[:, :inject_pos], ins, arcs[:, inject_pos:]], dim=1)
    pad = torch.cat(
        [
            pad[:, :inject_pos],
            torch.zeros(1, 1, dtype=torch.bool, device=pad.device),
            pad[:, inject_pos:],
        ],
        dim=1,
    )
    fast = model.fast(arcs, pad_mask=pad)
    slow, _, _ = model.slow(arcs, pad)
    return model.head(torch.cat([fast, slow], dim=-1))


def encode_query(tok: Tokenizer, pad_id: int, a0: str, k: int) -> list[int]:
    text = QUERY_TMPL.format(a0=a0, k=k)
    return [i for i in tok.encode(text).ids if i != pad_id]


def encode_word(tok: Tokenizer, pad_id: int, w: str) -> list[int]:
    return [i for i in tok.encode(" " + w).ids if i != pad_id]


def pack_ctx_cand(ctx_ids: list[int], cand_ids: list[int]) -> tuple[list[int], int]:
    """Fit in MAX_ARCS-1 tokens (+1 inject arc). Return seq, inject_pos."""
    seq = (ctx_ids + cand_ids)[-(MAX_ARCS - 1) :]
    n_cand = min(len(cand_ids), len(seq))
    inject_pos = len(seq) - n_cand
    return seq, inject_pos


def span_logprob_inject(
    model,
    char_table,
    pad_id: int,
    ctx_ids: list[int],
    cand_ids: list[int],
    mem_arc: torch.Tensor,
    gate: float,
    device,
    always_insert: bool = False,
) -> float:
    seq, inject_pos = pack_ctx_cand(ctx_ids, cand_ids)
    n_cand = len(seq) - inject_pos
    x = torch.tensor([seq], dtype=torch.long, device=device)
    gval = float(gate) if not isinstance(gate, torch.Tensor) else float(gate.item())
    logits = forward_inject_logits(
        model, char_table, x, pad_id, mem_arc, inject_pos, gval, always_insert=always_insert
    )[0]
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for j in range(n_cand):
        tid = seq[inject_pos + j]
        li = inject_pos + j if always_insert or gval > 1e-6 else inject_pos + j - 1
        if li < 0:
            continue
        total += float(logp[li, tid])
    return total / max(1, n_cand)


def span_logprob_inject_train(
    model,
    char_table,
    pad_id: int,
    ctx_ids: list[int],
    cand_ids: list[int],
    mem_arc: torch.Tensor,
    gate: float,
    device,
    always_insert: bool = True,
) -> torch.Tensor:
    seq, inject_pos = pack_ctx_cand(ctx_ids, cand_ids)
    n_cand = len(seq) - inject_pos
    x = torch.tensor([seq], dtype=torch.long, device=device)
    gval = float(gate) if not isinstance(gate, torch.Tensor) else float(gate.detach())
    logits = forward_inject_logits(
        model, char_table, x, pad_id, mem_arc, inject_pos, gval, always_insert=always_insert
    )[0]
    logp = F.log_softmax(logits, dim=-1)
    loss = torch.tensor(0.0, device=device)
    for j in range(n_cand):
        tid = seq[inject_pos + j]
        li = inject_pos + j if always_insert or gval > 1e-6 else inject_pos + j - 1
        if li >= 0:
            loss = loss - logp[li, tid]
    return loss / max(1, n_cand)


def train_reader_fp(pack: InjectPack, train_s, bank, K, Vv, device, rng):
    """Phase A: same fp CE as 203 (cheap — no P1 forward in loop)."""
    opt = torch.optim.AdamW(pack.reader.parameters(), lr=LR, weight_decay=0.01)
    pack.reader.train()
    running = None
    for step in range(1, STEPS_READER + 1):
        batch = [train_s[rng.randint(0, len(train_s) - 1)] for _ in range(BATCH_READER)]
        q = bank.fp([b[0] for b in batch])
        kk = torch.tensor([b[1] for b in batch], device=device)
        tgt = bank.fp([b[2] for b in batch])
        ans = pack.reader(q, kk, K, Vv)
        loss = F.cross_entropy(ans @ tgt.T / 0.1, torch.arange(len(batch), device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log(f"  reader step {step} loss~{float(loss):.3f}")
        running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
    pack.reader.eval()
    return running


def train_inject_only(pack: InjectPack, train_s, bank, K, Vv, tok, pad_id, model, char_table, device, rng):
    """Phase B: frozen reader; train inject_proj + log_gate through P1 (small batch)."""
    for p in pack.reader.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([pack.proj.weight, pack.log_gate], lr=LR, weight_decay=0.01)
    pack.eval()
    running = None
    for step in range(1, STEPS_INJECT + 1):
        batch = [train_s[rng.randint(0, len(train_s) - 1)] for _ in range(BATCH_INJECT)]
        loss = torch.tensor(0.0, device=device)
        for a0, k, ak in batch:
            q = bank.fp([a0]).detach()
            kk = torch.tensor([k], device=device)
            mem, g = pack.mem_arc(q, kk, K, Vv)
            ctx = encode_query(tok, pad_id, a0, k)
            cand = encode_word(tok, pad_id, ak)
            loss = loss + span_logprob_inject_train(model, char_table, pad_id, ctx, cand, mem, g, device, True)
        loss = loss / BATCH_INJECT
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            log(f"  inject step {step} loss~{float(loss):.3f} gate={float(torch.sigmoid(pack.log_gate)):.3f}")
        running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
    return running


def train_pack(pack: InjectPack, train_s, bank, K, Vv, tok, pad_id, model, char_table, device, rng):
    lr = train_reader_fp(pack, train_s, bank, K, Vv, device, rng)
    li = train_inject_only(pack, train_s, bank, K, Vv, tok, pad_id, model, char_table, device, rng)
    return lr, li


@torch.no_grad()
def eval_token_pack(pack, sset, bank, K, Vv, tok, pad_id, model, char_table, device, rng, ent_pool, gate_zero: bool):
    accs = {}
    for k in range(1, CHAIN_LEN):
        items = [s for s in sset if s[1] == k]
        ok = 0
        for a0, _, ak in items:
            q = bank.fp([a0])
            kk = torch.tensor([k], device=device)
            mem, g = pack.mem_arc(q, kk, K, Vv, force_zero=gate_zero)
            ctx = encode_query(tok, pad_id, a0, k)
            cands = [ak] + [ent_pool[rng.randint(0, len(ent_pool) - 1)] for _ in range(3)]
            order = list(range(4))
            rng.shuffle(order)
            shuf = [cands[i] for i in order]
            gold = order.index(0)
            scores = [
                span_logprob_inject(
                    model,
                    char_table,
                    pad_id,
                    ctx,
                    encode_word(tok, pad_id, c),
                    mem,
                    g,
                    device,
                    always_insert=not gate_zero,
                )
                for c in shuf
            ]
            ok += int(int(np.argmax(scores)) == gold)
        accs[k] = ok / max(1, len(items))
    return accs


@torch.no_grad()
def hand_loop_cosine(sset, bank, K, Vv, rng, ent_pool, pool_fp):
    accs = {}
    for k in range(1, CHAIN_LEN):
        items = [s for s in sset if s[1] == k]
        ok = 0
        for a0, _, ak in items:
            v = bank.fp([a0])[0]
            for _ in range(k):
                v = Vv[int((K @ v).argmax())]
            pred = ent_pool[int((pool_fp @ v).argmax())]
            ok += int(pred == ak)
        accs[k] = ok / max(1, len(items))
    return accs


@torch.no_grad()
def bit_identity_check(model, char_table, pad_id, device, d):
    rng = random.Random(0)
    tok_ids = list(range(100, 200))
    rng.shuffle(tok_ids)
    seq = tok_ids[:32]
    x = torch.tensor([seq], dtype=torch.long, device=device)
    pad = x == pad_id
    base = model.forward_all(char_table[x], pad, ids=x)[0]
    mem0 = torch.zeros(1, d, device=device)
    inj = forward_inject_logits(model, char_table, x, pad_id, mem0, inject_pos=16, gate=0.0)
    diff = (base - inj).abs().max().item()
    return diff < 1e-5, diff


@torch.no_grad()
def next_tok_slice(model, char_table, pad_id, device, n=80):
    if not EXAM.exists():
        return None, None
    items = []
    with EXAM.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    items = [it for it in items if it.get("type") == "next_tok"][:n]
    if not items:
        return None, None

    ok_b, ok_i, n = 0, 0, 0
    d = model.head.in_features // 2
    mem0 = torch.zeros(1, d, device=device)
    for it in items:
        cands = it["cand_ids"]
        sb = [span_logprob_x(model, char_table, pad_id, it["ctx_ids"], cd, device) for cd in cands]
        si = [
            span_logprob_inject(model, char_table, pad_id, it["ctx_ids"], cd, mem0, 0.0, device) for cd in cands
        ]
        ok_b += int(np.argmax(sb) == it["gold_idx"])
        ok_i += int(np.argmax(si) == it["gold_idx"])
        n += 1
    return ok_b / n, ok_i / n


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage210 start {datetime.now(timezone.utc).isoformat()}")
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
    d = model.head.in_features // 2
    enc_sig = sum(float(p.abs().sum()) for p in model.arc_enc.parameters())

    bank = FpBank(model, stoi, device)
    log(f"P1 frozen d={d} ({time.time()-t0:.0f}s)")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)[MID_START % CORPUS_CHARS :]
    real_ents = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 4))[:3000]
    del text

    fakes = gen_fakes(set(), rng, N_CHAIN * CHAIN_LEN + 50)
    chains = [fakes[i * CHAIN_LEN : (i + 1) * CHAIN_LEN] for i in range(N_CHAIN)]
    chains = [c for c in chains if len(c) == CHAIN_LEN]
    rng.shuffle(chains)
    train_ch = chains[: int(0.75 * len(chains))]
    test_ch = chains[int(0.75 * len(chains)) :]

    key_fp, val_fp = [], []
    for c in chains:
        for i in range(len(c) - 1):
            key_fp.append(bank.fp([c[i]])[0])
            val_fp.append(bank.fp([c[i + 1]])[0])
    for i in range(min(N_DISTRACT, len(real_ents) - 1)):
        key_fp.append(bank.fp([real_ents[i]])[0])
        val_fp.append(bank.fp([real_ents[i + 1]])[0])
    K = torch.stack(key_fp, 0)
    Vv = torch.stack(val_fp, 0)

    def samples(chs):
        out = []
        for c in chs:
            for k in range(1, CHAIN_LEN):
                out.append((c[0], k, c[k]))
        return out

    train_s = samples(train_ch)
    test_s = samples(test_ch)
    ent_pool = list(dict.fromkeys([n for c in chains for n in c]))
    pool_fp = bank.fp(ent_pool)
    log(f"tape={K.shape[0]} train/test chains={len(train_ch)}/{len(test_ch)} ({time.time()-t0:.0f}s)")

    pack_soft = InjectPack(s203.SoftFollow(d, T_STEPS).to(device), d).to(device)
    ls_r, ls_i = train_pack(pack_soft, train_s, bank, K, Vv, tok, pad_id, model, char_table, device, random.Random(SEED))
    log(f"soft_follow reader~{ls_r:.3f} inject~{ls_i:.3f} gate={float(torch.sigmoid(pack_soft.log_gate)):.3f}")

    pack_free = InjectPack(s203.HopReader(d, T_STEPS, CHAIN_LEN).to(device), d).to(device)
    lf_r, lf_i = train_pack(pack_free, train_s, bank, K, Vv, tok, pad_id, model, char_table, device, random.Random(SEED + 1))
    log(f"free_form reader~{lf_r:.3f} inject~{lf_i:.3f}")

    ev = random.Random(SEED + 2)
    soft_tr = eval_token_pack(pack_soft, train_s, bank, K, Vv, tok, pad_id, model, char_table, device, ev, ent_pool, False)
    soft_te = eval_token_pack(pack_soft, test_s, bank, K, Vv, tok, pad_id, model, char_table, device, random.Random(SEED + 3), ent_pool, False)
    free_tr = eval_token_pack(pack_free, train_s, bank, K, Vv, tok, pad_id, model, char_table, device, random.Random(SEED + 4), ent_pool, False)
    free_te = eval_token_pack(pack_free, test_s, bank, K, Vv, tok, pad_id, model, char_table, device, random.Random(SEED + 5), ent_pool, False)
    no_mem = eval_token_pack(pack_soft, test_s, bank, K, Vv, tok, pad_id, model, char_table, device, random.Random(SEED + 6), ent_pool, gate_zero=True)
    ext = hand_loop_cosine(test_s, bank, K, Vv, random.Random(SEED + 7), ent_pool, pool_fp)

    enc_sig2 = sum(float(p.abs().sum()) for p in model.arc_enc.parameters())
    anticf = abs(enc_sig - enc_sig2) < 1e-3
    bit_ok, bit_diff = bit_identity_check(model, char_table, pad_id, device, d)
    acc_b, acc_i = next_tok_slice(model, char_table, pad_id, device)
    delta_nt = abs(acc_b - acc_i) if acc_b is not None else None

    g1 = soft_te.get(2, 0) >= 0.70 and soft_te.get(3, 0) >= 0.60
    g2 = min(free_tr.values()) >= 0.90 and max(free_te.values()) <= 0.45
    g3 = delta_nt is None or delta_nt <= 0.01
    g4 = bit_ok and anticf
    g5 = max(no_mem.values()) <= 0.35

    if g1 and g2 and g3 and g4 and g5:
        overall = "THESIS_YES"
    elif g1 and g4 and g5:
        overall = "ENGINEERING_ONLY"
    else:
        overall = "THESIS_NO"

    log(f"soft token test k2={soft_te.get(2):.3f} k3={soft_te.get(3):.3f}")
    log(f"free  token test k2={free_te.get(2):.3f} k3={free_te.get(3):.3f} train k2={free_tr.get(2):.3f}")
    log(f"no_mem test max={max(no_mem.values()):.3f} external cosine={ext}")
    log(f"next_tok base={acc_b} inject_mem0={acc_i} delta={delta_nt} bit_diff={bit_diff}")

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "softfollow_forward_tokens_210",
        "overall": overall,
        "soft_follow_token": {"train": soft_tr, "test": soft_te},
        "free_form_token": {"train": free_tr, "test": free_te},
        "no_memory_test": no_mem,
        "external_loop_cosine_test": ext,
        "next_tok": {"p1_acc": acc_b, "inject_mem0_acc": acc_i, "delta": delta_nt},
        "bit_identity_max_diff": bit_diff,
        "gates": {
            "g1_generalize": g1,
            "g2_structure": g2,
            "g3_no_ce_cost": g3,
            "g4_bit_identity": g4,
            "g5_needs_memory": g5,
        },
        "anticf_encoder_frozen": anticf,
        "chance": 0.25,
        "interpretation": "THESIS_YES = structured hops inside forward answer via CE tokens without P1 grad",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage210 — SoftFollow in forward, token answers",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- soft-follow token test: k2={soft_te.get(2):.3f} k3={soft_te.get(3):.3f}",
                f"- free-form token test: k2={free_te.get(2):.3f} k3={free_te.get(3):.3f} (train k2={free_tr.get(2):.3f})",
                f"- no_memory max={max(no_mem.values()):.3f} | external cosine {ext}",
                f"- next_tok delta={delta_nt} bit_diff={bit_diff}",
                f"- gates: {out['gates']}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[210] {overall} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
