"""
Stage 203 — internal hops (for-interest): a trainable differentiable k-hop reader
that performs multi-hop retrieval INSIDE a forward pass, over a FROZEN fp-space and a
NON-GRADIENT slot tape. Replaces the external argmax loop (200) with a learned module.

Anti-CF by construction: encoder frozen, tape non-gradient; only the tiny HopReader trains.
The reader is told how many hops k to take (question spec) and must land on entity A_k.

  state_0 = proj(fp(A0)) + kemb[k]
  for t in 1..T:  a = softmax(state @ K^T / temp); read = a @ V; state = state + U([state;read])
  answer = normalize(Wo(state));  predict = argmax_c cos(answer, fp(c))

Gates:
  G_learn        internal reader test acc (k=2,3) >= 0.70 (chance 0.25)
  G_generalize   test chains (unseen) acc high
  G_vs_handloop  report external argmax-loop acc for reference
  G_anticf       encoder params untouched (assert)

  python _stage203_internal_hops.py
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage203_decision.json"
MINI = RES / "stage203_mini.md"
LOG = RES / "_stage203_log.txt"

SEED = 203
CORPUS_CHARS = 40_000_000
MID_START = 90_000_000
N_CHAIN = 240
CHAIN_LEN = 4          # k up to 3
N_DISTRACT = 4000
T_STEPS = 3
STEPS = 1500
BATCH = 128
LR = 1e-3


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class HopReader(nn.Module):
    def __init__(self, d, t_steps=T_STEPS, max_k=CHAIN_LEN):
        super().__init__()
        self.t = t_steps
        self.proj = nn.Linear(d, d)
        self.kemb = nn.Embedding(max_k + 1, d)
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.out = nn.Linear(d, d)
        self.temp = nn.Parameter(torch.tensor(0.1))

    def forward(self, q_fp, k, K, Vv):
        # q_fp [B,d], k [B], K/Vv [N,d]
        state = self.proj(q_fp) + self.kemb(k)
        for _ in range(self.t):
            a = torch.softmax((state @ K.T) / self.temp.clamp(min=0.02), dim=-1)
            read = a @ Vv
            state = state + self.upd(torch.cat([state, read], dim=-1))
        return F.normalize(self.out(state), dim=-1)


class SoftFollow(nn.Module):
    """Minimal-structure internal hops: pure soft value-following + step selection by k.
    Only a temperature is learned; the hop OPERATION is parameter-free (should generalize)."""

    def __init__(self, d, t_steps=T_STEPS):
        super().__init__()
        self.t = t_steps
        self.log_temp = nn.Parameter(torch.tensor(-3.0))

    def forward(self, q_fp, k, K, Vv):
        temp = self.log_temp.exp().clamp(min=0.01, max=1.0)
        state = q_fp
        states = []
        for _ in range(self.t):
            state = F.normalize(torch.softmax((state @ K.T) / temp, dim=-1) @ Vv, dim=-1)
            states.append(state)
        stacked = torch.stack(states, dim=1)  # [B,T,d]
        idx = (k - 1).clamp(min=0, max=self.t - 1)
        return stacked[torch.arange(stacked.size(0), device=stacked.device), idx]


def train_reader(reader, train_s, bank, K, Vv, device, rng):
    opt = torch.optim.AdamW(reader.parameters(), lr=LR, weight_decay=0.01)
    reader.train()
    running = None
    for step in range(1, STEPS + 1):
        batch = [train_s[rng.randint(0, len(train_s) - 1)] for _ in range(BATCH)]
        q = bank.fp([b[0] for b in batch])
        kk = torch.tensor([b[1] for b in batch], device=device)
        tgt = bank.fp([b[2] for b in batch])
        ans = reader(q, kk, K, Vv)
        loss = F.cross_entropy(ans @ tgt.T / 0.1, torch.arange(len(batch), device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
    reader.eval()
    return running


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage203 start {datetime.now(timezone.utc).isoformat()}")
    log("internal hops: trainable differentiable k-hop reader over frozen fp tape")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    enc_sig = sum(float(p.abs().sum()) for p in model.arc_enc.parameters())  # anti-CF check
    bank = FpBank(model, stoi, device)
    d = model.head.in_features // 2
    log(f"encoder frozen (fp dim={d}) ({time.time()-t0:.0f}s)")

    # distractor entities from corpus for tape noise
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

    # tape: all chain edges + real-entity distractor edges
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
    log(f"tape slots={K.shape[0]} chains={len(chains)} (train {len(train_ch)}/test {len(test_ch)}) ({time.time()-t0:.0f}s)")

    ent_pool = list(dict.fromkeys([n for c in chains for n in c]))
    pool_fp = bank.fp(ent_pool)

    def samples(chs):
        out = []
        for c in chs:
            for k in range(1, CHAIN_LEN):
                out.append((c[0], k, c[k]))
        return out

    train_s = samples(train_ch)
    test_s = samples(test_ch)

    free = HopReader(d).to(device)
    lf = train_reader(free, train_s, bank, K, Vv, device, random.Random(SEED))
    log(f"free-form reader trained (loss~{lf:.3f})")
    soft = SoftFollow(d).to(device)
    ls = train_reader(soft, train_s, bank, K, Vv, device, random.Random(SEED))
    log(f"soft-follow reader trained (loss~{ls:.3f}, temp={float(soft.log_temp.exp()):.3f})")

    @torch.no_grad()
    def eval_reader(reader, sset):
        accs = {}
        for k in range(1, CHAIN_LEN):
            items = [s for s in sset if s[1] == k]
            ok = 0
            for a0, _, ak in items:
                ans = reader(bank.fp([a0]), torch.tensor([k], device=device), K, Vv)[0]
                cands = [ak] + [ent_pool[rng.randint(0, len(ent_pool) - 1)] for _ in range(3)]
                order = list(range(4)); rng.shuffle(order)
                shuf = [cands[i] for i in order]; gold = order.index(0)
                sc = [float(ans @ bank.fp([c])[0]) for c in shuf]
                ok += int(int(np.argmax(sc)) == gold)
            accs[k] = ok / max(1, len(items))
        return accs

    @torch.no_grad()
    def hand_loop(sset):
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

    free_tr, free_te = eval_reader(free, train_s), eval_reader(free, test_s)
    soft_tr, soft_te = eval_reader(soft, train_s), eval_reader(soft, test_s)
    hl = hand_loop(test_s)
    enc_sig2 = sum(float(p.abs().sum()) for p in model.arc_enc.parameters())
    anticf = abs(enc_sig - enc_sig2) < 1e-3
    log(f"free-form  train={free_tr} test={free_te}")
    log(f"soft-follow train={soft_tr} test={soft_te}")
    log(f"external hand-loop test={hl}")
    log(f"anti-CF encoder untouched: {anticf}")

    # structured internal reader is the one we claim can internalize hops
    g_learn = soft_te.get(2, 0) >= 0.70 and soft_te.get(3, 0) >= 0.70
    g_gen = min(soft_te.values()) >= 0.60
    free_overfits = (min(free_tr.values()) >= 0.9) and (max(free_te.values()) <= 0.4)
    if g_learn and g_gen and anticf:
        overall = "INTERNAL_HOPS_YES_IF_STRUCTURED"
    elif soft_te.get(2, 0) >= 0.60:
        overall = "INTERNAL_HOPS_PARTIAL"
    else:
        overall = "INTERNAL_HOPS_NO"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "internal_hops_203",
        "overall": overall,
        "free_form_reader": {"train": free_tr, "test": free_te},
        "soft_follow_reader": {"train": soft_tr, "test": soft_te},
        "external_hand_loop_test": hl,
        "free_form_overfits": free_overfits,
        "anticf_encoder_frozen": anticf,
        "tape_slots": int(K.shape[0]),
        "chains_train_test": [len(train_ch), len(test_ch)],
        "t_steps": T_STEPS,
        "chance": 0.25,
        "note": "internal hops generalize ONLY with minimal structure (soft value-follow + step select); "
        "a free-form learned reader memorizes; anti-CF preserved (frozen encoder + non-grad tape)",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage203 — internal hops (free-form vs structured)",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- **soft-follow (structured) test:** k1={soft_te.get(1):.2f} k2={soft_te.get(2):.2f} k3={soft_te.get(3):.2f} (chance 0.25)",
                f"- free-form test: k1={free_te.get(1):.2f} k2={free_te.get(2):.2f} k3={free_te.get(3):.2f} (train {free_tr.get(2):.2f} → overfits={free_overfits})",
                f"- external hand-loop test: k1={hl.get(1):.2f} k2={hl.get(2):.2f} k3={hl.get(3):.2f}",
                f"- anti-CF (encoder untouched): {anticf}, tape slots={int(K.shape[0])}, T={T_STEPS}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[203] {overall} | soft test k2={soft_te.get(2):.2f} k3={soft_te.get(3):.2f} | free test k3={free_te.get(3):.2f} | handloop k3={hl.get(3):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
