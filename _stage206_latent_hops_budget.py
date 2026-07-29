"""
Stage 206 — W5: latent hops as O(1)-encoder vector ops vs O(k) token/decode routes.

Four routes on the SAME planted k-hop chains (k up to 6, beyond window):
  curve_latent : encode the start entity ONCE, then k matvecs in fp-space (no decode, no re-encode)
  rag_text     : GPT index whose values are TEXT -> every hop decodes to a string and re-encodes (k+1 encoder calls)
  rag_vector   : HONESTY FLIP - GPT index whose values are GPT embeddings -> hops are matvecs too (1 encoder call)
  gpt_cot      : vanilla in-context reference (facts beyond window)

Conditions: clean, and noisy (independent char-noise realizations on stored anchors/values and the
query, as in 204) to test whether error COMPOUNDS across hops differently per substrate.

Metrics: accuracy vs k, encoder forward calls per query, wall-clock ms per query.

Gates:
  G_deep      curve_latent acc at k=6 >= 0.70 (chance 0.25)
  G_cheap     curve_latent >= 2x faster than rag_text and uses 1 encoder call vs k+1
  G_flip      does rag_vector tie curve_latent on CLEAN? (if yes -> architectural, report honestly)
  G_noise     curve_latent beats rag_vector at k>=4 under noise by >= 0.10 (compounding advantage)

  python _stage206_latent_hops_budget.py
"""
from __future__ import annotations

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
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _stage204_noise_robustness import noisy

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage206_decision.json"
MINI = RES / "stage206_mini.md"
LOG = RES / "_stage206_log.txt"

SEED = 206
CORPUS_CHARS = 20_000_000
N_CHAIN = 120
CHAIN_LEN = 7  # k up to 6
N_DISTRACT = 3000
KS = [1, 2, 4, 6]
NOISE_P = 0.15
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
    log(f"Stage206 start {datetime.now(timezone.utc).isoformat()}")
    log("W5: latent vector hops vs decode/token routes under compute budget")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    V = tok.get_vocab_size()
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bank = FpBank(model, stoi, device)
    gm = load_gpt(device)
    log(f"models loaded ({time.time()-t0:.0f}s)")

    gcache: dict[str, torch.Tensor] = {}
    calls = {"n": 0}

    @torch.no_grad()
    def gpt_word(w, count=True):
        """Cached GPT word embedding — used ONLY for offline index building."""
        if w in gcache:
            return gcache[w]
        ids = [i for i in tok.encode(" " + w).ids if i != pad_id][-MAX_ARCS:]
        h = gm.transformer(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0].mean(0)
        v = F.normalize(h, dim=-1)
        gcache[w] = v
        if count:
            calls["n"] += 1
        return v

    @torch.no_grad()
    def gpt_word_nc(w):
        """UNCACHED: what a query actually costs at inference (counted)."""
        calls["n"] += 1
        ids = [i for i in tok.encode(" " + w).ids if i != pad_id][-MAX_ARCS:]
        h = gm.transformer(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0].mean(0)
        return F.normalize(h, dim=-1)

    @torch.no_grad()
    def curve_fp_nc(w):
        """UNCACHED curve arc-encoder call (counted) — the curve's per-query encode cost."""
        calls["n"] += 1
        row = torch.zeros(1, 1, s177.MAX_CHARS_PER_ARC, dtype=torch.long)
        for j, ch in enumerate(w[: s177.MAX_CHARS_PER_ARC]):
            row[0, 0, j] = stoi.get(ch, 0)
        return F.normalize(model.arc_enc(row.to(device))[:, 0], dim=-1)[0]

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    real = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5))[:N_DISTRACT + 10]

    ents = [w for w in gen_fakes(set(real), rng, N_CHAIN * CHAIN_LEN + 200) if len(w) >= 5]
    chains = [ents[i * CHAIN_LEN : (i + 1) * CHAIN_LEN] for i in range(N_CHAIN)]
    chains = [c for c in chains if len(c) == CHAIN_LEN]
    pool = list(dict.fromkeys([w for c in chains for w in c]))
    log(f"chains={len(chains)} depth={CHAIN_LEN-1} pool={len(pool)} ({time.time()-t0:.0f}s)")

    def build_banks(noise_p):
        """Slots for all routes. Under noise, stored anchor/value use independent realizations."""
        nrng = random.Random(SEED + 17 + int(noise_p * 100))
        kc, vc, kg, vg, vstr = [], [], [], [], []
        for c in chains:
            for i in range(len(c) - 1):
                u_s = noisy(c[i], noise_p, nrng)
                v_s = noisy(c[i + 1], noise_p, nrng)
                kc.append(bank.fp([u_s])[0])
                vc.append(bank.fp([v_s])[0])
                kg.append(gpt_word(u_s, count=False))
                vg.append(gpt_word(v_s, count=False))
                vstr.append(v_s)
        for i in range(min(N_DISTRACT, len(real) - 1)):
            u_s = noisy(real[i], noise_p, nrng)
            v_s = noisy(real[i + 1], noise_p, nrng)
            kc.append(bank.fp([u_s])[0])
            vc.append(bank.fp([v_s])[0])
            kg.append(gpt_word(u_s, count=False))
            vg.append(gpt_word(v_s, count=False))
            vstr.append(v_s)
        return (
            torch.stack(kc, 0),
            torch.stack(vc, 0),
            torch.stack(kg, 0),
            torch.stack(vg, 0),
            vstr,
        )

    pool_c = bank.fp(pool)
    pool_g = torch.stack([gpt_word(w, count=False) for w in pool], 0)

    def cands_of(gold, qrng):
        others = [x for x in pool if x != gold]
        qrng.shuffle(others)
        c = [gold] + others[:3]
        order = list(range(4))
        qrng.shuffle(order)
        return [c[i] for i in order], order.index(0)

    def run_routes(noise_p):
        Kc, Vc, Kg, Vg, vstr = build_banks(noise_p)
        nrng = random.Random(SEED + 41 + int(noise_p * 100))
        qnoise = {c[0]: noisy(c[0], noise_p, nrng) for c in chains}
        res = {}

        for name in ("curve_latent", "curve_latent_snap", "rag_vector", "rag_vector_snap", "rag_text"):
            accs, times, ecalls = {}, {}, {}
            for k in KS:
                qrng = random.Random(SEED + 7)
                ok = 0
                calls["n"] = 0
                if device.type == "cuda":
                    torch.cuda.synchronize()
                tstart = time.time()
                for c in chains:
                    q0 = qnoise[c[0]]
                    if name.startswith("curve_latent"):
                        v = curve_fp_nc(q0)
                        for _ in range(k):
                            v = Vc[int((Kc @ v).argmax())]
                            if name.endswith("snap"):  # re-anchor to the entity lexicon each hop
                                v = pool_c[int((pool_c @ v).argmax())]
                        scores = pool_c @ v
                    elif name.startswith("rag_vector"):
                        v = gpt_word_nc(q0)
                        for _ in range(k):
                            v = Vg[int((Kg @ v).argmax())]
                            if name.endswith("snap"):
                                v = pool_g[int((pool_g @ v).argmax())]
                        scores = pool_g @ v
                    else:  # rag_text: decode to a string and re-encode every hop
                        cur = q0
                        for _ in range(k):
                            v = gpt_word_nc(cur)
                            cur = vstr[int((Kg @ v).argmax())]
                        v = gpt_word_nc(cur)
                        scores = pool_g @ v
                    shuf, gold = cands_of(c[k], qrng)
                    sc = [float(scores[pool.index(s)]) for s in shuf]
                    ok += int(int(np.argmax(sc)) == gold)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - tstart) / len(chains) * 1000.0
                accs[k] = ok / len(chains)
                times[k] = dt
                ecalls[k] = calls["n"] / len(chains)
                log(
                    f"  [{'noisy' if noise_p else 'clean'}] {name} k={k}: acc={accs[k]:.3f} "
                    f"{dt:.2f} ms/query enc_calls={ecalls[k]:.2f} ({time.time()-t0:.0f}s)"
                )
            res[name] = {"acc": accs, "ms": times, "enc_calls": ecalls}
        return res

    clean = run_routes(0.0)
    noisyres = run_routes(NOISE_P)

    # vanilla in-context reference (facts beyond window)
    stream = " ".join(f"{c[i]} leads to {c[i+1]} ." for c in chains for i in range(len(c) - 1))
    tail = [i for i in tok.encode(stream).ids if i != pad_id][-MAX_ARCS + 8 :]
    qrng = random.Random(SEED + 7)
    ok = 0
    for c in chains[:60]:
        cue = [i for i in tok.encode(f" {c[0]} eventually leads to").ids if i != pad_id]
        ctx = (tail + cue)[-MAX_ARCS:]
        shuf, gold = cands_of(c[2], qrng)
        sc = [gpt_span(gm, device, ctx, [i for i in tok.encode(" " + s).ids if i != pad_id]) for s in shuf]
        ok += int(int(np.argmax(sc)) == gold)
    cot_acc = ok / 60
    log(f"  gpt in-context (k=2, beyond window) acc={cot_acc:.3f}")

    cl, rv, rt = clean["curve_latent"], clean["rag_vector"], clean["rag_text"]
    cls = clean["curve_latent_snap"]
    ncl, nrv = noisyres["curve_latent"], noisyres["rag_vector"]
    ncls, nrvs = noisyres["curve_latent_snap"], noisyres["rag_vector_snap"]
    g_deep = cl["acc"][6] >= 0.70
    g_cheap = rt["ms"][6] >= 2 * cl["ms"][6] and rt["enc_calls"][6] >= 2
    g_flip_ties = abs(rv["acc"][6] - cl["acc"][6]) <= 0.05
    g_noise = all(ncl["acc"][k] >= nrv["acc"][k] + 0.10 for k in (4, 6))
    g_snap = all(ncls["acc"][k] >= nrvs["acc"][k] + 0.10 for k in (4, 6))
    g_snap_helps = ncls["acc"][6] >= ncl["acc"][6] + 0.10
    if g_deep and g_cheap and (g_noise or g_snap):
        overall = "LATENT_HOPS_CHEAP_AND_NOISE_ROBUST"
    elif g_deep and g_cheap and g_flip_ties:
        overall = "LATENT_HOPS_CHEAPER_BUT_RAG_VECTOR_TIES"
    else:
        overall = "LATENT_HOPS_PARTIAL"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "latent_hops_budget_206",
        "overall": overall,
        "clean": {k: {m: {str(kk): vv for kk, vv in d.items()} for m, d in v.items()} for k, v in clean.items()},
        "noisy": {k: {m: {str(kk): vv for kk, vv in d.items()} for m, d in v.items()} for k, v in noisyres.items()},
        "noise_p": NOISE_P,
        "gpt_incontext_k2": cot_acc,
        "gates": {
            "g_deep": g_deep,
            "g_cheap": g_cheap,
            "g_rag_vector_ties_clean": g_flip_ties,
            "g_noise": g_noise,
            "g_noise_with_snap": g_snap,
            "g_snap_helps_curve": g_snap_helps,
        },
        "chance": CHANCE,
        "note": "rag_vector is the honesty flip: a vector-valued GPT index also hops with 1 encoder call, so a "
        "compute win over rag_text alone is architectural; the substrate claim must come from noise compounding",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage206 — W5 latent hops under compute budget",
                "",
                f"**Overall:** `{overall}`",
                "",
                "**Clean (compute):**",
                "",
                "| k | curve_latent acc/ms/enc | rag_vector acc/ms/enc | rag_text acc/ms/enc |",
                "|---|-------------------------|-----------------------|---------------------|",
            ]
            + [
                f"| {k} | {cl['acc'][k]:.3f} / {cl['ms'][k]:.2f} / {cl['enc_calls'][k]:.1f} "
                f"| {rv['acc'][k]:.3f} / {rv['ms'][k]:.2f} / {rv['enc_calls'][k]:.1f} "
                f"| {rt['acc'][k]:.3f} / {rt['ms'][k]:.2f} / {rt['enc_calls'][k]:.1f} |"
                for k in KS
            ]
            + [
                "",
                f"**Noisy (p={NOISE_P}) — compounding across hops, with and without lexicon re-anchoring (snap):**",
                "",
                "| k | curve | curve+snap | rag_vector | rag_vector+snap |",
                "|---|-------|-----------|------------|-----------------|",
            ]
            + [
                f"| {k} | {ncl['acc'][k]:.3f} | **{ncls['acc'][k]:.3f}** | {nrv['acc'][k]:.3f} | {nrvs['acc'][k]:.3f} |"
                for k in KS
            ]
            + [
                "",
                f"- vanilla GPT in-context (k=2, beyond window): {cot_acc:.3f} (chance {CHANCE})",
                f"- gates: deep={g_deep} cheap={g_cheap} rag_vector_ties_clean={g_flip_ties} "
                f"noise={g_noise} noise_with_snap={g_snap} snap_helps_curve={g_snap_helps}",
            ]
        ),
        encoding="utf-8",
    )
    log(
        f"[206] {overall} | clean k6 curve={cl['acc'][6]:.2f} ragvec={rv['acc'][6]:.2f} ragtext={rt['acc'][6]:.2f} "
        f"| noisy k6 curve={ncl['acc'][6]:.2f} ragvec={nrv['acc'][6]:.2f} | ms k6 curve={cl['ms'][6]:.2f} vs ragtext={rt['ms'][6]:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
