"""
Stage 258 — Semantic query: let the trunk's understanding reach the retrieval.

209 measured something that has been sitting unused: curve states carry MiniLM geometry
at least as well as matched GPT (r 0.256 vs 0.270), and PAWS tracks GPT at every scale
(STRUCTURAL_BLOCK_NO). Meanwhile the retrieval query never sees the trunk at all:

    q = normalize( W_q( ctx_fp(prefix) [+ fp(anchor)] ) )      W_q maps fp -> fp

`ctx_fp` is a MEAN over word fingerprints — a bag of spellings, no word order, no meaning.
`h_t` only reaches the read gate ("should I look at the tape now?"), never the query
("what am I looking for?"). So the tape can only be asked with the literal anchor string.

This stage adds one channel and changes nothing else:

    q_fp  = normalize( W_q(fp query) )                 existing path, untouched
    q_sem = normalize( W_sem(h_t) )                    trunk understanding -> key space
    q     = normalize( (1 - a) * q_fp + a * q_sem )    a = sigmoid(MLP(h_t)), starts ~0.12

Keys stay canonical frozen fp, P1 stays frozen, the trunk stays frozen. Only W_q, W_sem
and the blend train. At a=0 this is EXACTLY stage 256, so the baseline is a special case
of the model rather than a separate implementation.

The exam is built so the fp path CANNOT win, by construction. One subject S carries FOUR
facts with different relations. Every one of those slots has the SAME anchor fp(S), so the
anchor contributes nothing to telling them apart, and the query paraphrases the relation
with NO content word in common with the written sentence:

    written : "{S} was appointed director of {V} in the regional chronicle of 1987 ."
    asked   : "the body that {S} led was named"          (led / body / named vs appointed / director)

A bag of spellings cannot bridge "led" to "appointed director". Trunk semantics might.
Chance is 1/4 and fp-only is PRE-REGISTERED to sit there (G_fp_only_at_chance) — if it
does not, the exam leaks and the rest of the numbers mean nothing.

Held out twice, so a win cannot be four memorised templates:
    seen_rel       unseen SUBJECTS, paraphrase A (fit during training)
    unseen_para    unseen SUBJECTS, paraphrase B of the SAME relations (258c — not held-out
                   relations, which were structurally penalized as InfoNCE negatives)

Matched GPT-2 control (the 210-212 lesson): those stages closed as THESIS_NO on a single
scale with no control, which reads as "impossible" when the evidence only said "not here".
Here the same semantic channel is trained on GPT-2 states too. If curve fails and GPT also
fails, the verdict is SEM_QUERY_NO_AT_SCALE, not SEM_QUERY_NO — a statement about a 3050,
not about the architecture. If GPT succeeds where curve does not, that IS architectural and
the verdict says so.

Retrieval only: whether a won slot reaches the output is already 256's result, so this
stage does not touch decode and cannot break language.

  python _stage258_semantic_query.py [--smoke] [--no-gpt-control]
"""
from __future__ import annotations

import argparse
import json
import math
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
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, hidden_and_logits

RES = Path("results")
DECISION = RES / "stage258_decision.json"
MINI = RES / "stage258_mini.md"
LOG = RES / "_stage258_log.txt"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage258_semantic_query.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 258

# Four relations on ONE subject. Query wording shares no content word with the written form —
# only stopwords — so a bag of spellings is at chance and semantics is the only bridge.
RELS = {
    "lead": {
        "write": "{S} was appointed director of {V} in the regional chronicle of 1987 .",
        "para": "the body that {S} led was named",
        "para_b": "the organisation under {S} leadership was called",
        "para_hold": "which institution {S} commanded was known as",
        "anchored": "in the ledger {S} was appointed director of",
    },
    "birth": {
        "write": "{S} was born in {V} according to the parish register .",
        "para": "the birthplace recorded for {S} is",
        "para_b": "records show {S} entered the world in",
        "para_hold": "the city listed as birthplace of {S} is",
        "anchored": "in the ledger {S} was born in",
    },
    "death": {
        "write": "{S} died in {V} at the close of the season .",
        "para": "the place where {S} passed away is",
        "para_b": "the locale of {S} death is given as",
        "para_hold": "where {S} drew their final breath was",
        "anchored": "in the ledger {S} died in",
    },
    "marriage": {
        "write": "{S} married in {V} during the spring assembly .",
        "para": "the wedding of {S} was held in",
        "para_b": "the town of {S} nuptials was",
        "para_hold": "the municipality hosting {S} wedding was",
        "anchored": "in the ledger {S} married in",
    },
    "work": {
        "write": "{S} worked at {V} for eleven consecutive terms .",
        "para": "the employer of {S} was listed as",
        "para_b": "the workplace associated with {S} was",
        "para_hold": "the firm recorded as employer of {S} was",
        "anchored": "in the ledger {S} worked at",
    },
    "prison": {
        "write": "{S} was jailed in {V} following the tribunal .",
        "para": "the prison that confined {S} stood in",
        "para_b": "the gaol holding {S} was situated in",
        "para_hold": "the detention site holding {S} was located in",
        "anchored": "in the ledger {S} was jailed in",
    },
    "study": {
        "write": "{S} studied at {V} before the reorganisation .",
        "para": "the school that taught {S} is called",
        "para_b": "the college attended by {S} was",
        "para_hold": "the academy {S} attended was known as",
        "anchored": "in the ledger {S} studied at the",
    },
    "burial": {
        "write": "{S} was buried in {V} after the civic ceremony .",
        "para": "the grave of {S} lies in",
        "para_b": "the resting place of {S} is recorded as",
        "para_hold": "the cemetery where {S} was interred is in",
        "anchored": "in the ledger {S} was buried in",
    },
}
# Six fit relations, not two. With a 2-way fit the projection only ever learned to point at two
# kinds of slot, so an unseen paraphrase landed inside that learned corridor and scored BELOW
# chance (0.000 vs 0.250) - "output space collapsed", not "no semantics". Six gives the channel
# a wider target space; the confusion matrix below tells the two apart directly.
FIT_RELS = ["lead", "birth", "death", "marriage", "work", "prison"]
# 258c: generalization is paraphrase B of fit relations, not held-out relation types (those slots
# were InfoNCE negatives during training).
CHANCE = 1.0 / len(RELS)


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class SemQuery(nn.Module):
    """Trunk state -> key space, plus how much to trust it. Starts near a=0.12 so the fp path
    dominates at init and the semantic channel has to earn its weight against CE."""

    def __init__(self, in_dim: int, device):
        super().__init__()
        self.proj = nn.Linear(in_dim, 256).to(device)
        # +2: how well the fp path is already doing. Without it the blend only sees h_t, cannot
        # tell "fp is lost" from "fp is perfect", and taxes anchored queries it should leave alone
        # (the 2/2 run: anchored 1.000 -> 0.583 at a=0.23).
        self.blend = nn.Sequential(nn.Linear(in_dim + 2, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.blend[-1].weight)
        nn.init.constant_(self.blend[-1].bias, -2.0)

    def q(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(h), dim=-1)

    def a(self, h: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.blend(torch.cat([h, conf], dim=-1))).squeeze(-1)


@torch.no_grad()
def curve_state(model, char_table, tok, pad_id, device, text: str) -> torch.Tensor:
    ids = [i for i in tok.encode(text).ids if i != pad_id][-MAX_ARCS:]
    if not ids:
        return None
    h, _ = hidden_and_logits(model, char_table, torch.tensor([ids], device=device), pad_id)
    return h[0, -1].detach()


@torch.no_grad()
def gpt_state(gm, tok, pad_id, device, text: str) -> torch.Tensor:
    ids = [i for i in tok.encode(text).ids if i != pad_id]
    e = L.gpt_emb(gm, tok, pad_id, device, ids)
    return None if e is None else e.detach()


def fp_query_raw(bank: FpBank, text: str):
    """Exactly the 256 recipe: anchor fingerprint + bag-of-spellings context."""
    c = bank.ctx_fp(text)
    if c is None:
        return None
    anchors = ANCHOR_RE.findall(text)
    return F.normalize(bank.fp([anchors[-1]])[0] + c, dim=-1) if anchors else c


def fp_confidence(q_fp: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """[top1, margin] of the fp-only query. Raw margins are ~0.02–0.05 — scale so the blend MLP
    can see them next to h_t. Batch callers may z-score on top of this."""
    if q_fp.dim() == 1:
        q_fp = q_fp.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    sims = q_fp @ K.t()
    two = torch.topk(sims, min(2, sims.size(-1)), dim=-1).values
    top1 = two[..., 0]
    margin = top1 - two[..., 1] if two.size(-1) > 1 else top1
    # map into ~O(1): top1 already in [-1,1]; margin * 20 → typical 0.4–1.0
    conf = torch.stack([top1 * 2.0 - 1.0, margin * 20.0], dim=-1)
    return conf[0] if squeeze else conf


def zscore_conf(conf: torch.Tensor) -> torch.Tensor:
    """Per-feature z-score over the batch so relative fp-margin drives the blend."""
    if conf.size(0) < 2:
        return conf
    return (conf - conf.mean(0, keepdim=True)) / (conf.std(0, keepdim=True) + 1e-5)


def blended_query(W_q, semq: SemQuery, q_fp_raw, h, K=None):
    """(1-a) * W_q(fp) + a * W_sem(h). a=0 reproduces stage 256 bit for bit."""
    q_fp = F.normalize(W_q(q_fp_raw.unsqueeze(0)), dim=-1)[0]
    if semq is None or h is None:
        return q_fp, torch.zeros((), device=q_fp.device)
    a = semq.a(h, fp_confidence(q_fp, K))
    return F.normalize((1.0 - a) * q_fp + a * semq.q(h), dim=-1), a


def make_subjects(pool, rng, n_subj: int, n_val: int):
    """One subject, four relations, four distinct values -> four slots with the same anchor fp."""
    subs = [w for w in gen_fakes(set(pool), rng, n_subj + 40) if len(w) >= 5][:n_subj]
    vals = [v for v in pool if len(v) >= 5][: n_val]
    out = []
    for i, S in enumerate(subs):
        facts = {}
        for r in RELS:
            facts[r] = vals[rng.randrange(len(vals))]
        out.append({"S": S, "facts": facts, "sid": f"s{i}"})
    return out


def build_tape(bank: FpBank, subjects, device):
    keys, values, owner = [], [], []
    for su in subjects:
        for r in RELS:
            v = su["facts"][r]
            sent = RELS[r]["write"].format(S=su["S"], V=v)
            c = bank.ctx_fp(sent, exclude=v)
            kf = bank.fp([su["S"]])[0]
            keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
            values.append(v)
            owner.append((su["sid"], r))
    return keys, values, owner


def queries_for(subjects, rels, kind: str):
    return [
        {"sid": su["sid"], "S": su["S"], "rel": r, "kind": kind,
         "text": RELS[r][kind].format(S=su["S"]), "value": su["facts"][r]}
        for su in subjects
        for r in rels
    ]


@torch.no_grad()
def evaluate(W_q, semq, bank, K, owner_idx, qs, states, use_sem: bool):
    """Primary metric: among the FOUR slots of this subject, does the asked relation win?
    Chance is 1/4 by construction. Bank-wide rank is reported so a semantic channel that
    wins locally by wrecking global retrieval cannot pass unnoticed."""
    sel, ranks, alphas, skipped = [], [], [], 0
    confusion: dict[str, dict[str, int]] = {}
    for q in qs:
        raw = fp_query_raw(bank, q["text"])
        h = states.get(q["text"]) if use_sem else None
        if raw is None:
            skipped += 1
            continue
        qq, a = blended_query(W_q, semq if use_sem else None, raw, h, K)
        sims = K @ qq
        sib = owner_idx[q["sid"]]
        best = max(sib, key=lambda j: float(sims[j[1]]))
        sel.append(int(best[0] == q["rel"]))
        confusion.setdefault(q["rel"], {}).setdefault(best[0], 0)
        confusion[q["rel"]][best[0]] += 1
        gold = [j for r, j in sib if r == q["rel"]][0]
        ranks.append(1 + int((sims > sims[gold]).sum()))
        alphas.append(float(a))
    r = np.asarray(ranks, dtype=np.float64) if ranks else np.asarray([np.nan])
    return {
        "sel_acc": float(np.mean(sel)) if sel else float("nan"),
        "bank_top1": float(np.mean(r == 1)),
        "bank_mrr": float(np.mean(1.0 / r)),
        "alpha": float(np.mean(alphas)) if alphas else float("nan"),
        "n": len(sel),
        "skipped": skipped,
        # which relation did it actually pick? collapse onto fit relations looks completely
        # different from "does not understand", and sel_acc alone cannot tell them apart.
        "confusion": confusion,
    }


def hard_neg_ce(logits: torch.Tensor, gold: torch.Tensor, k_hard: int) -> torch.Tensor:
    """CE over {gold} ∪ top-k hardest bank slots (by current logits). Focuses gradient on
    confusable keys instead of diluting across the whole bank softmax."""
    B, N = logits.shape
    k = min(max(1, k_hard), max(1, N - 1))
    masked = logits.detach().clone()
    masked.scatter_(1, gold.view(-1, 1), float("-inf"))
    neg = masked.topk(k, dim=-1).indices
    idx = torch.cat([gold.view(-1, 1), neg], dim=1)
    gathered = logits.gather(1, idx)
    return F.cross_entropy(gathered, torch.zeros(B, dtype=torch.long, device=logits.device))


def train_channel(
    W_q, semq, bank, K, fit_qs, states, wiki_q, wiki_gold, steps, lr, tau, rng, tag,
    *, blend_l1: float = 0.0, k_hard: int = 32,
):
    """InfoNCE with hard-negative mining over the bank + wiki grounding for W_q."""
    params = list(semq.parameters()) + list(W_q.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    raws, hs, golds, kinds = [], [], [], []
    for q in fit_qs:
        raw = fp_query_raw(bank, q["text"])
        h = states.get(q["text"])
        if raw is None or h is None:
            continue
        raws.append(raw)
        hs.append(h)
        golds.append(q["gold_idx"])
        kinds.append(q.get("kind", "para"))
    if not raws:
        return float("nan")
    Rq = torch.stack(raws)
    Hq = torch.stack(hs)
    Gq = torch.tensor(golds, device=K.device)
    is_anch = torch.tensor([k == "anchored" for k in kinds], device=K.device)
    losses = []
    for step in range(1, steps + 1):
        sel = torch.randint(0, Rq.size(0), (min(32, Rq.size(0)),), device=K.device)
        q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
        conf = zscore_conf(fp_confidence(q_fp, K))
        a = semq.a(Hq[sel], conf).unsqueeze(-1)
        q = F.normalize((1.0 - a) * q_fp + a * semq.q(Hq[sel]), dim=-1)
        logits = (q @ K.t()) / tau
        # full-bank CE keeps global ranking; hard-neg focuses on the confusing top-k
        loss = 0.5 * F.cross_entropy(logits, Gq[sel]) + 0.5 * hard_neg_ce(logits, Gq[sel], k_hard)
        if blend_l1 > 0:
            with torch.no_grad():
                fp_hit = (q_fp @ K.t()).argmax(dim=-1) == Gq[sel]
            mask = is_anch[sel] & fp_hit
            if mask.any():
                loss = loss + blend_l1 * a.squeeze(-1)[mask].mean()
        if wiki_q is not None:
            w = torch.randint(0, wiki_q.size(0), (min(64, wiki_q.size(0)),), device=K.device)
            qw = F.normalize(W_q(wiki_q[w]), dim=-1)
            w_logits = (qw @ K.t()) / tau
            loss = loss + 0.5 * F.cross_entropy(w_logits, wiki_gold[w]) + 0.5 * hard_neg_ce(
                w_logits, wiki_gold[w], k_hard
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss))
        if step % max(1, steps // 5) == 0:
            log(f"  [{tag}] step {step}/{steps} loss={float(loss):.3f} a={float(a.mean()):.3f}")
    return float(np.mean(losses[-20:])) if losses else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument(
        "--blend-l1",
        type=float,
        default=0.25,
        help="L1 on blend a for anchored fit queries where fp-only already hits gold",
    )
    ap.add_argument("--k-hard", type=int, default=32, help="top-k hard negatives for InfoNCE mining")
    ap.add_argument("--no-gpt-control", action="store_true")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    steps = args.steps or (150 if args.smoke else 600)
    n_subj = args.subjects or (12 if args.smoke else 64)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_lines = 400 if args.smoke else 6000

    log(
        f"Stage258 semantic query start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"steps={steps} subjects={n_subj} distractors={n_dist} chance={CHANCE:.3f}"
    )

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
    log(f"  trunk={trunk_ckpt.name} (frozen)  keys=canonical P1 (frozen)")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_000_000 if args.smoke else 6_000_000)
    pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:max_lines]

    subjects = make_subjects(pool, rng, n_subj, min(len(pool), 400))
    n_fit = n_subj // 2
    fit_subj, ev_subj = subjects[:n_fit], subjects[n_fit:]
    log(f"  subjects: fit={len(fit_subj)} held_out={len(ev_subj)}  rels fit={FIT_RELS}")

    keys, values, owner = build_tape(bank, subjects, device)
    n_slot = len(values)
    used = {su["S"] for su in subjects} | set(values)
    wiki_raw, wiki_slot = [], []
    for ln in lines:
        if len(values) >= n_slot + n_dist:
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
            a_fp = bank.fp([anchors[-1]])[0]
            keys.append(F.normalize(a_fp + c, dim=-1))
            cq = bank.ctx_fp(ln[lo : m.start()])
            if cq is not None:
                wiki_raw.append(F.normalize(a_fp + cq, dim=-1))
                wiki_slot.append(len(values))
            values.append(ent)
            owner.append((None, None))
            used.add(ent)
            if len(values) >= n_slot + n_dist:
                break
    K = torch.stack(keys, 0).to(device).float()
    log(f"  tape slots={len(values)} ({n_slot} subject facts + {len(values)-n_slot} wiki noise)")

    owner_idx: dict[str, list] = {}
    for j, (sid, r) in enumerate(owner):
        if sid is not None:
            owner_idx.setdefault(sid, []).append((r, j))
    gold_of = {(sid, r): j for j, (sid, r) in enumerate(owner) if sid is not None}

    fit_qs = (
        queries_for(fit_subj, FIT_RELS, "para")
        + queries_for(fit_subj, FIT_RELS, "para_b")
        + queries_for(fit_subj, FIT_RELS, "anchored")
    )
    for q in fit_qs:
        q["gold_idx"] = gold_of[(q["sid"], q["rel"])]
    ev_seen = queries_for(ev_subj, FIT_RELS, "para")
    ev_unseen = queries_for(ev_subj, FIT_RELS, "para_hold")
    ev_anchored = queries_for(ev_subj, list(RELS), "anchored")
    log(
        f"  queries: fit={len(fit_qs)} seen_rel={len(ev_seen)} unseen_para={len(ev_unseen)} "
        f"anchored={len(ev_anchored)}"
    )

    all_qs = fit_qs + ev_seen + ev_unseen + ev_anchored
    texts = sorted({q["text"] for q in all_qs})

    # trunk is frozen -> every state is computed exactly once and cached
    st_curve = {}
    for t in texts:
        s = curve_state(model, char_table, tok, pad_id, device, t)
        if s is not None:
            st_curve[t] = s.float()
    d_curve = next(iter(st_curve.values())).numel()
    log(f"  cached curve states: {len(st_curve)} (dim {d_curve})")

    wiki_q = torch.stack(wiki_raw).to(device).float() if wiki_raw else None
    wiki_gold = torch.tensor(wiki_slot, device=device) if wiki_slot else None

    W_q = L.init_query_adapter(device)   # same near-identity init as 256/257
    semq = SemQuery(d_curve, device)
    K_snapshot = K.clone()

    base_seen = evaluate(W_q, None, bank, K, owner_idx, ev_seen, st_curve, use_sem=False)
    base_unseen = evaluate(W_q, None, bank, K, owner_idx, ev_unseen, st_curve, use_sem=False)
    base_anch = evaluate(W_q, None, bank, K, owner_idx, ev_anchored, st_curve, use_sem=False)
    log(
        f"fp-only (a=0, i.e. stage 256): seen_rel={base_seen['sel_acc']:.3f} "
        f"unseen_para={base_unseen['sel_acc']:.3f} anchored={base_anch['sel_acc']:.3f} (chance {CHANCE:.3f})"
    )

    loss_curve = train_channel(
        W_q, semq, bank, K, fit_qs, st_curve, wiki_q, wiki_gold, steps, args.lr, args.tau, rng, "curve",
        blend_l1=args.blend_l1, k_hard=args.k_hard,
    )
    sem_seen = evaluate(W_q, semq, bank, K, owner_idx, ev_seen, st_curve, use_sem=True)
    sem_unseen = evaluate(W_q, semq, bank, K, owner_idx, ev_unseen, st_curve, use_sem=True)
    sem_anch = evaluate(W_q, semq, bank, K, owner_idx, ev_anchored, st_curve, use_sem=True)
    log(
        f"curve+sem: seen_rel={sem_seen['sel_acc']:.3f} unseen_para={sem_unseen['sel_acc']:.3f} "
        f"anchored={sem_anch['sel_acc']:.3f} | alpha para={sem_unseen['alpha']:.3f} "
        f"anchored={sem_anch['alpha']:.3f} | bank_top1={sem_unseen['bank_top1']:.3f}"
    )

    # keys must be untouched — the whole editability story rests on it
    keys_intact = bool(torch.equal(K, K_snapshot))

    shuf = K[torch.randperm(K.size(0), generator=torch.Generator().manual_seed(SEED + 1))]
    sem_shuf = evaluate(W_q, semq, bank, shuf, owner_idx, ev_unseen, st_curve, use_sem=True)

    # ---- matched GPT-2 control: the 210-212 lesson, applied ----
    gpt = None
    if not args.no_gpt_control:
        try:
            gm = L.load_gpt(device)
            st_gpt = {}
            for t in texts:
                s = gpt_state(gm, tok, pad_id, device, t)
                if s is not None:
                    st_gpt[t] = s.float()
            if st_gpt:
                d_gpt = next(iter(st_gpt.values())).numel()
                W_q_g = L.init_query_adapter(device)  # fp path identical; only the sem input differs
                semq_g = SemQuery(d_gpt, device)
                loss_gpt = train_channel(
                    W_q_g, semq_g, bank, K, fit_qs, st_gpt, wiki_q, wiki_gold, steps, args.lr,
                    args.tau, rng, "gpt", blend_l1=args.blend_l1, k_hard=args.k_hard,
                )
                g_seen = evaluate(W_q_g, semq_g, bank, K, owner_idx, ev_seen, st_gpt, use_sem=True)
                g_unseen = evaluate(W_q_g, semq_g, bank, K, owner_idx, ev_unseen, st_gpt, use_sem=True)
                gpt = {"dim": d_gpt, "loss": loss_gpt, "seen_rel": g_seen, "unseen_para": g_unseen}
                log(f"gpt2+sem: seen_rel={g_seen['sel_acc']:.3f} unseen_para={g_unseen['sel_acc']:.3f}")
        except Exception as e:  # noqa: BLE001 - control is optional, never block the stage on it
            log(f"  gpt control unavailable: {type(e).__name__}: {e}")

    g_fp_at_chance = base_unseen["sel_acc"] <= CHANCE + 0.10
    g_sem_beats_fp = sem_unseen["sel_acc"] >= base_unseen["sel_acc"] + 0.15
    g_sem_beats_fp_seen = sem_seen["sel_acc"] >= base_seen["sel_acc"] + 0.15
    g_unseen_para = sem_unseen["sel_acc"] >= CHANCE + 0.20
    g_seen_rel = sem_seen["sel_acc"] >= CHANCE + 0.20
    g_anchored_intact = sem_anch["sel_acc"] >= base_anch["sel_acc"] - 0.05
    g_bankwide = sem_unseen["bank_top1"] >= 0.50
    g_sem_selective = (
        not math.isnan(sem_unseen["alpha"])
        and not math.isnan(sem_anch["alpha"])
        and sem_unseen["alpha"] >= sem_anch["alpha"] + 0.05
    )
    g_tape_causal = sem_shuf["sel_acc"] <= CHANCE + 0.10
    g_keys_frozen = keys_intact

    gpt_also_fails = (
        gpt is not None and gpt["unseen_para"]["sel_acc"] < CHANCE + 0.20
    )
    # Validity is about the EXAM, not about results. The 2/2 run put g_anchored_intact inside
    # `core`, so a real finding (the blend over-applies where fp is already perfect) voided the
    # whole run as INVALID and its actual result — seen relations 0.000 -> 1.000 on unseen
    # subjects — went unread. Anchored regression is now a result gate, where it belongs.
    valid = g_fp_at_chance and g_keys_frozen and g_tape_causal
    unseen_reading = (
        "gpt_parity" if gpt_also_fails else ("curve_gap" if gpt is not None else "no_control")
    )
    if not valid:
        overall = "SEM_QUERY_INVALID"  # exam leaked or keys moved; nothing else is interpretable
    elif g_sem_beats_fp and g_unseen_para and g_bankwide and g_anchored_intact:
        overall = "SEM_QUERY_OK"
    elif g_sem_beats_fp_seen and g_seen_rel:
        overall = "SEM_QUERY_PARTIAL"  # bridges fit paraphrases; unseen_para is the real claim
    elif gpt_also_fails:
        overall = "SEM_QUERY_NO_AT_SCALE"  # matched GPT no better — a 3050 statement, not architecture
    else:
        overall = "SEM_QUERY_NO"

    out = {
        "stage": 258,
        "overall": overall,
        "trunk": trunk_ckpt.name,
        "fp_version": L.canonical_fp_version(),
        "chance": CHANCE,
        "steps": steps,
        "n_subjects": len(subjects),
        "n_fit_subjects": len(fit_subj),
        "n_eval_subjects": len(ev_subj),
        "fit_rels": FIT_RELS,
        "exam_holdout": "para_hold (same relations, alternate paraphrase on held-out subjects)",
        "tape_slots": len(values),
        "subject_slots": n_slot,
        "curve_state_dim": d_curve,
        "loss_curve": loss_curve,
        "unseen_reading": unseen_reading,
        "gates": {
            "G_fp_only_at_chance": g_fp_at_chance,
            "G_sem_beats_fp": g_sem_beats_fp,
            "G_sem_beats_fp_seen": g_sem_beats_fp_seen,
            "G_unseen_para": g_unseen_para,
            "G_seen_rel": g_seen_rel,
            "G_anchored_intact": g_anchored_intact,
            "G_bankwide_retrieval": g_bankwide,
            "G_sem_selective": g_sem_selective,
            "G_tape_causal": g_tape_causal,
            "G_keys_frozen": g_keys_frozen,
        },
        "summary": {
            "fp_only": {"seen_rel": base_seen, "unseen_para": base_unseen, "anchored": base_anch},
            "curve_sem": {"seen_rel": sem_seen, "unseen_para": sem_unseen, "anchored": sem_anch},
            "curve_sem_shuffled_keys": sem_shuf,
            "gpt_control": gpt,
        },
        "note": "Keys stay canonical frozen fp; P1 and trunk frozen; only W_q, W_sem and the blend "
        "train. a=0 reproduces stage 256 exactly, so fp-only is the same code path, not a "
        "reimplementation. The exam gives one subject four relations, so every candidate slot shares "
        "the anchor fingerprint and the query paraphrases the relation with no shared content word — "
        "fp-only is pre-registered at chance (G_fp_only_at_chance); if that gate fails the exam leaks "
        "and nothing else here is interpretable. unseen_para is paraphrase B on held-out subjects "
        "(258c). Fit training uses para + anchored so the blend sees fp-perfect queries. "
        "The matched GPT-2 channel exists so a negative can be told apart from a small "
        "GPU: NO_AT_SCALE means GPT did no better, which is what 210-212 never checked.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 258 semantic query\n\n**{overall}** trunk={trunk_ckpt.name} slots={len(values)} "
        f"chance={CHANCE:.2f}\n\n"
        f"- unseen paraphrase: fp-only **{base_unseen['sel_acc']:.3f}** -> +sem **{sem_unseen['sel_acc']:.3f}**\n"
        f"- seen relation:   fp-only {base_seen['sel_acc']:.3f} -> +sem {sem_seen['sel_acc']:.3f}\n"
        f"- anchored (must not regress): {base_anch['sel_acc']:.3f} -> {sem_anch['sel_acc']:.3f}\n"
        f"- blend a: paraphrase {sem_unseen['alpha']:.3f} vs anchored {sem_anch['alpha']:.3f} "
        f"(blend now sees fp top1-top2 margin)\n"
        f"- unseen_para reading: **{unseen_reading}**; predicted relations "
        f"{json.dumps(sem_unseen.get('confusion', {}))}\n"
        f"- bank-wide top1 {sem_unseen['bank_top1']:.3f}, shuffled keys {sem_shuf['sel_acc']:.3f}\n"
        + (
            f"- matched GPT-2 unseen paraphrase: {gpt['unseen_para']['sel_acc']:.3f}\n"
            if gpt
            else "- matched GPT-2 control: not run\n"
        ),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))

    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save(
            {"W_q": W_q.state_dict(), "sem": semq.state_dict(), "stage": 258}, CKPT_OUT
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
