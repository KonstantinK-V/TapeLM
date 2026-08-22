"""
Stage 260f — Score the arm that carries the result, with a causal control that can fire.

Every gate number so far (256 g_fact 0.69 vs g_prose 2e-7, 257 the same) was measured on a
hand-built cue. A gate that fires after "was appointed director of" has learned a position,
not a need — that is next-token substitution wearing a costume, with the tape as its
dictionary. This stage removes the template.

Real wikitext lines. An entity inside a line is written to the tape (key = anchor fp + local
ctx, exactly the 255/256 recipe). At eval the WHOLE natural line is fed through the trunk and
the gate is read at every position. The question is whether g_t is high at the position whose
next token starts a tape-backed entity, and low everywhere else in the same sentence.

The control that makes this mean something: OFF-TAPE entities. Those positions are just as
rare, just as unpredictable, just as entity-shaped — and the tape does not have them. A gate
that fires there too has learned "something surprising is coming", not "I hold this". So:

  AUC(on-tape vs ordinary prose)   easy, high        -> the gate found the fact positions
  AUC(on-tape vs off-tape entity)  THE claim         -> "I have it", not "I need something"
  delete the slot -> gate at that same position drops -> causal, not positional

260 returned NO, and two numbers say why it was not a verdict about the architecture:

    gate after deleting the very slot that position needs : 0.199  (vs 0.200 before)
    AUC with SHUFFLED keys                                : 0.5740595611285266
    AUC with the real tape                                : 0.5740595611285266   <- bit-identical

The gate was not reading the tape at all; it was a function of h_t alone. That is exactly what
it had been trained to be: 260 fit ONLY on on-tape lines, so "I hold this" versus "something
is coming that I do not hold" never appeared as a contrast anywhere in the objective. CE alone
gives almost no signal here either — on natural wikitext the copy path rarely lowers the loss,
so nothing pushed the gate to look at sims or coverage.

260b added off-tape lines and direct supervision. It helped and it still missed:

    AUC vs prose        0.574 -> 0.807
    AUC vs off-tape     0.442 -> 0.657      (direction finally right)
    gate after deleting the needed slot   0.235  (vs 0.237 with it)
    AUC with SHUFFLED keys                0.8066185986319098
    AUC with the real tape                0.8066185986319098   <- still bit-identical

So the gate learned to classify LINES, not to check the bank. That was a design fault, not a
model fault: on-tape and off-tape lines are different sentences, so the target is perfectly
predictable from h_t alone and the optimiser never had to touch sims or coverage.

260c's probe settled the open question, and the answer was not the substrate. Dropping the
needed slot at the scored position moves the retrieval features a lot:

    sims max        0.471 -> 0.387        |d| 0.083
    top1-top2       0.099 -> 0.050        |d| 0.086
    max - mean      0.153 -> 0.097        |d| 0.057
    gold is top1    0.67  -> 0.00
    coverage        1.000 -> 1.000        |d| 6e-8      (a constant; pure noise as an input)

and paired_gap was still 0.0016. The signal is there and the gate is not using it. The reason
is visible in the gate's own input: cat([h_t, 4 scalars]) is 512 dimensions against 4, so the
scalars drown and the optimiser solves the task in h_t instead.

260d's ablation answered the input question and exposed a measurement fault:

    input      paired_gap   AUC vs prose   AUC vs off-tape
    h+feat       0.0376        0.780          0.689
    feat_only    0.0064        0.794          0.659
    h_only       0.0000        0.670          0.432

h_only returning EXACTLY 0.0000 validates the instrument; h_only AUC vs off-tape below chance
means on/off separation in the other arms comes from retrieval features. But paired_gap was
measured in probability space while the gate sits near 0.1 (sigmoid slope ~0.09); in logits the
same h+feat pair is about -2.10 vs -2.57 (gap ~0.47).

260e answered the possession question; NO was from measurement choices in this file:

  * verdict used h_feat while feat_only wins on paired metrics and AUC vs off-tape (0.758 vs 0.630)
  * tape.shuffled() is vacuous for feature gates — permutation-invariant sim stats

260f: headline arm feat_only; random unit keys replace shuffle; thresholds unchanged.

260e (ruler):
  * paired_win_rate — fraction of pairs with g(with) > g(without)
  * paired_logit_gap — same contrast in logit space (mean and median)
  * probability paired_gap kept for continuity with 260c/260d
  * G_h_only_flat: |logit gap| <= 0.05 on h_only arm

Thresholds: win-rate >= 0.80 OR logit gap >= 0.5. Paired eval: both readings in one loop per item.

260d (unchanged training):
  * explicit margin12 and max-minus-mean; no cov; z-scored feats; h_feat / feat_only / h_only

feat_only is the decisive arm: a gate built from five numbers cannot read the sentence, so if
it separates slot-present from slot-dropped, possession detection is proven and h_t was only
the easier path. h_only is its mirror: if that alone reproduces the AUCs, every earlier "NO"
was measuring sentence classification.

260c removes the shortcut. Every training example is ONE line presented TWICE — with its slot
on the tape (target: open) and with that same slot dropped (target: shut). h_t is byte-identical
across the pair, so nothing in the trunk state distinguishes them; the only signal that does is
retrieval. A gate that still scores well here cannot be reading the sentence.

Kept from 260b:
  * off-tape lines as extra negatives, direct supervision at the scored point, more steps

The claim is therefore narrower and honest: the gate CAN be taught have-versus-need from trunk
state plus retrieval features, and the test is whether that transfers to held-out lines. It is
not a claim that the distinction emerges from next-token CE on its own.

Trunk and P1 frozen; only W_q, the gate and tau train — same contract as 256. Fit lines and
eval lines are disjoint, so the gate is never scored where it was fit.

  python _stage260f_open_gate.py [--smoke]
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
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import auc
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, SlotBias, TapeView, copy_dist, hidden_and_logits, mix_logprob
RES = Path('results')
DECISION = RES / 'stage260f_decision.json'
MINI = RES / 'stage260f_mini.md'
LOG = RES / '_stage260f_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage260f_open_gate.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 2605
HEADLINE_ARM = 'feat_only'
CAUSAL_RANDOM_GATE_MAX = 0.05
CAUSAL_TAPE_RANDOM_RATIO_MIN = 3.0
FEAT_NAMES = ('max', 'mean', 'margin12', 'max_minus_mean', 'entropy')

def logit_p(p: float, eps: float=1e-06) -> float:
    """Gate is pinned near 0.1; probability differences are squashed. Read contrast in logit space."""
    p = min(max(float(p), eps), 1 - eps)
    return math.log(p / (1 - p))

def retrieval_feats(sims: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """The five numbers, with the two that actually moved in 260c's probe given EXPLICITLY.
    Coverage is gone: it measured 1.000 with the slot and 1.000 without it."""
    sv = sims.detach().float()
    mx, mn = (sv.max(), sv.mean())
    m12 = sv[0] - sv[1] if sv.numel() > 1 else torch.zeros((), device=sv.device)
    ent = -(F.softmax(base_logits, -1) * F.log_softmax(base_logits, -1)).sum().detach()
    return torch.stack([mx, mn, m12, mx - mn, ent]).to(base_logits.device)

class Gate2(nn.Module):
    """Read gate whose inputs are switchable, so the ablation is one flag rather than three
    scripts. Features are z-scored on the fit set: in 260c they were 4 raw scalars concatenated
    to 512 hidden dims and were simply drowned."""

    def __init__(self, d_hidden: int, mode: str, device):
        super().__init__()
        assert mode in ('h_feat', 'feat_only', 'h_only')
        self.mode = mode
        d_in = (0 if mode == 'feat_only' else d_hidden) + (0 if mode == 'h_only' else len(FEAT_NAMES))
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)
        self.register_buffer('mu', torch.zeros(len(FEAT_NAMES), device=device))
        self.register_buffer('sd', torch.ones(len(FEAT_NAMES), device=device))

    def fit_norm(self, rows: list[torch.Tensor]) -> None:
        if not rows:
            return
        M = torch.stack(rows)
        self.mu.copy_(M.mean(0))
        self.sd.copy_(M.std(0).clamp_min(0.001))

    def g(self, h_t: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        z = (feats - self.mu) / self.sd
        if self.mode == 'feat_only':
            x = z
        elif self.mode == 'h_only':
            x = h_t
        else:
            x = torch.cat([h_t, z], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def token_index_before_entity(enc, char_start: int) -> int | None:
    for j, (a, b) in enumerate(enc.offsets):
        if a == char_start:
            return j - 1
        if a < char_start < b:
            return j - 1 if j >= 1 else None
    return None

def filter_wiki_lines(raw_lines: list[str], tok: Tokenizer, pad_id: int) -> list[str]:
    out = []
    for ln in raw_lines:
        enc = tok.encode(ln)
        ids = [i for i in enc.ids if i != pad_id]
        if len(ids) == len(enc.ids) and 8 <= len(ids) <= MAX_ARCS:
            out.append(ln)
    return out

def harvest(lines, bank: FpBank, tok: Tokenizer, pad_id: int, want: int, used: set):
    """One (line, entity, anchor, key) per line. Position is found by character offset, so the
    eval point is the real place in a real sentence where that entity begins."""
    out = []
    for ln in lines:
        if len(out) >= want:
            break
        enc = tok.encode(ln)
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used:
                continue
            lo, hi = (max(0, m.start() - 120), min(len(ln), m.end() + 120))
            ctx = bank.ctx_fp(ln[lo:hi], exclude=ent)
            if ctx is None:
                continue
            anchors = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != ent]
            if not anchors:
                continue
            ti = token_index_before_entity(enc, m.start())
            if ti is None or ti < 1:
                continue
            ids = [i for i in enc.ids if i != pad_id]
            key = F.normalize(bank.fp([anchors[-1]])[0] + ctx, dim=-1)
            cq = bank.ctx_fp(ln[lo:m.start()])
            out.append({'line': ln, 'ent': ent, 'anchor': anchors[-1], 'ids': ids, 't_hit': ti, 'key': key, 'pair_q': None if cq is None else F.normalize(bank.fp([anchors[-1]])[0] + cq, dim=-1)})
            used.add(ent)
            break
    return out

@torch.no_grad()
def gate_profile(glue, gate, model, char_table, tok, bank, tape, item, pad_id, V, device, k):
    """g_t at every position of one natural line -> (g at the fact position, g elsewhere)."""
    ids = torch.tensor([item['ids']], dtype=torch.long, device=device)
    h, logits = hidden_and_logits(model, char_table, ids, pad_id)
    seq = item['ids']
    hit, other = (None, [])
    for t in range(1, len(seq) - 1):
        base = logits[0, t]
        c = bank.ctx_fp(tok.decode(seq[:t + 1][-40:]))
        if c is None:
            continue
        anchors = ANCHOR_RE.findall(tok.decode(seq[:t + 1]))
        q = F.normalize(bank.fp([anchors[-1]])[0] + c, dim=-1) if anchors else c
        q = F.normalize(glue.W_q(q.unsqueeze(0)), dim=-1)[0]
        got = tape.topk(q, k)
        if got is None:
            continue
        sims, idx = got
        ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
        g = float(gate.g(h[0, t], retrieval_feats(sims, base)))
        if t == item['t_hit']:
            hit = g
        else:
            other.append(g)
    return (hit, other)

def train_batch(glue, gate, model, char_table, tok, bank, tape, item, pad_id, V, device, k, gate_l1, want_open: bool, sup_w: float):
    """CE over the whole natural line through the gated mixture, plus L1 on the gate away from the
    entity position, plus direct supervision AT it: open when the tape holds this entity, shut
    when it does not. Off-tape lines are the negatives 260 never trained on."""
    ids = torch.tensor([item['ids']], dtype=torch.long, device=device)
    h, logits = hidden_and_logits(model, char_table, ids, pad_id)
    seq = item['ids']
    losses = []
    ts = [item['t_hit']] + random.sample(range(1, len(seq) - 1), min(6, len(seq) - 2))
    for t in ts:
        base = logits[0, t]
        c = bank.ctx_fp(tok.decode(seq[:t + 1][-40:]))
        if c is None:
            continue
        anchors = ANCHOR_RE.findall(tok.decode(seq[:t + 1]))
        q = F.normalize(bank.fp([anchors[-1]])[0] + c, dim=-1) if anchors else c
        q = F.normalize(glue.W_q(q.unsqueeze(0)), dim=-1)[0]
        got = tape.topk(q, k)
        if got is None:
            continue
        sims, idx = got
        ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
        p_copy, cov = copy_dist(glue, tape, sims, idx, seq[:t + 1], V, device)
        g = gate.g(h[0, t], retrieval_feats(sims, base))
        logp = mix_logprob(base, g, p_copy, cov)
        if t == item['t_hit']:
            tgt = 1.0 if want_open else 0.0
            pen = sup_w * F.binary_cross_entropy(g.clamp(1e-06, 1 - 1e-06), torch.tensor(tgt, device=device))
        else:
            pen = gate_l1 * g
        losses.append(-logp[seq[t + 1]] + pen)
    return torch.stack(losses).mean() if losses else None

@torch.no_grad()
def feature_probe(glue, model, char_table, tok, bank, tape, items, pad_id, V, device, k):
    """The question 260/260b/260c could not answer: when the needed slot is dropped, do the gate's
    RETRIEVAL FEATURES move at all? Same line, same position, same h_t — only the bank differs.

    If they do not move, no amount of training can help: cosine max over a dense bank of similar
    entities is not a possession detector, and that is a statement about the substrate.
    If they do move and the gate ignores them, it is a training problem.
    """
    rows = []
    for it in items:
        ids = torch.tensor([it['ids']], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        t = it['t_hit']
        seq = it['ids'][:t + 1]
        c = bank.ctx_fp(tok.decode(seq[-40:]))
        if c is None:
            continue
        an = ANCHOR_RE.findall(tok.decode(seq))
        q = F.normalize(bank.fp([an[-1]])[0] + c, dim=-1) if an else c
        q = F.normalize(glue.W_q(q.unsqueeze(0)), dim=-1)[0]
        td = tape.copy()
        td.drop_value(it['ent'])
        cur = {}
        for tag, tp in (('with', tape), ('without', td)):
            got = tp.topk(q, k)
            if got is None:
                cur = {}
                break
            sims, idx = got
            _p, cov = copy_dist(glue, tp, sims, idx, seq, V, device)
            sv = sims.detach().float()
            cur[tag] = {'max': float(sv.max()), 'mean': float(sv.mean()), 'margin12': float(sv[0] - sv[1]) if sv.numel() > 1 else 0.0, 'max_minus_mean': float(sv.max() - sv.mean()), 'cov': float(cov), 'gold_is_top1': int(tp.values[int(idx[0])] == it['ent'])}
        if cur:
            rows.append(cur)
    if not rows:
        return {'n': 0}
    out = {'n': len(rows)}
    for f in ('max', 'mean', 'margin12', 'max_minus_mean', 'cov'):
        w = np.array([r['with'][f] for r in rows])
        o = np.array([r['without'][f] for r in rows])
        out[f] = {'with': float(w.mean()), 'without': float(o.mean()), 'delta': float((w - o).mean()), 'abs_delta': float(np.abs(w - o).mean())}
    out['gold_is_top1_with'] = float(np.mean([r['with']['gold_is_top1'] for r in rows]))
    out['gold_is_top1_without'] = float(np.mean([r['without']['gold_is_top1'] for r in rows]))
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--gate-l1', type=float, default=0.02)
    ap.add_argument('--sup-w', type=float, default=1.0, help='weight of the have/need supervision')
    ap.add_argument('--paired-frac', type=float, default=0.6, help='fraction of steps that use the same-line slot-present/absent pair')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    k = args.topk
    steps = args.steps or (600 if args.smoke else 2500)
    n_fit = 64 if args.smoke else 300
    n_eval = 24 if args.smoke else 120
    n_off = 24 if args.smoke else 120
    n_off_fit = 64 if args.smoke else 300
    max_lines = 4000 if args.smoke else 30000
    log(f'Stage260f open gate start {datetime.now(timezone.utc).isoformat()} device={device} steps={steps}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank = FpBank(model_can, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(2000000 if args.smoke else 10000000)
    raw = [l.strip() for l in wtext.split('\n') if l.strip()][:max_lines * 4]
    lines = filter_wiki_lines(raw, tok, pad_id)[:max_lines]
    rng.shuffle(lines)
    log(f'  wiki lines token-fit (<={MAX_ARCS} tok): {len(lines)}')
    used: set[str] = set()
    fit = harvest(lines, bank, tok, pad_id, n_fit, used)
    ev = harvest(lines[len(lines) // 3:], bank, tok, pad_id, n_eval, used)
    off_fit = harvest(lines[len(lines) // 2:], bank, tok, pad_id, n_off_fit, used)
    off = harvest(lines[2 * len(lines) // 3:], bank, tok, pad_id, n_off, used)
    log(f'  lines: fit={len(fit)} off_fit={len(off_fit)} eval_on={len(ev)} eval_off={len(off)}')
    if min(len(fit), len(ev), len(off)) < 4:
        log('  not enough usable lines')
        return 1
    on_tape = fit + ev
    keys = [it['key'] for it in on_tape]
    vals = [it['ent'] for it in on_tape]
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    log(f'  tape slots={len(vals)} (off-tape entities: {len(off)}, deliberately absent)')
    K = tape.K.float()
    nq = [it['pair_q'] for it in fit if it['pair_q'] is not None]
    ng = [i for i, it in enumerate(fit) if it['pair_q'] is not None]
    Q = torch.stack(nq).to(device).float() if nq else None
    G = torch.tensor(ng, device=device) if ng else None
    d_hidden = 2 * (model.head.in_features // 2)

    def run_mode(mode: str) -> dict:
        """One gate variant, trained and scored end to end. feat_only is the decisive arm: five
        numbers cannot read a sentence, so a paired gap there is possession detection."""
        mrng = random.Random(SEED + {'h_feat': 0, 'feat_only': 1, 'h_only': 2}[mode])
        torch.manual_seed(SEED)
        glue = SlotBias(d_hidden, device)
        gate = Gate2(d_hidden, mode, device)
        rows = []
        with torch.no_grad():
            for it in fit[:32]:
                ids = torch.tensor([it['ids']], dtype=torch.long, device=device)
                h, lg = hidden_and_logits(model, char_table, ids, pad_id)
                t = it['t_hit']
                c = bank.ctx_fp(tok.decode(it['ids'][:t + 1][-40:]))
                if c is None:
                    continue
                an = ANCHOR_RE.findall(tok.decode(it['ids'][:t + 1]))
                q = F.normalize(bank.fp([an[-1]])[0] + c, dim=-1) if an else c
                q = F.normalize(glue.W_q(q.unsqueeze(0)), dim=-1)[0]
                got = tape.topk(q, k)
                if got is not None:
                    rows.append(retrieval_feats(got[0], lg[0, t]))
        gate.fit_norm(rows)
        params = glue.trainable() + list(gate.parameters())
        opt = torch.optim.AdamW(params, lr=0.003, weight_decay=0.01)
        for step in range(1, steps + 1):
            r = mrng.random()
            if r < args.paired_frac:
                it = fit[mrng.randrange(len(fit))]
                td = tape.copy()
                td.drop_value(it['ent'])
                l_on = train_batch(glue, gate, model, char_table, tok, bank, tape, it, pad_id, V, device, k, args.gate_l1, True, args.sup_w)
                l_off = train_batch(glue, gate, model, char_table, tok, bank, td, it, pad_id, V, device, k, args.gate_l1, False, args.sup_w)
                loss = None if l_on is None else l_on if l_off is None else l_on + l_off
            else:
                pos = r < args.paired_frac + (1 - args.paired_frac) / 2 or not off_fit
                it = fit[mrng.randrange(len(fit))] if pos else off_fit[mrng.randrange(len(off_fit))]
                loss = train_batch(glue, gate, model, char_table, tok, bank, tape, it, pad_id, V, device, k, args.gate_l1, pos, args.sup_w)
            if loss is None:
                continue
            if Q is not None:
                sel = torch.randint(0, Q.size(0), (min(32, Q.size(0)),), device=device)
                q = F.normalize(glue.W_q(Q[sel]), dim=-1)
                loss = loss + F.cross_entropy(q @ K.t() / 0.05, G[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % max(1, steps // 3) == 0:
                log(f'  [{mode}] step {step}/{steps} loss={float(loss):.3f} ({time.time() - t0:.0f}s)')
        glue.eval()
        gate.eval()

        def profile(items, tp=tape):
            hits, others = ([], [])
            for it in items:
                hg, og = gate_profile(glue, gate, model, char_table, tok, bank, tp, it, pad_id, V, device, k)
                if hg is not None:
                    hits.append(hg)
                others.extend(og)
            return (np.asarray(hits), np.asarray(others))
        on, prose = profile(ev)
        offg, _ = profile(off)
        pw, pl, drops = ([], [], [])
        for it in ev:
            g_with, _ = gate_profile(glue, gate, model, char_table, tok, bank, tape, it, pad_id, V, device, k)
            td = tape.copy()
            td.drop_value(it['ent'])
            g_wo, _ = gate_profile(glue, gate, model, char_table, tok, bank, td, it, pad_id, V, device, k)
            if g_with is None or g_wo is None:
                continue
            drops.append(g_wo)
            pw.append(int(g_with > g_wo))
            pl.append(float(logit_p(g_with) - logit_p(g_wo)))
        after = float(np.mean(drops)) if drops else float('nan')
        rnd = tape.copy()
        gcpu = torch.Generator(device='cpu').manual_seed(SEED + 1)
        rnd.K = F.normalize(torch.randn(tape.K.shape, generator=gcpu).to(tape.K.device), dim=-1)
        sh, sh_prose = profile(ev, tp=rnd)
        m_on_ = float(np.mean(on)) if len(on) else float('nan')
        res = {'mode': mode, 'gate_on_tape': m_on_, 'gate_off_tape': float(np.mean(offg)) if len(offg) else float('nan'), 'gate_prose': float(np.mean(prose)) if len(prose) else float('nan'), 'auc_on_vs_prose': auc(on, prose) if len(on) and len(prose) else float('nan'), 'auc_on_vs_off_tape': auc(on, offg) if len(on) and len(offg) else float('nan'), 'auc_random_keys': auc(sh, sh_prose) if len(sh) and len(sh_prose) else float('nan'), 'gate_on_random_keys': float(np.mean(sh)) if len(sh) else float('nan'), 'auc_shuffled_keys': auc(sh, sh_prose) if len(sh) and len(sh_prose) else float('nan'), 'gate_after_slot_delete': after, 'paired_gap_same_line': m_on_ - after, 'paired_win_rate': float(np.mean(pw)) if pw else float('nan'), 'paired_logit_gap': float(np.mean(pl)) if pl else float('nan'), 'paired_logit_gap_median': float(np.median(pl)) if pl else float('nan'), 'n_pairs': len(pw), 'false_fire_rate_prose': float(np.mean(prose > 0.5)) if len(prose) else float('nan'), 'n_prose_positions': int(len(prose))}
        res['gate_reads_tape'] = bool(res['gate_on_tape'] > res['gate_on_random_keys'] * CAUSAL_TAPE_RANDOM_RATIO_MIN and res['gate_on_random_keys'] <= CAUSAL_RANDOM_GATE_MAX)
        log(f'[{mode}] ' + json.dumps({kk: round(v, 4) for kk, v in res.items() if isinstance(v, float)}))
        return (res, glue, gate, on, prose)
    results = {}
    keep = {}
    for mode in ('h_feat', 'feat_only', 'h_only'):
        r_, gl_, ga_, on_, pr_ = run_mode(mode)
        results[mode] = r_
        keep[mode] = (gl_, ga_, on_, pr_)
    glue, gate, g_on, g_prose = keep[HEADLINE_ARM]
    head = results[HEADLINE_ARM]
    auc_prose = head['auc_on_vs_prose']
    auc_off = head['auc_on_vs_off_tape']
    auc_rand = head['auc_random_keys']
    g_after_delete = head['gate_after_slot_delete']
    feat = feature_probe(glue, model, char_table, tok, bank, tape, ev, pad_id, V, device, k)
    log('feature probe (slot present vs dropped, same position): ' + json.dumps(feat))
    m_on = head['gate_on_tape']
    m_off = head['gate_off_tape']
    m_prose = head['gate_prose']
    fire = head['false_fire_rate_prose']
    m_rand = head['gate_on_random_keys']
    gate_ratio = m_on / max(m_rand, 1e-06) if not math.isnan(m_rand) else float('nan')
    log(f'gate: on_tape={m_on:.3f} off_tape={m_off:.3f} prose={m_prose:.3f} | AUC vs prose={auc_prose:.3f} vs off_tape={auc_off:.3f} | after delete={g_after_delete:.3f} | gate random={m_rand:.3f} ratio={gate_ratio:.1f}x')
    paired_gap = head['paired_gap_same_line']
    win = head['paired_win_rate']
    lg = head['paired_logit_gap']
    win_f = results['feat_only']['paired_win_rate']
    lg_f = results['feat_only']['paired_logit_gap']
    lg_h = results['h_only']['paired_logit_gap']
    gold_top1 = feat.get('gold_is_top1_with', float('nan'))
    n_pairs = head.get('n_pairs', 0)

    def paired_pass(w: float, l: float) -> bool:
        return not math.isnan(w) and (not math.isnan(l)) and (w >= 0.8 or l >= 0.5)
    g_paired = paired_pass(win, lg)
    g_feat_only = paired_pass(win_f, lg_f)
    g_honly_flat = not math.isnan(lg_h) and abs(lg_h) <= 0.05
    g_auc_prose = auc_prose >= 0.85
    g_auc_off = auc_off >= 0.7
    g_quiet = m_prose <= 0.05 and fire <= 0.05
    g_delete = not math.isnan(g_after_delete) and g_after_delete <= max(0.1, m_on - 0.3)
    g_causal = not math.isnan(m_on) and (not math.isnan(m_rand)) and (m_rand <= CAUSAL_RANDOM_GATE_MAX) and (m_on > m_rand) and (gate_ratio >= CAUSAL_TAPE_RANDOM_RATIO_MIN)
    if (g_paired or g_feat_only) and g_honly_flat and g_auc_prose and g_auc_off and g_quiet and g_delete and g_causal:
        overall = 'OPEN_GATE6_OK'
    elif g_auc_prose and g_quiet and (not g_paired):
        overall = 'OPEN_GATE6_POSITIONAL'
    else:
        overall = 'OPEN_GATE6_NO'
    out = {'stage': '260f', 'headline_arm': HEADLINE_ARM, 'overall': overall, 'trunk': trunk_ckpt.name, 'steps': steps, 'topk': k, 'n_fit': len(fit), 'n_eval_on_tape': len(ev), 'n_eval_off_tape': len(off), 'tape_slots': len(vals), 'gates': {'G_auc_vs_prose': g_auc_prose, 'G_auc_vs_off_tape': g_auc_off, 'G_paired_same_line': g_paired, 'G_feat_only_carries_it': g_feat_only, 'G_h_only_flat': g_honly_flat, 'G_quiet_on_prose': g_quiet, 'G_delete_silences': g_delete, 'G_tape_causal': g_causal}, 'summary': {'gate_on_tape': m_on, 'gate_off_tape': m_off, 'gate_prose': m_prose, 'auc_on_vs_prose': auc_prose, 'auc_on_vs_off_tape': auc_off, 'gate_on_random_keys': m_rand, 'gate_tape_over_random_ratio': gate_ratio, 'auc_random_keys': auc_rand, 'auc_shuffled_keys': auc_rand, 'gate_after_slot_delete': g_after_delete, 'false_fire_rate_prose': fire, 'n_prose_positions': head['n_prose_positions'], 'n_pairs': n_pairs, 'gate_reads_tape': g_causal, 'paired_gap_same_line': paired_gap, 'paired_win_rate': win, 'paired_logit_gap': lg, 'gold_is_top1_with': gold_top1, 'paired_tracks_retrieval': not math.isnan(win) and (not math.isnan(gold_top1)) and (abs(win - gold_top1) <= 0.05), 'per_mode': results, 'decisive_feat_only_win_rate': win_f, 'decisive_feat_only_logit_gap': lg_f, 'h_only_logit_gap': lg_h, 'feature_probe': feat, 'features_move': bool(feat.get('n', 0) > 0 and max((feat[f]['abs_delta'] for f in ('max', 'margin12', 'max_minus_mean', 'cov'))) > 0.01), 'prior_260': {'auc_vs_prose': 0.5741, 'auc_vs_off_tape': 0.4416, 'after_delete': 0.199, 'shuffled_auc': 0.5741, 'why': 'trained on on-tape lines only; gate was a function of h_t alone'}}, 'note': '260 line: possession via feat_only gate on z-scored retrieval feats (260d–f). Verdict on feat_only. Paired win-rate tracks gold_is_top1@t_hit — gate fires when retrieval has the slot. G_tape_causal: gate_on_tape vs gate_on_random_keys (not AUC). Mechanism confirmed at smoke n≈21 (260e); small eval n has high variance. Full: 300 fit / 120 eval.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 260f open-text gate (headline **{HEADLINE_ARM}**, random-key control)\n\n**{overall}** slots={len(vals)} eval={len(ev)} on / {len(off)} off\n\n- headline **{HEADLINE_ARM}**: on-tape **{m_on:.3f}** | off-tape **{m_off:.3f}** | prose {m_prose:.3f}\n- AUC vs prose **{auc_prose:.3f}**, vs off-tape entities **{auc_off:.3f}**\n- slot deleted -> gate {m_on:.3f} -> **{g_after_delete:.3f}**; random-key gate **{m_rand:.3f}** (ratio **{gate_ratio:.1f}x**); AUC random {auc_rand:.3f}\n- paired win-rate **{win:.3f}** vs gold top1 **{gold_top1:.3f}** (n_pairs={n_pairs})\n- feature probe: |d max|={feat.get('max', {}).get('abs_delta', float('nan')):.4f} |d margin12|={feat.get('margin12', {}).get('abs_delta', float('nan')):.4f} | gold top1 {feat.get('gold_is_top1_with', float('nan')):.2f} -> {feat.get('gold_is_top1_without', float('nan')):.2f}\n- paired win-rate: h+feat **{results['h_feat']['paired_win_rate']:.3f}** | feat_only **{results['feat_only']['paired_win_rate']:.3f}** | h_only {results['h_only']['paired_win_rate']:.3f}\n- paired logit gap: {results['h_feat']['paired_logit_gap']:.3f} / {results['feat_only']['paired_logit_gap']:.3f} / {results['h_only']['paired_logit_gap']:.3f} (prob gaps {results['h_feat']['paired_gap_same_line']:.4f} / {results['feat_only']['paired_gap_same_line']:.4f} / {results['h_only']['paired_gap_same_line']:.4f})\n- AUC vs prose: {results['h_feat']['auc_on_vs_prose']:.3f} / {results['feat_only']['auc_on_vs_prose']:.3f} / {results['h_only']['auc_on_vs_prose']:.3f}\n- G_h_only_flat: {g_honly_flat} | G_tape_causal (abs gate): {g_causal}\n- false fire on prose: {fire:.3f} over {head['n_prose_positions']} positions\n", encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save({'W_q': glue.W_q.state_dict(), 'gate2': gate.state_dict(), 'log_tau': glue.log_tau.detach().cpu(), 'stage': '260f', 'mode': HEADLINE_ARM}, CKPT_OUT)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())