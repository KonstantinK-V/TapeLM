"""
Stage 260c — Open-text gate, with the ONE contrast h_t cannot solve.

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

  python _stage260c_open_gate.py [--smoke]

After eval, feature_probe logs retrieval features (max, margin12, cov, …) with vs without the
needed slot at t_hit; summary.features_move reads the verdict (|delta| > 0.01 on key feats).
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
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import auc
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, SlotBias, TapeView, copy_dist, hidden_and_logits, mix_logprob
RES = Path('results')
DECISION = RES / 'stage260c_decision.json'
MINI = RES / 'stage260c_mini.md'
LOG = RES / '_stage260c_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage260c_open_gate.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 2602

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
def gate_profile(glue, model, char_table, tok, bank, tape, item, pad_id, V, device, k):
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
        _p, cov = copy_dist(glue, tape, sims, idx, seq[:t + 1], V, device)
        g = float(glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov))
        if t == item['t_hit']:
            hit = g
        else:
            other.append(g)
    return (hit, other)

def train_batch(glue, model, char_table, tok, bank, tape, item, pad_id, V, device, k, gate_l1, want_open: bool, sup_w: float):
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
        g = glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov)
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
    """Same line, same t_hit, same h_t — only the bank differs (slot present vs dropped)."""
    rows = []
    for it in items:
        ids = torch.tensor([it['ids']], dtype=torch.long, device=device)
        _h, _logits = hidden_and_logits(model, char_table, ids, pad_id)
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

def probe_features_move(feat: dict, *, thresh: float=0.01) -> bool:
    if feat.get('n', 0) <= 0:
        return False
    try:
        return max((feat[f]['abs_delta'] for f in ('max', 'margin12', 'max_minus_mean', 'cov'))) > thresh
    except KeyError:
        return False

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
    log(f'Stage260c open gate start {datetime.now(timezone.utc).isoformat()} device={device} steps={steps} paired_frac={args.paired_frac}')
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
    glue = SlotBias(2 * (model.head.in_features // 2), device)
    opt = torch.optim.AdamW(glue.trainable(), lr=0.003, weight_decay=0.01)
    K = tape.K.float()
    nq = [it['pair_q'] for it in fit if it['pair_q'] is not None]
    ng = [i for i, it in enumerate(fit) if it['pair_q'] is not None]
    Q = torch.stack(nq).to(device).float() if nq else None
    G = torch.tensor(ng, device=device) if ng else None
    for step in range(1, steps + 1):
        r = rng.random()
        if r < args.paired_frac:
            it = fit[rng.randrange(len(fit))]
            td = tape.copy()
            td.drop_value(it['ent'])
            l_on = train_batch(glue, model, char_table, tok, bank, tape, it, pad_id, V, device, k, args.gate_l1, True, args.sup_w)
            l_off = train_batch(glue, model, char_table, tok, bank, td, it, pad_id, V, device, k, args.gate_l1, False, args.sup_w)
            loss = None if l_on is None else l_on if l_off is None else l_on + l_off
        else:
            pos = r < args.paired_frac + (1 - args.paired_frac) / 2 or not off_fit
            it = fit[rng.randrange(len(fit))] if pos else off_fit[rng.randrange(len(off_fit))]
            loss = train_batch(glue, model, char_table, tok, bank, tape, it, pad_id, V, device, k, args.gate_l1, pos, args.sup_w)
        if loss is None:
            continue
        if Q is not None:
            sel = torch.randint(0, Q.size(0), (min(32, Q.size(0)),), device=device)
            q = F.normalize(glue.W_q(Q[sel]), dim=-1)
            loss = loss + F.cross_entropy(q @ K.t() / 0.05, G[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(glue.trainable(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0:
            log(f'  step {step}/{steps} loss={float(loss):.3f} ({time.time() - t0:.0f}s)')
    glue.eval()

    def profile(items, tp=tape):
        hits, others = ([], [])
        for it in items:
            hg, og = gate_profile(glue, model, char_table, tok, bank, tp, it, pad_id, V, device, k)
            if hg is not None:
                hits.append(hg)
            others.extend(og)
        return (np.asarray(hits), np.asarray(others))
    g_on, g_prose = profile(ev)
    g_off, _ = profile(off)
    auc_prose = auc(g_on, g_prose) if len(g_on) and len(g_prose) else float('nan')
    auc_off = auc(g_on, g_off) if len(g_on) and len(g_off) else float('nan')
    drops = []
    for it in ev:
        td = tape.copy()
        td.drop_value(it['ent'])
        hg, _ = gate_profile(glue, model, char_table, tok, bank, td, it, pad_id, V, device, k)
        if hg is not None:
            drops.append(hg)
    g_after_delete = float(np.mean(drops)) if drops else float('nan')
    feat = feature_probe(glue, model, char_table, tok, bank, tape, ev, pad_id, V, device, k)
    features_move = probe_features_move(feat)
    log('feature probe (slot present vs dropped, same position): ' + json.dumps(feat))
    g_shuf, prose_shuf = profile(ev, tp=tape.shuffled(SEED + 1))
    auc_shuf = auc(g_shuf, prose_shuf) if len(g_shuf) and len(prose_shuf) else float('nan')
    m_on = float(np.mean(g_on)) if len(g_on) else float('nan')
    m_off = float(np.mean(g_off)) if len(g_off) else float('nan')
    m_prose = float(np.mean(g_prose)) if len(g_prose) else float('nan')
    fire = float(np.mean(g_prose > 0.5)) if len(g_prose) else float('nan')
    log(f'gate: on_tape={m_on:.3f} off_tape={m_off:.3f} prose={m_prose:.3f} | AUC vs prose={auc_prose:.3f} vs off_tape={auc_off:.3f} | after delete={g_after_delete:.3f} | shuffled AUC={auc_shuf:.3f}')
    paired_gap = m_on - g_after_delete
    gate_reads_tape = bool(abs(auc_prose - auc_shuf) > 1e-06)
    g_paired = not math.isnan(g_after_delete) and paired_gap >= 0.15
    g_auc_prose = auc_prose >= 0.85
    g_auc_off = auc_off >= 0.7
    g_quiet = m_prose <= 0.05 and fire <= 0.05
    g_delete = not math.isnan(g_after_delete) and g_after_delete <= max(0.1, m_on - 0.3)
    g_causal = not math.isnan(auc_shuf) and auc_shuf <= 0.65
    if g_paired and g_auc_prose and g_auc_off and g_quiet and g_delete and g_causal:
        overall = 'OPEN_GATE3_OK'
    elif g_auc_prose and g_quiet and (not g_paired or not gate_reads_tape):
        overall = 'OPEN_GATE3_POSITIONAL'
    else:
        overall = 'OPEN_GATE3_NO'

    def _prior(path: Path, defaults: dict) -> dict:
        if not path.exists():
            return defaults
        try:
            d = json.loads(path.read_text(encoding='utf-8'))
            s = d.get('summary') or {}
            return {'overall': d.get('overall'), 'auc_vs_prose': s.get('auc_on_vs_prose', defaults.get('auc_vs_prose')), 'auc_vs_off_tape': s.get('auc_on_vs_off_tape', defaults.get('auc_vs_off_tape')), 'after_delete': s.get('gate_after_slot_delete', defaults.get('after_delete')), 'shuffled_auc': s.get('auc_shuffled_keys', defaults.get('shuffled_auc')), 'paired_gap': s.get('paired_gap_same_line', s.get('paired_gap_same_line')), 'gate_reads_tape': s.get('gate_reads_tape')}
        except (json.JSONDecodeError, OSError):
            return defaults
    prior_260 = _prior(RES / 'stage260_decision.json', {'auc_vs_prose': 0.5741, 'auc_vs_off_tape': 0.4416, 'after_delete': 0.199, 'shuffled_auc': 0.5741})
    prior_260b = _prior(RES / 'stage260b_decision.json', {'auc_vs_prose': 0.807, 'auc_vs_off_tape': 0.657, 'after_delete': 0.235, 'shuffled_auc': 0.8066, 'paired_gap': 0.002, 'gate_reads_tape': False})
    out = {'stage': '260c', 'overall': overall, 'trunk': trunk_ckpt.name, 'steps': steps, 'topk': k, 'paired_frac': args.paired_frac, 'n_fit': len(fit), 'n_eval_on_tape': len(ev), 'n_eval_off_tape': len(off), 'tape_slots': len(vals), 'gates': {'G_auc_vs_prose': g_auc_prose, 'G_auc_vs_off_tape': g_auc_off, 'G_paired_same_line': g_paired, 'G_quiet_on_prose': g_quiet, 'G_delete_silences': g_delete, 'G_tape_causal': g_causal}, 'summary': {'gate_on_tape': m_on, 'gate_off_tape': m_off, 'gate_prose': m_prose, 'auc_on_vs_prose': auc_prose, 'auc_on_vs_off_tape': auc_off, 'auc_shuffled_keys': auc_shuf, 'gate_after_slot_delete': g_after_delete, 'false_fire_rate_prose': fire, 'n_prose_positions': int(len(g_prose)), 'gate_reads_tape': gate_reads_tape, 'paired_gap_same_line': paired_gap, 'feature_probe': feat, 'features_move': features_move, 'prior_260': prior_260, 'prior_260b': prior_260b}, 'note': '260c: paired same-line train (h_t fixed); feature_probe at eval t_hit with vs without the needed slot. features_move=false => cosine features do not detect possession (substrate limit; bears on 212b). features_move=true but paired_gap~0 => train the gate. gold_is_top1_with low => retrieval misses gold even with slot present — rebuild exam. G_paired_same_line ≥ 0.15; gate_reads_tape = real vs shuffled AUC differ.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 260c open-text gate (paired same-line contrast)\n\n**{overall}** slots={len(vals)} eval={len(ev)} on / {len(off)} off\n\n- gate: on-tape **{m_on:.3f}** | off-tape **{m_off:.3f}** | prose {m_prose:.3f}\n- AUC vs prose **{auc_prose:.3f}**, vs off-tape entities **{auc_off:.3f}**\n- slot deleted -> gate {m_on:.3f} -> **{g_after_delete:.3f}**; shuffled keys AUC {auc_shuf:.3f}\n- paired gap (on − delete) **{paired_gap:.4f}** | gate_reads_tape={gate_reads_tape}\n- **features_move={features_move}** | probe |d max|={feat.get('max', {}).get('abs_delta', float('nan')):.4f} |d margin12|={feat.get('margin12', {}).get('abs_delta', float('nan')):.4f} |d cov|={feat.get('cov', {}).get('abs_delta', float('nan')):.4f} | gold top1 {feat.get('gold_is_top1_with', float('nan')):.2f} → {feat.get('gold_is_top1_without', float('nan')):.2f}\n- false fire on prose: {fire:.3f} over {len(g_prose)} positions\n", encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save({'W_q': glue.W_q.state_dict(), 'gate': glue.gate.state_dict(), 'log_tau': glue.log_tau.detach().cpu(), 'stage': 260}, CKPT_OUT)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())