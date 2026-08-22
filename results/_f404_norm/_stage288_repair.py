"""
Stage 288 — Repair: the mind learns to fix the tape, scored against ground truth it cannot see.

286 taught reading agreement; this teaches RESTORING it. The loop is: break a clean address,
show the mind the broken evidence, require (1) WHERE the break is and (2) WHAT stood there -
or the admission that the remaining evidence cannot say. The proposal's original reward -
"did the world become more consistent by the judges" - is refused as a training signal,
because its fixed point is an empty tape: erase every dispute and the judges purr. It is kept
as an OBSERVER (verdict_restored_rate), which is where signals that cannot be trusted with
gradients live in this project.

What replaces it is better than a judge: the corruption is synthetic, so the truth is known
for free. We broke it, we know what it was. That gives three things 286 could not have:
  - unlimited examples: corruption is generated, not harvested, so the diagnosed bottleneck
    (about a thousand distinct examples, 22 usable addresses per tape) disappears;
  - honestly-trained honesty: we KNOW which breaks are unrecoverable, because we made them;
  - the write side becomes learnable: WRITE/CONFIRM/DISPUTE is code today; a mind that can
    point at the forged mention is the mind that can one day dispute it.

Cloze was a special case all along: hiding a mention is the DELETE corruption, the lying tape
is REPLACE, --lie-dup is REPLACE with copies. 286's exams were points in this family; 288
trains on the family. DELETE itself stays in 286 (it is measured there); here the ops are
NONE / REPLACE / DUP, because they are the ones with a visible culprit.

The mind is the relational one - the only arm that ever passed anything - with two heads on
one shared graph embedding: a DIAGNOSIS head scoring every mention plus a CLEAN row (the same
shape as candidates plus UNKNOWN), and a REPAIR head scoring the surviving values plus
UNKNOWN. Identity stays unrepresentable: ranks and indicators only. The answer stays an index.

Falsifiers, all relative, none tunable:
  G_detects_forgery       beat the counting detector (flag the minority value) AND the
                          per-example random floor, held out
  G_detects_dup           same, on the subset where the forged value is the MAJORITY -
                          the counting detector is wrong there by construction
  G_flags_clean           the CLEAN margin separates untouched addresses from forged ones
                          (AUC above its own noise) - false alarms are the failure mode of
                          every repair system
  G_repairs               with the true flag given, restore the original value better than
                          majority-of-the-rest does
  G_honest_unrecoverable  UNKNOWN's margin ranks unrecoverable breaks above recoverable ones

  python _stage288_repair.py --smoke
  python _stage288_repair.py --train-steps 6000
  python _stage288_repair.py --train-steps 6000 --holdout address
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage286_evidence as s286
import _stage279_write_decision as s279
from _tape_speed import CachedBank, install_all
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 288
OPS = ('none', 'replace', 'dup')
LOG_PATH = RES / '_stage288_log.txt'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line)

def corrupt(pack, item, rng, op: str, pos_slot: int | None):
    """One broken address, with the truth attached - because we are the ones who broke it.

    none      the address as written. The target diagnosis is CLEAN, and its being a real
              class is what keeps the detector honest: a mind that flags everything scores
              zero here, no judge needed to say so.
    replace   one genuine mention swapped for a foreign one (another subject's value and
              text). The culprit is visible and the original value is recorded.
    dup       the same swap, but the forged mention arrives in 2-3 copies - the regime where
              counting is wrong BY CONSTRUCTION, since the forgery is now the majority.
    """
    vals, texts = (pack['tape'].values, pack['texts'])
    slots = list(item['slots'])
    if op == 'none':
        ev_slots, forged, orig = (slots, set(), None)
    else:
        j = slots.index(pos_slot)
        k = 1 if op == 'replace' else rng.choice((2, 3))
        src = None
        for _ in range(64):
            c = rng.randrange(pack['n_slots'])
            if c not in set(slots) and vals[c] != vals[pos_slot]:
                src = c
                break
        if src is None:
            return None
        ev_slots = slots[:j] + [src] * k + slots[j + 1:]
        forged = set(range(j, j + k))
        orig = vals[pos_slot]
    ev_vals = [vals[s] for s in ev_slots]
    if op != 'none' and len(set(ev_vals)) < 2:
        return None
    return {'op': op, 'slots': ev_slots, 'vals': ev_vals, 'texts': [texts[s] for s in ev_slots], 'forged': forged, 'orig': orig}

def repair_candidates(ev, flagged: set[int]):
    """What survives once the flagged mentions are set aside, plus UNKNOWN."""
    keep = [v for i, v in enumerate(ev['vals']) if i not in flagged]
    cands = [v for v, _ in Counter(keep).most_common(8)]
    target = cands.index(ev['orig']) if ev['orig'] in cands else len(cands)
    return (cands, target, keep)

def votes_detector(ev):
    """Flag a mention of the minority value; CLEAN when unanimous. The counter's best."""
    cnt = Counter(ev['vals'])
    if len(cnt) == 1:
        return None
    minority = min(cnt, key=lambda v: cnt[v])
    return ev['vals'].index(minority)

def votes_repair(ev, flagged: set[int]):
    keep = [v for i, v in enumerate(ev['vals']) if i not in flagged]
    cnt = Counter(keep)
    top = cnt.most_common(2)
    if not top or (len(top) > 1 and top[0][1] == top[1][1]):
        return None
    return top[0][0]

class RepairMind(nn.Module):
    """One relational embedding, two heads. Identity cannot enter: ranks and indicators only.

    The diagnosis head scores each MENTION (plus a CLEAN row from the global pool) - the same
    shape as 286's candidates-plus-UNKNOWN, pointed at slots instead of values. The repair
    head is 286's candidate head. Sharing the embedding is the claim that the same relations
    answer both questions: a forged mention sits in the graph as the value nobody's context
    agrees with, and the repaired value is the one whose coalition is tightest without it.
    """

    def __init__(self, device, d: int=32):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(3, d), nn.GELU()).to(device)
        self.node = nn.Sequential(nn.Linear(2 + 2 * d, d), nn.GELU()).to(device)
        self.diag = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        self.rep = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        for head in (self.diag, self.rep):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def embed(self, E, same, nf):
        e = self.edge(E)
        own = (e * same).sum(1) / same.sum(1).clamp(min=1.0)
        h = self.node(torch.cat([nf, own, e.mean(1)], -1))
        return (h, h.mean(0))

    def diagnose(self, h, g):
        z, o = (torch.zeros(1, device=g.device), torch.ones(1, device=g.device))
        outs = [self.diag(torch.cat([h[i], g, z])) for i in range(h.shape[0])]
        outs.append(self.diag(torch.cat([torch.zeros_like(g), g, o])))
        return torch.cat(outs)

    def repair(self, h, g, masks):
        z, o = (torch.zeros(1, device=g.device), torch.ones(1, device=g.device))
        outs = [self.rep(torch.cat([h[m].mean(0), g, z])) for m in masks]
        outs.append(self.rep(torch.cat([torch.zeros_like(g), g, o])))
        return torch.cat(outs)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train-steps', type=int, default=0)
    ap.add_argument('--tape-period', type=int, default=0)
    ap.add_argument('--addresses', type=int, default=0)
    ap.add_argument('--min-mentions', type=int, default=2)
    ap.add_argument('--address-tau', type=float, default=0.9)
    ap.add_argument('--address-overlap', type=int, default=2)
    ap.add_argument('--addr-key', choices=('two', 'set', 'mean'), default='two')
    ap.add_argument('--abstain-reward', type=float, default=0.75)
    ap.add_argument('--wrong-cost', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--holdout', choices=('corpus', 'address'), default='corpus', help='corpus: eval tape from the disjoint 30% of lines. address: one corpus, subjects split by a stable hash of the anchor - 286 measured honesty crossing that split (0.754) while failing the corpus one, so both views matter and neither is redundant.')
    ap.add_argument('--run-tag', type=str, default='')
    ap.add_argument('--no-speedups', action='store_true', help='run the original unmemoised paths - they are byte-identical, and this is how that stays checkable')
    args = ap.parse_args()
    global LOG_PATH
    tag = args.run_tag and f'_{args.run_tag}' or ''
    tag += '_addrholdout' if args.holdout == 'address' else ''
    tag = '_smoke' + tag if args.smoke else tag
    LOG_PATH = RES / f'_stage288_log{tag}.txt'
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_steps = args.train_steps or (600 if args.smoke else 6000)
    tape_period = args.tape_period or (50 if args.smoke else 50)
    n_addr = args.addresses or (300 if args.smoke else 400)
    log(f'Stage288 repair start {datetime.now(timezone.utc).isoformat()} device={device} steps={n_steps} holdout={args.holdout}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    if not args.no_speedups:
        install_all(s279)
    bank = FpBank(can, stoi, device) if args.no_speedups else CachedBank(FpBank(can, stoi, device))
    arc0 = s271.arc_enc_hash(can)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(4000000 if args.smoke else 30000000)
    all_lines = [l.strip() for l in wtext.split('\n') if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][:3000 if args.smoke else 25000]
    eval_lines = all_lines[cut:][:1500 if args.smoke else 12000]
    if args.holdout == 'address':
        eval_lines = train_lines

    def anchor_side(address: str) -> int:
        anchor = address.split(':', 1)[-1].split('|')[0]
        return int(hashlib.sha1(anchor.encode('utf-8')).hexdigest(), 16) & 1

    def split_items(p, side):
        if args.holdout == 'corpus':
            return p
        p = dict(p)
        p['items'] = [it for it in p['items'] if anchor_side(it['address']) == side]
        return p

    def new_pack(r, lines, side):
        return split_items(s280.pack_from_corpus(lines, bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, n_addr=n_addr, min_mentions=args.min_mentions, tau=args.address_tau, overlap=args.address_overlap, soft_match=0.0, min_per_family=8, addr_key=args.addr_key), side)

    def graph_inputs(p, ev, item):
        slots, vals_e = (ev['slots'], ev['vals'])
        n = len(slots)
        ck, ws = (p.setdefault('_ctx', {}), p.setdefault('_words', {}))
        for sl in set(slots):
            if sl not in ck:
                c = bank.ctx_fp(p['texts'][sl], exclude=p['tape'].values[sl])
                ck[sl] = F.normalize(c, dim=-1) if c is not None else None
                ws[sl] = set(context_words(p['texts'][sl], exclude=p['tape'].values[sl]))
        med = p.get('_median')
        if med is None:
            lens = sorted((len(v) for v in p['postings'].values()))
            med = lens[len(lens) // 2] if lens else 1
            p['_median'] = med
        same = torch.zeros(n, n)
        cos = torch.zeros(n, n)
        shared = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = (slots[i], slots[j])
                same[i, j] = same[j, i] = float(vals_e[i] == vals_e[j])
                if ck[si] is not None and ck[sj] is not None:
                    cos[i, j] = cos[j, i] = float(ck[si] @ ck[sj])
                rare = sum((1 for w in ws[si] & ws[sj] if len(p['postings'].get(w, ())) < med))
                shared[i, j] = shared[j, i] = rare / max(1, min(len(ws[si]), len(ws[sj])))
        iu = torch.triu_indices(n, n, offset=1)

        def rank_norm(M):
            if iu.numel() == 0:
                return M
            v = M[iu[0], iu[1]]
            order = v.argsort()
            r = torch.empty_like(order, dtype=torch.float32)
            r[order] = torch.arange(len(v), dtype=torch.float32)
            uniq, inv = v.unique(return_inverse=True)
            if len(uniq) > 1:
                mean_r = torch.zeros(len(uniq)).index_reduce_(0, inv, r, 'mean', include_self=False)
                r = mean_r[inv] / (len(v) - 1 if len(v) > 1 else 1)
            else:
                r = torch.zeros_like(r)
            R = torch.zeros_like(M)
            R[iu[0], iu[1]] = r
            R[iu[1], iu[0]] = r
            return R
        E = torch.stack([same, rank_norm(cos), rank_norm(shared)], -1).to(device)
        cnt = Counter(vals_e)
        own = set(item['slots'])
        ext = {v: s286.ext_support(p, item['S'], v, own) for v in cnt}
        levels = sorted(set(ext.values()))
        nf = torch.tensor([[cnt[vals_e[i]] / n, levels.index(ext[vals_e[i]]) / max(1, len(levels) - 1)] for i in range(n)], dtype=torch.float32, device=device)
        return (E, same.unsqueeze(-1).to(device), nf)
    net = RepairMind(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = int(sum((q.numel() for q in net.parameters())))

    def usable(p):
        return [(it, s) for it in p['items'] if len(it['slots']) >= 2 for s in it['slots']]
    pack = new_pack(rng, train_lines, 0)
    pairs = usable(pack)
    log(f"  tape: {pack['n_addresses']} addresses, {pack['n_slots']} slots, {len(pairs)} corruption sites, params={n_params}")
    if len(pairs) < 4 * s286.MIN_ANSWERED:
        sizes = Counter((len(it['slots']) for it in pack['items']))
        log(f"  too few corruption sites: {len(pairs)} < {4 * s286.MIN_ANSWERED}. items {len(pack['items'])}, mentions-per-address {json.dumps(dict(sizes))}. Raise --addresses or lower --min-mentions.")
        return 1
    losses, curve, op_seen = ([], [], Counter())
    for step in range(1, n_steps + 1):
        if (step - 1) % tape_period == 0 and step > 1:
            pack = new_pack(rng, train_lines, 0)
            pairs = usable(pack)
            if not pairs:
                log('  empty tape after resample')
                return 1
        it, sl = pairs[rng.randrange(len(pairs))]
        op = OPS[rng.randrange(len(OPS))]
        ev = corrupt(pack, it, rng, op, sl)
        if ev is None:
            continue
        op_seen[op] += 1
        E, same, nf = graph_inputs(pack, ev, it)
        h, g = net.embed(E, same, nf)
        dg = net.diagnose(h, g)
        n = len(ev['vals'])
        if ev['forged']:
            lp = F.log_softmax(dg, dim=-1)
            d_loss = -torch.logsumexp(lp[list(ev['forged'])], dim=0)
        else:
            d_loss = F.cross_entropy(dg.unsqueeze(0), torch.tensor([n], device=device))
        loss = d_loss
        if ev['forged']:
            cands, target, keep = repair_candidates(ev, ev['forged'])
            if len(cands) >= 2:
                masks = [torch.tensor([i not in ev['forged'] and ev['vals'][i] == c for i in range(n)], device=device) for c in cands]
                rp = net.repair(h, g, masks)
                loss = loss + F.cross_entropy(rp.unsqueeze(0), torch.tensor([target], device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % max(1, n_steps // 8) == 0:
            curve.append({'step': step, 'loss': float(np.mean(losses[-200:])), 'op': ev['op']})
            log(f"  step {step}/{n_steps} loss={np.mean(losses[-200:]):.4f} [{ev['op']}]")
    net.eval()
    arc1 = s271.arc_enc_hash(can)

    @torch.no_grad()
    def examine(p):
        r = random.Random(SEED + 7)
        det = {op: defaultdict(list) for op in OPS}
        clean_margin_none, clean_margin_forged = ([], [])
        rep = defaultdict(list)
        unk_margin_rec, unk_margin_unrec = ([], [])
        restored_model, restored_votes = ([], [])
        rand_floor_det, rand_floor_rep = ([], [])
        single_cand = 0
        for it in p['items']:
            if len(it['slots']) < 2:
                continue
            for sl in it['slots']:
                for op in OPS:
                    ev = corrupt(p, it, r, op, sl)
                    if ev is None:
                        continue
                    E, same, nf = graph_inputs(p, ev, it)
                    h, g = net.embed(E, same, nf)
                    dg = net.diagnose(h, g)
                    n = len(ev['vals'])
                    pick = int(dg.argmax())
                    clean_m = float(dg[n] - dg[:n].max())
                    vd = votes_detector(ev)
                    if op == 'none':
                        det[op]['model'].append(int(pick == n))
                        det[op]['votes'].append(int(vd is None))
                        clean_margin_none.append(clean_m)
                        continue
                    clean_margin_forged.append(clean_m)
                    det[op]['model'].append(int(pick in ev['forged']))
                    det[op]['votes'].append(int(vd in ev['forged']))
                    rand_floor_det.append(len(ev['forged']) / (n + 1))
                    cands, target, keep = repair_candidates(ev, ev['forged'])
                    if len(cands) < 2:
                        single_cand += 1
                        continue
                    masks = [torch.tensor([i not in ev['forged'] and ev['vals'][i] == c for i in range(n)], device=device) for c in cands]
                    rp = net.repair(h, g, masks)
                    rpick = int(rp.argmax())
                    ans = cands[rpick] if rpick < len(cands) else None
                    recoverable = target < len(cands)
                    unk_m = float(rp[-1] - rp[:len(cands)].max()) if cands else 0.0
                    (unk_margin_rec if recoverable else unk_margin_unrec).append(unk_m)
                    rand_floor_rep.append(1.0 / len(cands) if recoverable else 0.0)
                    if ans is None:
                        rep['model_r'].append(args.abstain_reward)
                    else:
                        rep['model_r'].append(1.0 if ans == ev['orig'] else -args.wrong_cost)
                        rep['model_acc'].append(int(ans == ev['orig']))
                    rep['model_ans'].append(int(ans is not None))
                    va = votes_repair(ev, ev['forged'])
                    if va is None:
                        rep['votes_r'].append(args.abstain_reward)
                    else:
                        rep['votes_r'].append(1.0 if va == ev['orig'] else -args.wrong_cost)
                        rep['votes_acc'].append(int(va == ev['orig']))
                    rep['votes_ans'].append(int(va is not None))
                    orig_vals = [p['tape'].values[s] for s in it['slots']]
                    true_v = Counter(orig_vals).most_common(1)[0][0]
                    fixed = [v for i, v in enumerate(ev['vals']) if i not in ev['forged']]
                    restored_model.append(int(bool(ans) and Counter(fixed + [ans]).most_common(1)[0][0] == true_v))
                    restored_votes.append(int(bool(va) and Counter(fixed + [va]).most_common(1)[0][0] == true_v))
        m = lambda xs: float(np.mean(xs)) if len(xs) else float('nan')
        forged_model = [x for op in ('replace', 'dup') for x in det[op]['model']]
        forged_votes = [x for op in ('replace', 'dup') for x in det[op]['votes']]
        cm_auc = s286.auc(clean_margin_none, clean_margin_forged)
        um_auc = s286.auc(unk_margin_unrec, unk_margin_rec)
        return {'n_by_op': {op: len(det[op]['model']) for op in OPS}, 'detection': {'model_forged': m(forged_model), 'votes_forged': m(forged_votes), 'random_floor': m(rand_floor_det), 'model_dup': m(det['dup']['model']), 'votes_dup': m(det['dup']['votes']), 'model_clean_pass': m(det['none']['model']), 'votes_clean_pass': m(det['none']['votes']), 'clean_margin_auc': cm_auc, 'clean_margin_auc_z': s286.auc_z(cm_auc, len(clean_margin_none), len(clean_margin_forged))}, 'repair_true_flag': {'model_reward': m(rep['model_r']), 'votes_reward': m(rep['votes_r']), 'model_accuracy': m(rep['model_acc']), 'votes_accuracy': m(rep['votes_acc']), 'model_coverage': m(rep['model_ans']), 'votes_coverage': m(rep['votes_ans']), 'random_floor': m(rand_floor_rep), 'unknown_margin_auc': um_auc, 'unknown_margin_auc_z': s286.auc_z(um_auc, len(unk_margin_unrec), len(unk_margin_rec)), 'n_recoverable': len(unk_margin_rec), 'n_unrecoverable': len(unk_margin_unrec), 'single_candidate_skipped': single_cand}, 'observer_verdict_restored': {'model': m(restored_model), 'votes': m(restored_votes), 'n': len(restored_model), 'note': "the proposal's original reward, demoted to an observer: gradient on this has an empty tape as its fixed point - and measured to rank two real arms backwards (min3 repair 0.433 vs votes 0.049 while the observer said 0.507 vs 0.775; min2 repair 0.085 vs 0.184 while the observer said 0.842 vs 0.803)"}}
    ctrl = examine(pack)
    held = new_pack(random.Random(SEED + 99), eval_lines, 1)
    log(f"  held tape: {held['n_addresses']} addresses, {held['n_slots']} slots")
    ex = examine(held)
    log(f'  CONTROL {json.dumps(ctrl)}')
    log(f'  HELD {json.dumps(ex)}')
    d, rp_ = (ex['detection'], ex['repair_true_flag'])
    g_arc = arc0 == arc1
    g_task = all((ex['n_by_op'][op] >= 2 * s286.MIN_ANSWERED for op in OPS))
    g_detects = bool(d['model_forged'] > d['votes_forged'] and d['model_forged'] > d['random_floor'])
    g_dup = bool(not math.isnan(d['model_dup']) and d['model_dup'] > d['votes_dup'])
    g_clean = bool(not math.isnan(d['clean_margin_auc_z']) and d['clean_margin_auc_z'] > 1.645)
    g_repairs = bool(rp_['model_reward'] > rp_['votes_reward'] and rp_['model_coverage'] * ex['n_by_op']['replace'] >= s286.MIN_ANSWERED and (not math.isnan(rp_['model_accuracy'])))
    honest_testable = rp_['n_recoverable'] >= s286.MIN_ANSWERED and rp_['n_unrecoverable'] >= s286.MIN_ANSWERED
    g_honest = None if not honest_testable else bool(rp_['unknown_margin_auc_z'] > 1.645)
    overall = 'NO_TASK' if not g_task else 'REPAIR_OK' if g_arc and g_detects and g_dup and g_clean and g_repairs and g_honest else 'REPAIR_PARTIAL' if g_arc and (g_detects or g_repairs) else 'REPAIR_NO'
    out = {'stage': 288, 'overall': overall, 'seed': SEED, 'smoke': args.smoke, 'run_tag': args.run_tag, 'holdout': args.holdout, 'train_steps': n_steps, 'tape_period': tape_period, 'params': n_params, 'reward': {'correct': 1.0, 'wrong': -args.wrong_cost, 'abstain': args.abstain_reward}, 'train_ops_seen': dict(op_seen), 'gates': {'G_arc_enc_frozen': g_arc, 'G_task_exists': g_task, 'G_detects_forgery': g_detects, 'G_detects_dup': g_dup, 'G_flags_clean': g_clean, 'G_repairs': g_repairs, 'G_honest_unrecoverable': g_honest}, 'held_out': ex, 'train_control': ctrl, 'curve': curve, 'arc_enc_hash_before': arc0, 'arc_enc_hash_after': arc1, 'fp_version': s271.fp_version(), 'note': "The direction 286 opened, taken to its end: the tape is not only the label, it is the exercise machine. Break a clean address - swap a mention for a foreign one, or for two or three copies of it - and require where and what, or the admission that the surviving evidence cannot say. The truth is free because the break is ours, so examples are unlimited, honesty is trained on breaks KNOWN to be unrecoverable, and the counting rival is wrong by construction exactly where the forgery is the majority. The judges' consistency - the proposal's original reward - is an observer only: its fixed point as a gradient is an empty tape. Diagnosis and repair share one relational embedding of ranks and indicators, identity unrepresentable, answers are indices, CLEAN and UNKNOWN are rows. Cloze, the lying tape and lie-dup were all points in this corruption family; 288 trains on the family and is examined at points it did not train on: a held-out corpus or held-out subjects, chosen by --holdout.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'stage288_decision{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (RES / f'stage288_mini{tag}.md').write_text(f"# Stage 288 repair ({args.holdout} holdout)\n\n**{overall}**{(' · SMOKE' if args.smoke else '')} · params **{n_params}**\n\n| held out | model | votes | random |\n|---|---:|---:|---:|\n| detect forged | {d['model_forged']:.3f} | {d['votes_forged']:.3f} | {d['random_floor']:.3f} |\n| detect dup | {d['model_dup']:.3f} | {d['votes_dup']:.3f} | |\n| clean pass | {d['model_clean_pass']:.3f} | {d['votes_clean_pass']:.3f} | |\n| repair reward | {rp_['model_reward']:.3f} | {rp_['votes_reward']:.3f} | {rp_['random_floor']:.3f} |\n\n- clean-margin AUC {d['clean_margin_auc']:.3f} ({d['clean_margin_auc_z']:+.2f} sigma)\n- UNKNOWN margin AUC {rp_['unknown_margin_auc']:.3f} ({rp_['unknown_margin_auc_z']:+.2f} sigma) on {rp_['n_unrecoverable']} unrecoverable vs {rp_['n_recoverable']} recoverable\n- observer: verdict restored {ex['observer_verdict_restored']['model']:.3f} vs votes {ex['observer_verdict_restored']['votes']:.3f}\n\n## Gates\n\n" + ''.join((f'- {k}: **{v}**\n' for k, v in out['gates'].items())), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())