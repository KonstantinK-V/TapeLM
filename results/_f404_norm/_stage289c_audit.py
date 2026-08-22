"""
Stage 289c — the unprompted audit: the mind publishes where it is competent, before being asked.

Every stage so far answers a question someone posed. This one asks the mind to sweep its own
tape and say, unprompted, which regions of it it can be trusted on. That is not a new
capability bolted on - it is the only honest use of the confidence the earlier stages already
produce, and it is what turns a set of gate booleans into something a person can act on.

The claim is narrow and checkable: **the mind's stated confidence must be an empirical
frequency.** If it publishes "0.8 on this region", it must be right about 80% of the time
there. That is calibration, and it is measurable without any authored label, because the truth
of every question in the sweep is already free from the tape.

What makes this an audit rather than a report card:

  - The regions are not hand-drawn. They are cut by properties the mind can see WITHOUT the
    answer: how many mentions the address has, and whether they agree. Cutting by anything that
    needs the truth would make the map a restatement of the score. Rare context words and
    sibling relations are two more honest cuts and are deliberately held back - every extra cut
    splits the denominators, and a region under MIN_ANSWERED states nothing.
  - The mind ranks regions by its own confidence and the examiner checks that ranking against
    realised accuracy. A map that is right on average but orders the regions wrongly is
    useless: you would trust the wrong half of your own tape.
  - Refusal is a legitimate cell. A region the mind declines is scored on whether declining was
    right - measured as the accuracy it WOULD have had there, which the tape knows.

Gates:
  G_calibrated             stated confidence tracks realised accuracy (ECE under a bound that
                           is derived from the bin counts, not chosen)
  G_ranks_regions          Spearman between stated confidence and realised accuracy > 0
  G_refusal_is_informed    the regions it declines really are the ones it would fail on
  G_map_transfers          the ordering learned on the train tape survives on the held-out tape

  python _stage289c_audit.py --smoke
  python _stage289c_audit.py --train-steps 6000
  python _stage289c_audit.py --train-steps 6000 --holdout address
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
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage271_controller as s271
import _stage279_write_decision as s279
import _stage280_raw_exam as s280
import _stage286_evidence as s286
import _stage289_derivation as s289
import _stage289a_presupposition as s289a
from _tape_speed import CachedBank, install_assertion_cache, install_fast_fp_addresses
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 2893
LOG_PATH = RES / '_stage289c_log.txt'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line)

def region_of(p, q) -> str:
    """Cut the tape by what is visible WITHOUT the answer.

    Two coordinates, both readable from the question and the evidence alone: how deep the
    address is, and whether its mentions agree. Nothing that needs the truth may enter, or the
    map becomes a restatement of the score - 286's exam tautology in the shape of a chart. Rare
    words and sibling relations belong here too and are deliberately left out for now: each new
    cut splits the denominators, and a region under MIN_ANSWERED states nothing.
    """
    n = len(q['slots'])
    vals = q['vals']
    top = Counter(vals).most_common(1)[0][1]
    depth = 'thin' if n <= 2 else 'mid' if n <= 4 else 'deep'
    agree = 'unanimous' if top == n else 'majority' if top > n / 2 else 'split'
    return f"{q['verb']}|{depth}|{agree}"

def spearman(a, b) -> float:
    n = len(a)
    if n < 3:
        return float('nan')

    def rank(x):
        o = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[o[j + 1]] == x[o[i]]:
                j += 1
            m = (i + j) / 2.0
            for k in range(i, j + 1):
                r[o[k]] = m
            i = j + 1
        return r
    ra, rb = (rank(a), rank(b))
    ma, mb = (sum(ra) / n, sum(rb) / n)
    num = sum(((x - ma) * (y - mb) for x, y in zip(ra, rb)))
    da = math.sqrt(sum(((x - ma) ** 2 for x in ra)))
    db = math.sqrt(sum(((y - mb) ** 2 for y in rb)))
    return num / (da * db) if da > 0 and db > 0 else float('nan')

def ece(pairs, n_bins: int=10):
    """Expected calibration error, with the bound it must clear DERIVED rather than chosen.

    Each bin's realised accuracy is a binomial mean, so its own sampling noise is
    sqrt(acc(1-acc)/n). Summing that across bins, weighted the same way ECE is, gives the ECE a
    perfectly calibrated mind would still show on this many questions. The gate is ECE below
    that, so the threshold comes from the data and not from taste.
    """
    bins = defaultdict(list)
    for conf, hit in pairs:
        bins[min(n_bins - 1, int(conf * n_bins))].append((conf, hit))
    n = len(pairs)
    if not n:
        return (float('nan'), float('nan'), [])
    err, noise, detail = (0.0, 0.0, [])
    for b in sorted(bins):
        cs = [c for c, _ in bins[b]]
        hs = [h for _, h in bins[b]]
        k = len(hs)
        acc, cf = (sum(hs) / k, sum(cs) / k)
        err += k / n * abs(acc - cf)
        noise += k / n * math.sqrt(max(acc * (1 - acc), 1e-09) / k)
        detail.append({'bin': b, 'n': k, 'stated': cf, 'realised': acc})
    return (err, noise, detail)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train-steps', type=int, default=0)
    ap.add_argument('--tape-period', type=int, default=50)
    ap.add_argument('--addresses', type=int, default=0)
    ap.add_argument('--min-mentions', type=int, default=2)
    ap.add_argument('--address-tau', type=float, default=0.9)
    ap.add_argument('--address-overlap', type=int, default=2)
    ap.add_argument('--addr-key', choices=('two', 'set', 'mean'), default='two')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--holdout', choices=('corpus', 'address'), default='corpus')
    ap.add_argument('--no-scan-cache', action='store_true', help='disable the exact corpus-scan memo (use to verify it changes nothing)')
    ap.add_argument('--no-fast-grouping', action='store_true', help='disable the batched single-link grouping (use to verify it changes nothing)')
    ap.add_argument('--wiki-bytes', type=int, default=0)
    ap.add_argument('--train-lines', type=int, default=0)
    ap.add_argument('--eval-lines', type=int, default=0)
    ap.add_argument('--refuse-at', type=float, default=0.0, help="0 means derive the threshold: refuse where stated confidence is below the point at which answering stops beating the region's own floor")
    ap.add_argument('--edge-channels', type=str, default='same,cos,rare', help="match 289's edge ablation")
    ap.add_argument('--no-ladder', action='store_true', help="match 289's ablation: audit a mind trained without the ladder")
    ap.add_argument('--run-tag', type=str, default='')
    args = ap.parse_args()
    s289.LADDER_ON = not args.no_ladder
    s289.EDGES_ON = {c.strip() for c in args.edge_channels.split(',') if c.strip()}
    global LOG_PATH
    tag = args.run_tag and f'_{args.run_tag}' or ''
    tag += '_addrholdout' if args.holdout == 'address' else ''
    LOG_PATH = RES / f'_stage289c_log{tag}.txt'
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_steps = args.train_steps or (600 if args.smoke else 6000)
    n_addr = args.addresses or (300 if args.smoke else 400)
    log(f'Stage289c audit start {datetime.now(timezone.utc).isoformat()} device={device}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = CachedBank(FpBank(can, stoi, device))
    arc0 = s271.arc_enc_hash(can)
    _nouns: dict = {}
    _raw_common = s279.common_nouns

    def _cached_common(lines, min_lower: int=3):
        k = (id(lines), len(lines), min_lower)
        if k not in _nouns:
            _nouns[k] = _raw_common(lines, min_lower)
        return _nouns[k]
    s279.common_nouns = _cached_common
    if not args.no_scan_cache:
        install_assertion_cache(s279)
    if not args.no_fast_grouping:
        install_fast_fp_addresses(s279)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(args.wiki_bytes or (4000000 if args.smoke else 30000000))
    all_lines = [l.strip() for l in wtext.split('\n') if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][:args.train_lines or (3000 if args.smoke else 25000)]
    eval_lines = all_lines[cut:][:args.eval_lines or (1500 if args.smoke else 12000)]
    if args.holdout == 'address':
        eval_lines = train_lines

    def side(address: str) -> int:
        return int(hashlib.sha1(s289a.anchor_of(address).encode('utf-8')).hexdigest(), 16) & 1

    def new_pack(r, lines, want):
        p = s280.pack_from_corpus(lines, bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, n_addr=n_addr, min_mentions=args.min_mentions, tau=args.address_tau, overlap=args.address_overlap, soft_match=0.0, min_per_family=8, addr_key=args.addr_key)
        if args.holdout == 'address':
            p = dict(p)
            p['items'] = [it for it in p['items'] if side(it['address']) == want]
        return p
    net = s289.Deriver(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = int(sum((x.numel() for x in net.parameters())))
    pack = new_pack(rng, train_lines, 0)
    qs = s289.questions_for(pack, rng)
    held = new_pack(random.Random(SEED + 99), eval_lines, 1)
    held_qs = s289.questions_for(held, random.Random(SEED + 7))

    def by_verb(qq):
        d = defaultdict(list)
        for q in qq:
            d[q['verb']].append(q)
        return d
    bv = by_verb(qs)
    verbs = [v for v in s289.TRAIN_VERBS if bv.get(v)]
    log(f"  tape: {pack['n_addresses']} addresses | questions {json.dumps({k: len(v) for k, v in bv.items()})} | params {n_params}")
    for step in range(1, n_steps + 1):
        if (step - 1) % args.tape_period == 0 and step > 1:
            pack = new_pack(rng, train_lines, 0)
            qs = s289.questions_for(pack, rng)
            bv = by_verb(qs)
            verbs = [v for v in s289.TRAIN_VERBS if bv.get(v)]
        if not verbs:
            log('  empty tape after resample')
            return 1
        v = verbs[rng.randrange(len(verbs))]
        q = bv[v][rng.randrange(len(bv[v]))]
        loss = s289.loss_for(net, pack, q, device, bank)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % max(1, n_steps // 8) == 0:
            log(f'  step {step}/{n_steps} loss={float(loss):.4f}')
    net.eval()
    arc1 = s271.arc_enc_hash(can)

    @torch.no_grad()
    def sweep(p, qq):
        """Walk the whole tape unprompted and record, per question, what the mind would say,
        how sure it is, and whether it is right. Nothing is asked of it - it audits itself."""
        rows = []
        for q in qq:
            pr, pred, truth = s289.predict_with_confidence(net, p, q, device, bank)
            rows.append({'region': region_of(p, q), 'verb': q['verb'], 'conf': pr, 'hit': int(pred == truth), 'n_choices': s289.n_choices(q)})
        return rows

    def summarise(rows):
        by_r = defaultdict(list)
        for r in rows:
            by_r[r['region']].append(r)
        out = {}
        for reg, rs in sorted(by_r.items()):
            if len(rs) < s286.MIN_ANSWERED:
                continue
            out[reg] = {'n': len(rs), 'stated_confidence': float(np.mean([r['conf'] for r in rs])), 'realised_accuracy': float(np.mean([r['hit'] for r in rs])), 'random_floor': float(np.mean([1.0 / r['n_choices'] for r in rs]))}
        return out
    tr_rows, he_rows = (sweep(pack, qs), sweep(held, held_qs))
    tr_map, he_map = (summarise(tr_rows), summarise(he_rows))
    log(f'  MAP_HELD {json.dumps(he_map)}')
    e, e_noise, bins = ece([(r['conf'], r['hit']) for r in he_rows])

    def _temp_scale(rows_fit, rows_apply):
        best_t, best_l = (1.0, float('inf'))
        for t in [0.05 * k for k in range(1, 101)]:
            l = 0.0
            for r in rows_fit:
                q = min(max(r['conf'] ** (1.0 / t), 1e-06), 1 - 1e-06)
                l -= math.log(q) if r['hit'] else math.log(1 - q)
            if l < best_l:
                best_t, best_l = (t, l)
        return (best_t, [(min(max(r['conf'] ** (1.0 / best_t), 1e-06), 1 - 1e-06), r['hit']) for r in rows_apply])
    temp, scaled = _temp_scale(tr_rows, he_rows)
    e_t, e_t_noise, bins_t = ece(scaled)
    regions = sorted(he_map)
    rho = spearman([he_map[r]['stated_confidence'] for r in regions], [he_map[r]['realised_accuracy'] for r in regions])
    shared = [r for r in regions if r in tr_map]
    rho_transfer = spearman([tr_map[r]['stated_confidence'] for r in shared], [he_map[r]['realised_accuracy'] for r in shared])
    thr = args.refuse_at
    if thr <= 0.0:
        cands = sorted({round(r['conf'], 3) for r in he_rows})
        thr = 0.0
        for t in cands:
            keep = [r for r in he_rows if r['conf'] >= t]
            if len(keep) < s286.MIN_ANSWERED:
                break
            acc = np.mean([r['hit'] for r in keep])
            flo = np.mean([1.0 / r['n_choices'] for r in keep])
            if acc > flo:
                thr = t
                break
    answered = [r for r in he_rows if r['conf'] >= thr]
    refused = [r for r in he_rows if r['conf'] < thr]
    refusal = {'threshold': thr, 'answered_n': len(answered), 'refused_n': len(refused), 'answered_accuracy': float(np.mean([r['hit'] for r in answered])) if answered else float('nan'), 'refused_would_have_been': float(np.mean([r['hit'] for r in refused])) if refused else float('nan'), 'vacuous': not refused}
    g_arc = arc0 == arc1
    g_task = len(he_map) >= 3 and len(he_rows) >= 4 * s286.MIN_ANSWERED
    g_calibrated = bool(not math.isnan(e) and e <= e_noise)
    g_calibrated_t = bool(not math.isnan(e_t) and e_t <= e_t_noise)
    g_ranks = bool(not math.isnan(rho) and rho > 0)
    g_refusal = bool(refusal['refused_n'] >= s286.MIN_ANSWERED and refusal['answered_n'] >= s286.MIN_ANSWERED and (refusal['refused_would_have_been'] < refusal['answered_accuracy']))
    g_transfers = bool(not math.isnan(rho_transfer) and rho_transfer > 0)
    overall = 'NO_TASK' if not (g_task and g_arc) else 'AUDIT_OK' if g_calibrated and g_ranks and g_refusal and g_transfers else 'AUDIT_PARTIAL' if g_ranks or g_calibrated else 'AUDIT_NO'
    out = {'stage': '289c', 'overall': overall, 'seed': SEED, 'smoke': args.smoke, 'holdout': args.holdout, 'run_tag': args.run_tag, 'train_steps': n_steps, 'params': n_params, 'gates': {'G_arc_enc_frozen': g_arc, 'G_task_exists': g_task, 'G_calibrated': g_calibrated, 'G_calibrated_after_train_temperature': g_calibrated_t, 'G_ranks_regions': g_ranks, 'G_refusal_is_informed': g_refusal, 'G_map_transfers': g_transfers}, 'competence_map_held': he_map, 'competence_map_train': tr_map, 'calibration': {'ece': e, 'ece_sampling_noise': e_noise, 'bins': bins}, 'calibration_after_train_temperature': {'temperature': temp, 'ece': e_t, 'ece_sampling_noise': e_t_noise, 'bins': bins_t, 'note': 'one parameter, fitted on the TRAIN sweep only, never on the held-out rows. It can rescale confidence but cannot invent an ordering, so G_ranks_regions remains the primary claim and this says whether the remaining error was a scale problem or a knowledge problem'}, 'spearman_conf_vs_accuracy': rho, 'spearman_train_map_vs_held_accuracy': rho_transfer, 'refusal': refusal, 'region_note': "regions are cut by depth and agreement only - both readable without the answer. Cutting by anything that needs the truth would make the map a restatement of the score, which is 286's exam tautology wearing a chart", 'calibration_note': "the ECE bound is DERIVED: each bin's realised accuracy is a binomial mean whose own noise is sqrt(acc(1-acc)/n), weighted the same way ECE is. A perfectly calibrated mind still shows that much on this many questions, so the gate is ECE under its own noise floor and no threshold was chosen by taste", 'arc_enc_hash_before': arc0, 'arc_enc_hash_after': arc1, 'fp_version': s271.fp_version(), 'note': "The unprompted audit. The mind sweeps its own tape and publishes where it can be trusted, before anyone asks. The claim is narrow and checkable: stated confidence must be an empirical frequency - 0.8 must mean right eight times in ten - and the ordering of regions must be right, because a map that is accurate on average but orders regions wrongly makes you trust the wrong half of your tape. Refusal is a cell in the map, not an absence: the tape knows what the refused questions would have scored, so declining is checkable too. No new capability and no new model - 289's mind and 289's questions, read through the confidence they already produce, because a competence map that needed its own model would be a second mind grading the first.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'stage289c_decision{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates'], 'ece': e, 'noise': e_noise, 'rho': rho, 'refusal': refusal}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())