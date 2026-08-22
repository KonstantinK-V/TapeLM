"""
Stage 287 — Is there a second layer of ink at all, or only the first one averaged?

The proposal is a double tape: a lower ink over letters, and an UPPER ink over chunks of text
carrying something the letters do not have. Before building any of it, one thing has to be
settled, and it can be settled with no training whatsoever.

    Today's upper layer is not a layer. ctx_fp(T) = norm(mean over words of fp(word)).

It is the lower ink, averaged. And that has a consequence nobody has written down: the whole
stack is order-blind TWICE. fp pools characters before the feed-forward, so a word is a
character histogram and anagrams collide bit-exactly (277). ctx_fp then means over words, so a
sentence is a bag of those histograms and any permutation of the words gives a bit-identical
vector. "the dog bit the man" and "the man bit the dog" are the same point on this tape.

So the idea is not wrong, it is pointing at a real hole. The only question worth asking first
is whether an upper ink can be built that is NOT a function of the lower one - because if it
is a function of it, "semantics of the chunk" has nowhere to live, whatever we call it.

There are exactly two things a chunk has that its bag of words does not: ADJACENCY, and
corpus-level co-occurrence. This stage tests the first, because it needs no new machinery:

    lower  L(T) = norm( mean over words w of fp(w) )              - what the tape uses today
    upper  U(T) = norm( mean over adjacent k-word spans s of fp(s) )

U is built from the same frozen encoder and the same characters. It differs only in WHERE the
sum happens: fp of "was born" is the encoder applied to the two words' combined characters,
which is not any function of fp("was") and fp("born"), because the feed-forward sits between
the histogram and the output. Nothing is trained and nothing is tuned.

Two things about U to state before it is measured, so neither can be discovered later as a
surprise. The space is not in the character table and maps to the padding id, so fp("was born")
is fp of the concatenated characters - the span is a combined histogram, which is exactly the
point. And U is invariant to EXACT REVERSAL: "A B C" gives spans {AB, BC} while "C B A" gives
{CB, BA}, and a span's histogram is symmetric, so those multisets are identical. U therefore
separates most permutations but not that one. It is a partial order channel, not a full one,
and the shuffle control below measures how much of the order it actually recovers rather than
assuming.

Four measurements, in the order that can kill the idea fastest:

  1. WIRING PROOF. Shuffle the words inside every text. L must come back BIT-IDENTICAL - if it
     does not, this file is measuring something other than what it claims. U must move. If U
     does not move either, the upper channel has no order content and the idea dies here.

  2. IS IT A SECOND CHANNEL? Spearman rank correlation between cos_L and cos_U over slot pairs.
     At rho = 1 the upper ink is the lower one wearing a hat.

  3. DOES IT CARRY UNIQUE SIGNAL? Take the discrimination the tape actually needs - two
     mentions of the SAME address against two of different addresses - and score three
     channels: L alone, U alone, and U with its linear fit on L removed. Only the third
     answers "unique", and it is gated against its own noise the way 286 gates its AUCs.

  4. NEGATIVE CONTROLS. A residual test that cannot say "no" is not a test, and this project
     has shipped three vacuous gates already. The first attempt used L against ITSELF, which
     read AUC 0.163 at -12 sigma and correctly reported the measurement broken - but the fault
     was in the control, not the estimator. Regressing a channel on itself leaves
     residual = eps*L - delta with eps at machine scale: a perfectly MONOTONE function of L,
     whose AUC is therefore AUC(L) or 1-AUC(L) and can never be 0.5. The control was ill-posed
     by construction. Two well-posed ones replace it: a LABEL PERMUTATION null, which keeps the
     real residual and shuffles the classes, and a REDUNDANT-CHANNEL null, an affine copy of L
     plus noise matched to the observed residual scale. Both must read chance.

What this stage cannot say: whether the unique information is "semantics". It measures
information, not meaning. Unique information is necessary for meaning to be there and does not
establish it - the sufficient test is whether the mind uses it, and 286 already has the socket
(--ctx-channel) for that. Run this first because it is free; wire it only if it passes.

  python _stage287_two_layer_ink.py --smoke
  python _stage287_two_layer_ink.py --span 2
  python _stage287_two_layer_ink.py --span 3
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage286_evidence as s286
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 287
WORD_RE = re.compile('[A-Za-z][a-z]{2,}')
MAX_WORDS = 40
FLOAT_NOISE = 1e-06
LOG_PATH = RES / '_stage287_log.txt'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line)

def words_of(text: str, exclude: str | None=None) -> list[str]:
    return [w for w in WORD_RE.findall(text) if w != exclude][:MAX_WORDS]

def lower_ink(bank, ws: list[str]):
    """What the tape uses today: the mean of the word fingerprints."""
    if len(ws) < 3:
        return None
    return F.normalize(bank.fp(ws).float().mean(0), dim=-1)

def upper_ink(bank, ws: list[str], span: int):
    """The candidate second layer: the mean over ADJACENT spans.

    fp(" ".join(span)) is the frozen encoder applied to the span's combined characters. It is
    not a function of the member words' fingerprints - the feed-forward sits between the
    histogram and the output - so this channel can hold what the averaged one cannot. The
    encoder's own budget is 24 characters per fingerprint, which is why span stays small.
    """
    if len(ws) < span + 1:
        return None
    spans = [' '.join(ws[i:i + span]) for i in range(len(ws) - span + 1)]
    return F.normalize(bank.fp(spans).float().mean(0), dim=-1)

def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, so a monotone re-encoding of the same channel still reads as 1."""
    if len(a) < 3:
        return float('nan')
    ra, rb = (np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float))
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / d) if d > 0 else float('nan')

def residual_of(u: list[float], l: list[float]) -> list[float]:
    """U with its best linear explanation by L removed.

    Fitted on every pair regardless of label, so the label never touches the fit - what is
    removed is "the part of the upper channel that the lower channel already accounts for",
    and what survives is the only thing that could justify a second layer.
    """
    U, L = (np.asarray(u), np.asarray(l))
    A = np.stack([L, np.ones_like(L)], 1)
    coef, *_ = np.linalg.lstsq(A, U, rcond=None)
    return list(U - A @ coef)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--span', type=int, default=2, help="words per upper-ink span. 2 is the smallest that can hold adjacency; above 3 the spans start hitting the encoder's 24-character budget and the channel degrades into a truncation artefact.")
    ap.add_argument('--addresses', type=int, default=0)
    ap.add_argument('--min-mentions', type=int, default=2)
    ap.add_argument('--max-pairs', type=int, default=4000)
    ap.add_argument('--address-tau', type=float, default=0.9)
    ap.add_argument('--address-overlap', type=int, default=2)
    ap.add_argument('--addr-key', choices=('two', 'set', 'mean'), default='two')
    ap.add_argument('--run-tag', type=str, default='')
    args = ap.parse_args()
    global LOG_PATH
    tag = args.run_tag and f'_{args.run_tag}' or ''
    tag += f'_span{args.span}'
    LOG_PATH = RES / f'_stage287_log{tag}.txt'
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_addr = args.addresses or (60 if args.smoke else 400)
    log(f'Stage287 two-layer ink start {datetime.now(timezone.utc).isoformat()} device={device} span={args.span}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(4000000 if args.smoke else 30000000)
    all_lines = [l.strip() for l in wtext.split('\n') if 80 <= len(l.strip()) <= 400]
    lines = all_lines[:int(0.7 * len(all_lines))][:3000 if args.smoke else 25000]
    pack = s280.pack_from_corpus(lines, bank=bank, tok=tok, pad_id=pad_id, device=device, rng=rng, n_addr=n_addr, min_mentions=args.min_mentions, tau=args.address_tau, overlap=args.address_overlap, soft_match=0.0, min_per_family=0, addr_key=args.addr_key)
    texts, vals = (pack['texts'], pack['tape'].values)
    log(f"  tape: {pack['n_addresses']} addresses, {pack['n_slots']} slots")
    L, U, Ls, Us, keep = ({}, {}, {}, {}, [])
    srng = random.Random(SEED + 1)
    for sl, t in enumerate(texts):
        ws = words_of(t, exclude=vals[sl])
        lo, up = (lower_ink(bank, ws), upper_ink(bank, ws, args.span))
        if lo is None or up is None:
            continue
        sh = ws[:]
        srng.shuffle(sh)
        L[sl], U[sl] = (lo, up)
        Ls[sl], Us[sl] = (lower_ink(bank, sh), upper_ink(bank, sh, args.span))
        keep.append(sl)
    log(f'  usable slots: {len(keep)}')
    if len(keep) < 3 * s286.MIN_ANSWERED:
        log('  too few usable slots')
        return 1
    dL = [float((L[s] - Ls[s]).abs().max()) for s in keep]
    cU = [float(U[s] @ Us[s]) for s in keep]
    order_blind_L = max(dL) < FLOAT_NOISE
    u_moves = float(np.median(cU)) < 1.0 - FLOAT_NOISE
    log(f'  shuffle control: max|dL| = {max(dL):.3e} (must be ~0), median cos(U, U_shuffled) = {np.median(cU):.4f} (must be < 1)')
    by_addr = defaultdict(list)
    for it in pack['items']:
        for s in it['slots']:
            if s in L:
                by_addr[it['address']].append(s)
    same = [(a, b) for sl in by_addr.values() for i, a in enumerate(sl) for b in sl[i + 1:]]
    prng = random.Random(SEED + 2)
    diff, guard = ([], 0)
    while len(diff) < len(same) and guard < 40 * max(1, len(same)):
        guard += 1
        a, b = (prng.choice(keep), prng.choice(keep))
        if a != b and pack['slot_addr'][a] != pack['slot_addr'][b]:
            diff.append((a, b))
    if len(same) > args.max_pairs:
        same = prng.sample(same, args.max_pairs)
        diff = prng.sample(diff, args.max_pairs)
    log(f'  pairs: {len(same)} same-address, {len(diff)} different-address')
    if min(len(same), len(diff)) < s286.MIN_ANSWERED:
        log('  not enough pairs of one kind; raise --addresses')
        return 1

    def cosines(pairs):
        return ([float(L[a] @ L[b]) for a, b in pairs], [float(U[a] @ U[b]) for a, b in pairs])
    lS, uS = cosines(same)
    lD, uD = cosines(diff)
    rho = spearman(lS + lD, uS + uD)
    res = residual_of(uS + uD, lS + lD)
    rS, rD = (res[:len(uS)], res[len(uS):])

    def scored(pos, neg):
        a = s286.auc(pos, neg)
        return {'auc': a, 'auc_z': s286.auc_z(a, len(pos), len(neg))}
    ch_l, ch_u, ch_r = (scored(lS, lD), scored(uS, uD), scored(rS, rD))
    nrng = np.random.default_rng(SEED + 3)
    allr = np.asarray(rS + rD)
    n_pos = len(rS)
    perm = []
    for _ in range(99):
        idx = nrng.permutation(len(allr))
        perm.append(s286.auc(list(allr[idx[:n_pos]]), list(allr[idx[n_pos:]])))
    ch_perm = {'auc': float(np.mean(perm)), 'auc_z': s286.auc_z(float(np.mean(perm)), n_pos, len(allr) - n_pos), 'auc_p05': float(np.quantile(perm, 0.05)), 'auc_p95': float(np.quantile(perm, 0.95)), 'n_permutations': len(perm)}
    Lall = np.asarray(lS + lD)
    fake = Lall * 2.0 - 0.5 + nrng.normal(0.0, float(np.std(allr)) + 1e-12, size=len(Lall))
    fres = residual_of(list(fake), list(Lall))
    ch_fake = scored(fres[:n_pos], fres[n_pos:])
    ch_null = {'permutation': ch_perm, 'redundant_channel': ch_fake}
    log(f'  L {json.dumps(ch_l)}\n  U {json.dumps(ch_u)}\n  U|L {json.dumps(ch_r)}\n  nulls {json.dumps(ch_null)}')
    g_control = bool(abs(ch_perm['auc_z']) <= 1.645)
    g_order_blind = bool(order_blind_L)
    g_u_moves = bool(u_moves)
    g_distinct = bool(abs(rho) < 1.0 - FLOAT_NOISE)
    g_unique = bool(not math.isnan(ch_r['auc_z']) and ch_r['auc_z'] > 1.645)
    g_better = bool(ch_u['auc'] > ch_l['auc'])
    overall = 'MEASUREMENT_BROKEN' if not (g_order_blind and g_control) else 'NO_ORDER_CONTENT' if not g_u_moves else 'SECOND_LAYER_REAL' if g_distinct and g_unique else 'SECOND_LAYER_REDUNDANT'
    out = {'stage': 287, 'overall': overall, 'seed': SEED, 'smoke': args.smoke, 'span': args.span, 'run_tag': args.run_tag, 'trained_parameters': 0, 'tape': {'addresses': pack['n_addresses'], 'slots': pack['n_slots'], 'usable': len(keep)}, 'pairs': {'same_address': len(same), 'different_address': len(diff)}, 'shuffle_control': {'max_abs_delta_lower': max(dL), 'median_cos_upper_vs_shuffled': float(np.median(cU)), 'mean_cos_upper_vs_shuffled': float(np.mean(cU))}, 'spearman_upper_vs_lower': rho, 'order_content': 1.0 - float(np.median(cU)), 'channels': {'lower': ch_l, 'upper': ch_u, 'upper_given_lower': ch_r, 'nulls': ch_null}, 'gates': {'G_lower_is_order_blind': g_order_blind, 'G_residual_test_can_say_no': g_control, 'G_upper_sees_word_order': g_u_moves, 'G_channels_are_distinct': g_distinct, 'G_upper_carries_unique_signal': g_unique, 'G_upper_beats_lower_alone': g_better}, 'note': "Whether a second layer of ink exists at all, measured before anything is built and with nothing trained. Today's upper layer is not one: ctx_fp is the mean of the word fingerprints, so the stack is order-blind twice - fp pools characters before the feed-forward, and the mean over words then discards word order, which makes 'the dog bit the man' and 'the man bit the dog' the same point. The candidate upper ink is the mean over ADJACENT spans, built from the same frozen encoder: fp of a span is the encoder applied to the span's combined characters and is not any function of the member words' fingerprints, because the feed-forward sits between the histogram and the output. Four measurements in kill-fastest order: shuffling the words must leave the lower ink bit-identical (a wiring proof) and must move the upper one (or there is no order content and the idea is over); the Spearman correlation between the two cosines says whether it is a second channel or the first wearing a hat; the discrimination the tape actually needs - same address against different address - is scored for the lower channel, the upper channel, and the upper with its linear fit on the lower removed, which is the only one that can justify a second layer; and the same residual test is run on a channel redundant BY CONSTRUCTION, the lower against itself, which must come out at chance or the examiner is broken and every number above it is worthless. This measures information, not meaning: unique information is necessary for semantics to live there and does not establish it. The sufficient test is whether the mind uses it, and 286's --ctx-channel is the socket for that - wire it only if this passes.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'stage287_decision{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (RES / f'stage287_mini{tag}.md').write_text(f"# Stage 287 is there a second layer of ink (span {args.span})\n\n**{overall}**{(' · SMOKE' if args.smoke else '')} · trained parameters **0**\n\n| channel | AUC same-vs-different | sigma |\n|---|---:|---:|\n" + ''.join((f"| {k} | {v['auc']:.3f} | {v['auc_z']:+.2f} |\n" for k, v in (("lower (today's ctx_fp)", ch_l), ('upper (adjacent spans)', ch_u), ('upper given lower', ch_r), ('null: label permutation', ch_perm), ('null: redundant channel', ch_fake)))) + f'\n- shuffling words: lower moves by {max(dL):.2e} (must be ~0), upper stays at cos {np.median(cU):.4f} -> only {1 - float(np.median(cU)):.4f} of it depends on word order\n- Spearman between the two cosines: {rho:.4f}\n\n## Gates\n\n' + ''.join((f'- {k}: **{v}**\n' for k, v in out['gates'].items())), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())