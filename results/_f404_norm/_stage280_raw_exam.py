"""
Stage 280 — The whole thing, on text nobody wrote for it.

Every controller stage from 271 to 278 ran on a tape built from WITNESS_TMPL. The templates were
never the point; they existed because a controlled disagreement was needed and raw text does not
hand one over on request. 279 removed that need: writing as a WRITE / CONFIRM / DISPUTE decision
against an address produces witnesses and a support count from the corpus itself.

So this joins the two halves and asks the question GOAL.md left open. Everything that survived
its own measurement is here and nothing else:

    the tape        279's write decision, fp addressing with a votes check, common-noun anchors
                    excluded, values filtered - no template anywhere
    the families    NOT declared. An address with one value is clean, an address whose leader
                    beats the runner-up is decidable, an address where the top two tie is a tie.
                    Which questions exist is a property of the corpus.
    retrieval       word votes with classic idf, and where the votes go silent the fp address
                    proposes and a read verifies - the one job 277 left the ink
    the mind        278's policy: value baseline, BC anchor kept on through RL, the exhaustive
                    teacher, margin counted in votes, wrong -1.0 against abstain +0.75
    the exam        a disjoint slice of the corpus, so the tape at evaluation was never trained on

Two things will be worse here than on templates, and both are the point of running it.

Retrieval will fall silent. 264 measured that 49% of open-bank queries score zero on the gold,
and templates hid that because every witness shared the subject. --hop fp is the arm that says
whether the address can recover what the words missed.

And the teacher may simply be wrong. On manufactured disagreement the majority IS the truth by
construction; on wikitext a majority can be a popular error, and three sentences that mention
one name are often not three claims about one fact. G_teacher_ceiling therefore stops being a
pass/fail gate and becomes a measurement: teacher_acc_all on the corpus verdict is the ceiling
this exam actually has, and the policy is scored against that rather than against 1.0.

  python _stage280_raw_exam.py --smoke
  python _stage280_raw_exam.py --bc-episodes 4000 --rl-episodes 3000 --hop fp
  python _stage280_raw_exam.py --bc-episodes 4000 --rl-episodes 3000 --hop fp --no-hidden
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
import _stage274_truthfree_oracle as s274
import _stage278_value_baseline as s278
import _stage279_write_decision as s279
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _inprint_glue import TapeView
from _tape_index import context_words
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage280_raw_exam.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 280
FAMILIES = ('clean', 'decidable', 'tie')
LOG_PATH = RES / '_stage280_log.txt'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line)

def pack_from_corpus(lines, *, bank, tok, pad_id, device, rng, n_addr, min_mentions, tau, overlap, soft_match, idf_mode='classic', max_items=0, keep_frames=None, min_per_family=0, addr_key='set', assertions=None, group=True):
    """Write the corpus with 279's decision, then read the result as an exam.

    Nothing here declares a family or a truth. An address carries whatever the corpus said at it,
    the leader is the corpus verdict, and a tie means the corpus never settled - which is exactly
    the abstention case 278 learned, arrived at from data rather than from a template.
    """
    if assertions is not None:
        asserts = list(assertions)
    else:
        common = s279.common_nouns(lines)
        asserts, _ = s279.corpus_assertions(lines, rng, n_addr, min_mentions, 'anchor_rel', common=common)
    if keep_frames is not None and assertions is None:
        asserts = [a for a in asserts if (a['address'].split('|', 1) + [''])[1] in keep_frames]
    if callable(tau) and assertions is None:
        tau = tau(asserts, bank, overlap, min_mentions, addr_key)
    if group:
        asserts, addrs = s279.fp_addresses(asserts, bank, tau, overlap, min_mentions, addr_key=addr_key)
    else:
        seen_a = {}
        for a in asserts:
            seen_a.setdefault(a['address'], 0)
            a['straddr'] = a['address']
        addrs = list(seen_a)
    tape_w = s279.Tape(bank, soft_match)
    for a in asserts:
        tape_w.decide(a['address'], a['value'], a['source'])
    keys, akeys, ckeys, vals, texts, by_addr = ([], [], [], [], [], defaultdict(list))
    straddr = []
    poss, linos = ([], [])
    for a in asserts:
        straddr.append(a.get('straddr', a['address']))
        poss.append(a.get('pos', -1))
        linos.append(a.get('line', -1))
        anchor = a['address'].split(':', 1)[-1].split('|')[0]
        c = bank.ctx_fp(a['ctx'], exclude=a['value'])
        k = bank.fp([anchor])[0]
        keys.append(F.normalize(k + c if c is not None else k, dim=-1))
        akeys.append(F.normalize(k, dim=-1))
        ckeys.append(F.normalize(c, dim=-1) if c is not None else F.normalize(k, dim=-1))
        by_addr[a['address']].append(len(vals))
        vals.append(a['value'])
        texts.append(a['ctx'])
    postings: dict[str, list[int]] = defaultdict(list)
    postings_probe: dict[str, list[int]] = defaultdict(list)
    for cid, t in enumerate(texts):
        for w in context_words(t, exclude=vals[cid]):
            postings[w].append(cid)
        for w in context_words(t):
            postings_probe[w].append(cid)
    n_slots = len(vals)
    if idf_mode == 'classic':
        idf = {w: math.log(max(2.0, n_slots / max(1, len(postings[w])))) for w in postings}
        idf_probe = {w: math.log(max(2.0, n_slots / max(1, len(postings_probe[w])))) for w in postings_probe}
    else:
        idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in postings}
        idf_probe = {w: 1.0 / math.log(2.0 + len(postings_probe[w])) for w in postings_probe}
    items = []
    for addr in addrs:
        sids = by_addr.get(addr, [])
        if not sids:
            continue
        cnt = Counter((vals[i] for i in sids))
        ranked = cnt.most_common(2)
        lead = ranked[0][1]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if len(cnt) == 1:
            kind, truth = ('clean', ranked[0][0])
        elif lead == second:
            kind, truth = ('tie', None)
        else:
            kind, truth = ('decidable', ranked[0][0])
        tail = addr.split(':', 1)[-1]
        anchor, rel = (tail.split('|', 1) + [''])[:2]
        query = (anchor + ' ' + rel).strip()
        items.append({'S': anchor, 'query': query, 'truth': truth, 'slots': sids, 'kind': kind, 'address': addr})
    rng.shuffle(items)
    if min_per_family:
        by_kind = defaultdict(list)
        for it in items:
            by_kind[it['kind']].append(it)
        take, rest = ([], [])
        for f in ('clean', 'decidable', 'tie'):
            take += by_kind[f][:min_per_family]
            rest += by_kind[f][min_per_family:]
        items = take + rest
    if max_items:
        items = items[:max_items]
    akey, aslots, slot_addr = ([], [], [0] * len(vals))
    for ai, addr in enumerate(addrs):
        sids = by_addr.get(addr, [])
        if sids:
            akey.append(F.normalize(torch.stack([keys[i] for i in sids]).mean(0), dim=-1))
            aslots.append(sids)
            for i in sids:
                slot_addr[i] = len(aslots) - 1
    return {'tape': TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id), 'texts': texts, 'items': items, 'postings': postings, 'idf': idf, 'straddr': straddr, 'pos': poss, 'line': linos, 'postings_probe': postings_probe, 'idf_probe': idf_probe, 'texts_lc': [t.lower() for t in texts], 'addr_keys': F.normalize(torch.stack(akey).float(), dim=-1).to(device) if akey else None, 'addr_slots': aslots, 'slot_addr': slot_addr, 'addr_key': addr_key, 'slot_keys': F.normalize(torch.stack([keys[i] for sids in aslots for i in sids]).float(), dim=-1).to(device) if aslots else None, 'slot_keys_slot': [i for sids in aslots for i in sids], 'anc_keys': torch.stack([akeys[i] for sids in aslots for i in sids]).float().to(device) if aslots else None, 'ctx_keys': torch.stack([ckeys[i] for sids in aslots for i in sids]).float().to(device) if aslots else None, 'bank': bank, 'write_actions': dict(tape_w.counts), 'n_addresses': len(addrs), 'n_slots': n_slots}

def retrieve(pack, words, k, hop: str, item=None, subject_filter=False, hop_min=0.0, k_gap: float=0.0, index: str='main'):
    """Votes decide. The address proposes, and only where the votes had little to say.

    The first trigger was "no candidates", which never fired: votes always return k slots, they
    are simply the wrong ones. Silence on raw text is not an empty list, it is a weak one - so
    the hop fires on a low top score or a short list, and its slots are APPENDED behind the
    votes rather than replacing them. The exact channel keeps the front of the ranking.
    """
    if index == 'main':
        post, idfs = (pack['postings'], pack['idf'])
    else:
        if 'postings_probe' not in pack:
            raise KeyError('pack has no probe index: _stage280_raw_exam.py is out of date')
        post, idfs = (pack['postings_probe'], pack['idf_probe'])
    cands, sc = s271.vote(words, post, idfs, k)
    top = max((sc.get(c, 0.0) for c in cands), default=0.0)
    used_hop = False
    if hop == 'fp' and pack['addr_keys'] is not None and words and (len(cands) < k or top < hop_min):
        q = pack['bank'].fp(words).float()
        q = F.normalize(F.normalize(q, dim=-1).mean(0), dim=-1).to(pack['addr_keys'].device)
        mode = pack.get('addr_key', 'mean')
        if mode == 'two' and pack.get('anc_keys') is not None:
            row = int(torch.minimum(pack['anc_keys'] @ q, pack['ctx_keys'] @ q).argmax())
            best = pack['slot_addr'][pack['slot_keys_slot'][row]]
        elif mode == 'set' and pack.get('slot_keys') is not None:
            row = int((pack['slot_keys'] @ q).argmax())
            best = pack['slot_addr'][pack['slot_keys_slot'][row]]
        else:
            best = int((pack['addr_keys'] @ q).argmax())
        extra = [c for c in pack['addr_slots'][best] if c not in cands]
        if extra:
            floor = min((sc.get(c, 0.0) for c in cands), default=1.0)
            for c in extra:
                sc[c] = floor * 0.5
            cands = (cands + extra)[:k]
            used_hop = True
    if subject_filter and item is not None and cands:
        own = [c for c in cands if item['S'] in pack['texts_lc'][c]]
        cands = own or cands
    if k_gap > 0.0 and cands:
        best = max((sc.get(c, 0.0) for c in cands))
        if best > 0:
            cands = [c for c in cands if sc.get(c, 0.0) >= k_gap * best] or cands[:1]
    return (cands, {c: sc.get(c, 0.0) for c in cands}, used_hop)

def rollout(policy, model, char_table, tok, pack, item, pad_id, device, *, k, max_steps, max_reads, read_cost, wrong_cost, abstain_reward, subject_filter, hop, hop_min=0.0, k_gap=0.0, bc=False, greedy=True, teacher_only=False, bc_anchor=0.0, diag=False, teacher_fn=None):
    """278's rollout with one line changed: retrieval may take the ink's hop when words fail."""
    teach = teacher_fn or s278.teacher
    tape = pack['tape']
    s274._VALUE_OF = {i: v for i, v in enumerate(tape.values)}
    qtext = s271.CUE.format(S=item.get('query') or item['S'])
    qwords = context_words(qtext)
    transcript = qtext
    cands: list[int] = []
    last_read_words: list[str] = []
    seen_reads: set[int] = set()
    opened: list[str] = []
    losses, logps, ents, trace = ([], [], [], [])
    n_reads, answered, abstained, stalled = (0, None, False, True)
    prec, rec, hops = (float('nan'), float('nan'), 0)
    silent_first, n_cands_first = (None, 0)
    own_slots = set(item['slots'])
    tkw = dict(max_steps=max_steps, max_reads=max_reads, k=k)
    for _ in range(max_steps):
        if teacher_only:
            a = teach(cands=cands, seen_reads=seen_reads, opened_values=opened, n_reads=n_reads, cand_scores=pack.get('_sc'), **tkw)
        else:
            st = s278.state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads, opened, last_read_words, n_reads, pad_id, device, k, max_steps)
            if st is None:
                break
            logits, _ = st
            if bc:
                a = teach(cands=cands, seen_reads=seen_reads, opened_values=opened, n_reads=n_reads, cand_scores=pack.get('_sc'), **tkw)
                if not torch.isfinite(logits[a]) or logits[a] < -100000000.0:
                    break
                losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([a], device=device)))
            else:
                dist = torch.distributions.Categorical(logits=logits)
                a = int(logits.argmax()) if greedy else int(dist.sample())
                logps.append(dist.log_prob(torch.tensor(a, device=device)))
                ents.append(dist.entropy())
                if bc_anchor > 0.0:
                    a_t = teach(cands=cands, seen_reads=seen_reads, opened_values=opened, n_reads=n_reads, cand_scores=pack.get('_sc'), **tkw)
                    if torch.isfinite(logits[a_t]) and logits[a_t] > -100000000.0:
                        losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([a_t], device=device)))
        trace.append(s271.act_names(k)[a])
        if a in (s271.ASK_Q, s271.ASK_READ):
            words = qwords if a == s271.ASK_Q else last_read_words
            cands, sc, used = retrieve(pack, words, k, hop, item, subject_filter, hop_min, k_gap)
            hops += int(used)
            if silent_first is None:
                silent_first = used or not cands
            pack['_sc'] = sc
            if cands and math.isnan(prec):
                hit = sum((1 for c in cands if c in own_slots))
                prec = hit / len(cands)
                rec = hit / max(1, len(own_slots))
                n_cands_first = len(cands)
        elif a == 2 + 2 * k:
            abstained, stalled = (True, False)
            break
        elif a < 2 + k:
            i = a - 2
            if i >= len(cands):
                break
            slot = cands[i]
            transcript = (transcript + ' | ' + pack['texts'][slot])[-2000:]
            last_read_words = context_words(pack['texts'][slot], exclude=tape.values[slot])
            seen_reads.add(slot)
            opened.append(tape.values[slot])
            n_reads += 1
        else:
            i = a - 2 - k
            if i >= len(cands):
                break
            answered = tape.values[cands[i]]
            stalled = False
            break
    if abstained or answered is None:
        correct, abstained = (0, True)
        reward = 0.0 if stalled else abstain_reward
    else:
        correct = int(item['truth'] is not None and answered == item['truth'])
        reward = 1.0 if correct else -wrong_cost
    reward -= read_cost * n_reads
    return {'loss': torch.stack(losses).mean() if losses else torch.zeros((), device=device), 'logps': logps, 'entropy': ents, 'reward': reward, 'correct': correct, 'abstained': abstained, 'n_reads': n_reads, 'trace': trace, 'kind': item['kind'], 'answer_is_slot': answered is None or answered in set(tape.values), 'retrieval_precision': prec, 'witness_recall': rec, 'hops': hops, 'words_silent': bool(silent_first), 'n_cands': n_cands_first, 'stalled': bool(stalled and abstained), 'return_path': return_path(pack, item, answered) if diag else float('nan')}

def return_path(pack, item, answered) -> float:
    """Does the tape lead from the answer back to the subject? An observer, never a vote.

    282 gave the policy this as an action and it did not pay for its place: on the same tape the
    mind that never probes scores 0.704 held-out against 0.594 for the mind that does. But the
    question the probe asks outlives the exam that rejected it. Today an answer is a slot index,
    so G_answer_is_slot proves the answer came off the tape; when the answer stops being an
    index there is no such proof, and the only label-free replacement is this - ask the tape
    about the value and see whether some OTHER mention carries the subject and the value
    together. So it is measured here on every answer, costs the policy nothing, and is reported
    beside accuracy, where a reader can see whether it separates right answers from wrong ones
    before anything is built on it.
    """
    if answered is None or 'postings_probe' not in pack:
        return float('nan')
    words = context_words(answered) or [answered]
    posts = pack['postings_probe']
    scan = min((posts.get(w, ()) for w in words), key=len, default=())
    val_lc = answered.lower()
    seen = 0
    for c in scan:
        if item['S'] in pack['texts_lc'][c] and val_lc in pack['texts_lc'][c]:
            seen += 1
            if seen >= 2:
                return 1.0
    return 0.0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--bc-episodes', type=int, default=0)
    ap.add_argument('--rl-episodes', type=int, default=0)
    ap.add_argument('--tape-period', type=int, default=0)
    ap.add_argument('--addresses', type=int, default=0)
    ap.add_argument('--min-mentions', type=int, default=3)
    ap.add_argument('--max-items', type=int, default=0)
    ap.add_argument('--min-per-family', type=int, default=8, help='put this many of each family at the front of the exam. Noun-phrase anchors made addresses specific enough that exact ties are rare - the held-out tape had two - and abstention cannot be measured on two items. 0 leaves the natural frequency.')
    ap.add_argument('--address-tau', type=float, default=0.9)
    ap.add_argument('--address-overlap', type=int, default=2)
    ap.add_argument('--soft-match', type=float, default=0.0)
    ap.add_argument('--addr-key', choices=('two', 'set', 'mean'), default='set', help='how a query is matched to an address. two scores identity and situation separately and takes the minimum; set is MaxSim over the member keys; mean is the averaged key every number before 283 came from. new_pack reads this - without the flag the run died in argparse.')
    ap.add_argument('--run-tag', type=str, default='', help='prefix for the log, checkpoint and decision filenames, so two arms do not overwrite each other')
    ap.add_argument('--hop', choices=('none', 'fp'), default='fp')
    ap.add_argument('--k-gap', type=float, default=0.35, metavar='F', help='drop retrieved slots scoring below F x the top score. A fixed k fills the list with foreign slots on raw text and the teacher aggregates them; 0 restores the fixed-k behaviour.')
    ap.add_argument('--hop-min', type=float, default=1.0, help='fire the address hop when the best vote score falls below this. The first version fired on an empty candidate list, which never happens - votes always return k slots, just the wrong ones.')
    ap.add_argument('--topk', type=int, default=7)
    ap.add_argument('--max-steps', type=int, default=10)
    ap.add_argument('--max-reads', type=int, default=7)
    ap.add_argument('--read-cost', type=float, default=0.02)
    ap.add_argument('--wrong-cost', type=float, default=1.0)
    ap.add_argument('--abstain-reward', type=float, default=0.75)
    ap.add_argument('--entropy-bonus', type=float, default=0.01)
    ap.add_argument('--lr-policy', type=float, default=0.001)
    ap.add_argument('--lr-value', type=float, default=0.003)
    ap.add_argument('--lr-upper', type=float, default=3e-05)
    ap.add_argument('--value-coef', type=float, default=0.5)
    ap.add_argument('--bc-anchor', type=float, default=0.5)
    ap.add_argument('--no-hidden', action='store_true')
    ap.add_argument('--no-value-head', action='store_true')
    ap.add_argument('--subject-filter', choices=('off', 'on'), default='on', help="drop retrieved slots that do not mention the asked subject. Computable from the cue, and on raw text the teacher's majority rule is unsound without it - the smoke read two unrelated slots for every relevant one and the teacher stopped abstaining entirely.")
    ap.add_argument('--frozen-trunk', action='store_true')
    args = ap.parse_args()
    sf = args.subject_filter == 'on'
    use_v = not args.no_value_head
    s278.NO_HIDDEN = args.no_hidden
    global LOG_PATH
    tag = f'_{args.hop}' + ('_nohid' if args.no_hidden else '') + (f'_{args.run_tag}' if args.run_tag else '')
    LOG_PATH = RES / f'_stage280_log{tag}.txt'
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_bc = args.bc_episodes or (400 if args.smoke else 4000)
    n_rl = max(0, args.rl_episodes)
    tape_period = args.tape_period or (50 if args.smoke else 200)
    n_addr = args.addresses or (60 if args.smoke else 400)
    k = args.topk
    mode = 'none' if args.frozen_trunk else 'upper'
    log(f'Stage280 raw exam start {datetime.now(timezone.utc).isoformat()} device={device} hop={args.hop} no_hidden={args.no_hidden} bc={n_bc} rl={n_rl} k={k} mode={mode}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)['model'])
    if mode == 'none':
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        s213.set_train_mode(model, mode)
    arc0 = s271.arc_enc_hash(model)
    can = SelfModelXL(n_char, V).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(4000000 if args.smoke else 30000000)
    all_lines = [l.strip() for l in wtext.split('\n') if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][:3000 if args.smoke else 25000]
    eval_lines = all_lines[cut:][:1500 if args.smoke else 12000]
    log(f'  corpus split: {len(train_lines)} train lines / {len(eval_lines)} held-out lines')

    def new_pack(r, lines):
        return pack_from_corpus(lines, bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, n_addr=n_addr, min_mentions=args.min_mentions, tau=args.address_tau, overlap=args.address_overlap, soft_match=args.soft_match, max_items=args.max_items, min_per_family=args.min_per_family, addr_key=args.addr_key)
    pack = new_pack(rng, train_lines)
    fam = Counter((i['kind'] for i in pack['items']))
    log(f"  tape: {pack['n_addresses']} addresses, {pack['n_slots']} slots, write {json.dumps(pack['write_actions'])} | items {json.dumps(dict(fam))} ({time.time() - t0:.0f}s)")
    if len(pack['items']) < 8 or fam['tie'] == 0:
        log('  corpus produced too few items or no natural tie; raise --addresses')
        return 1
    d_hidden = 0 if args.no_hidden else 2 * (model.head.in_features // 2)
    policy = s278.PolicyV(d_hidden + s278.EXTRA, k, device) if use_v else s274.Policy(d_hidden + s278.EXTRA, k, device)
    live = [p for p in model.parameters() if p.requires_grad]
    groups = [{'params': [p for n_, p in policy.named_parameters() if not n_.startswith('v.')], 'lr': args.lr_policy}]
    if use_v:
        groups.append({'params': list(policy.v.parameters()), 'lr': args.lr_value})
    if live:
        groups.append({'params': live, 'lr': args.lr_upper})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    common = dict(k=k, max_steps=args.max_steps, max_reads=args.max_reads, read_cost=args.read_cost, wrong_cost=args.wrong_cost, abstain_reward=args.abstain_reward, subject_filter=sf, hop=args.hop, hop_min=args.hop_min, k_gap=args.k_gap)
    baseline, curve, v_err = (0.0, [], [])
    policy.train()
    model.train(mode != 'none')
    for ep in range(1, n_bc + 1):
        if (ep - 1) % tape_period == 0 and ep > 1:
            pack = new_pack(rng, train_lines)
        item = pack['items'][rng.randrange(len(pack['items']))]
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device, bc=True, **common)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_bc // 8) == 0:
            curve.append({'phase': 'bc', 'episode': ep, 'loss': float(out['loss']), 'kind': out['kind'], 'trace': out['trace']})
            log(f"  bc {ep}/{n_bc} loss={float(out['loss']):.4f} [{out['kind']}] {out['trace']}")
    for ep in range(1, n_rl + 1):
        if (ep - 1) % tape_period == 0 and ep > 1:
            pack = new_pack(rng, train_lines)
        item = pack['items'][rng.randrange(len(pack['items']))]
        if use_v:
            policy.collect = []
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device, greedy=False, bc_anchor=args.bc_anchor, **common)
        vals = policy.collect if use_v else None
        if use_v:
            policy.collect = None
        if not out['logps']:
            continue
        R = out['reward']
        if use_v and vals:
            vs = torch.stack(vals[:len(out['logps'])])
            v_loss = F.mse_loss(vs, torch.full_like(vs, R))
            v_err.append(float(v_loss))
            loss = -((R - vs).detach() * torch.stack(out['logps'])).sum() + args.value_coef * v_loss
        else:
            baseline = 0.99 * baseline + 0.01 * R
            loss = -(R - baseline) * torch.stack(out['logps']).sum()
        ent = torch.stack(out['entropy']).sum() if out['entropy'] else torch.zeros((), device=device)
        loss = loss - args.entropy_bonus * ent
        if args.bc_anchor > 0.0 and out['loss'].requires_grad:
            loss = loss + args.bc_anchor * out['loss']
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_rl // 8) == 0:
            curve.append({'phase': 'rl', 'episode': ep, 'v_mse': float(np.mean(v_err[-200:])) if v_err else None, 'kind': out['kind'], 'trace': out['trace']})
            log(f"  rl {ep}/{n_rl} v_mse={(np.mean(v_err[-200:]) if v_err else float('nan')):.3f} [{out['kind']}] {out['trace']}")
    policy.eval()
    model.eval()
    arc1 = s271.arc_enc_hash(model)

    @torch.no_grad()
    def evaluate(p):
        per = {f: defaultdict(list) for f in FAMILIES}
        tper = {f: defaultdict(list) for f in FAMILIES}
        slot_ok, traces, hops, silent, ncand, stalls = ([], [], [], [], [], [])
        ret, ret_c, ret_w = ([], [], [])
        for it in p['items']:
            o = rollout(policy, model, char_table, tok, p, it, pad_id, device, diag=True, **common)
            t = rollout(policy, model, char_table, tok, p, it, pad_id, device, teacher_only=True, **common)
            f = it['kind']
            per[f]['correct'].append(o['correct'])
            per[f]['abstain'].append(int(o['abstained']))
            per[f]['reads'].append(o['n_reads'])
            per[f]['reward'].append(o['reward'])
            if not math.isnan(o['retrieval_precision']):
                per[f]['prec'].append(o['retrieval_precision'])
                per[f]['rec'].append(o['witness_recall'])
            tper[f]['correct'].append(t['correct'])
            tper[f]['abstain'].append(int(t['abstained']))
            tper[f]['reward'].append(t['reward'])
            slot_ok.append(int(o['answer_is_slot']))
            hops.append(o['hops'])
            silent.append(int(o['words_silent']))
            ncand.append(o['n_cands'])
            stalls.append(int(o['stalled']))
            if not math.isnan(o['return_path']):
                ret.append(o['return_path'])
                (ret_c if o['correct'] else ret_w).append(o['return_path'])
            if len(traces) < 24:
                traces.append({'kind': f, 'S': it['S'], 'trace': o['trace'], 'correct': o['correct'], 'abstained': o['abstained'], 'hops': o['hops'], 'stalled': o['stalled']})
        m = lambda xs: float(np.mean(xs)) if xs else float('nan')
        out = {'answer_is_slot': m(slot_ok), 'traces': traces, 'reward_total': m([r for f in FAMILIES for r in per[f]['reward']]), 'teacher_reward_total': m([r for f in FAMILIES for r in tper[f]['reward']]), 'retrieval_precision': m([x for f in FAMILIES for x in per[f]['prec']]), 'witness_recall': m([x for f in FAMILIES for x in per[f]['rec']]), 'words_silent_rate': m(silent), 'hops_per_episode': m(hops), 'mean_candidates': m(ncand), 'stall_rate': m(stalls), 'return_path_rate': m(ret), 'return_path_when_correct': m(ret_c), 'return_path_when_wrong': m(ret_w), 'n_items': len(p['items'])}
        ac, an = (0, 0)
        for f in FAMILIES:
            n_ans = sum((1 for a in per[f]['abstain'] if not a))
            ac += sum(per[f]['correct'])
            an += n_ans
            out[f] = {'n': len(per[f]['abstain']), 'coverage': 1.0 - m(per[f]['abstain']), 'acc_answered': sum(per[f]['correct']) / n_ans if n_ans else float('nan'), 'abstain': m(per[f]['abstain']), 'mean_reads': m(per[f]['reads']), 'reward': m(per[f]['reward']), 'precision': m(per[f]['prec']), 'recall': m(per[f]['rec']), 'teacher_abstain': m(tper[f]['abstain']), 'teacher_acc_all': m(tper[f]['correct'])}
        out['coverage_all'] = an / max(1, len(p['items']))
        out['acc_answered_all'] = ac / max(1, an)
        return out
    train_eval = evaluate(pack)
    held = new_pack(random.Random(SEED + 99), eval_lines)
    log(f"  held-out tape: {held['n_addresses']} addresses, {held['n_slots']} slots, items {json.dumps(dict(Counter((i['kind'] for i in held['items']))))}")
    novel = evaluate(held)
    log(f"  HELD-OUT {json.dumps({kk: vv for kk, vv in novel.items() if kk != 'traces'})}")
    ceiling = novel['teacher_reward_total']
    floor_silence = args.abstain_reward
    g_teacher_usable = ceiling >= 0.5 * floor_silence
    g_arc = arc0 == arc1
    g_slot = novel['answer_is_slot'] >= 0.99
    g_families = all((novel[f]['n'] >= 4 for f in FAMILIES))
    g_reaches_teacher = novel['reward_total'] >= ceiling - 0.1
    g_answers_when_decidable = novel['clean']['abstain'] <= 0.25 and novel['decidable']['abstain'] <= 0.4
    g_abstain_on_tie = novel['tie']['abstain'] >= 0.6
    g_acc_when_answering = novel['acc_answered_all'] >= 0.6
    g_generalises = novel['reward_total'] >= train_eval['reward_total'] - 0.15
    g_hop_used = args.hop == 'none' or novel['words_silent_rate'] < 0.05 or novel['hops_per_episode'] > 0.0
    if not (g_arc and g_slot and g_families):
        overall = 'RAW_EXAM_INVALID'
    elif not g_teacher_usable:
        overall = 'TEACHER_UNUSABLE_ON_RAW'
    elif g_reaches_teacher and g_abstain_on_tie and g_answers_when_decidable and g_generalises:
        overall = 'RAW_EXAM_OK'
    elif g_reaches_teacher and g_generalises:
        overall = 'RAW_EXAM_REACHES_CEILING'
    elif g_abstain_on_tie or g_answers_when_decidable:
        overall = 'RAW_EXAM_PARTIAL'
    else:
        overall = 'RAW_EXAM_NO'
    ckpt_out = CKPT_OUT.with_name(f'{CKPT_OUT.stem}{tag}{CKPT_OUT.suffix}')
    torch.save({'policy': policy.state_dict(), 'model': model.state_dict(), 'stage': 280, 'hop': args.hop, 'no_hidden': args.no_hidden, 'min_per_family': args.min_per_family, 'arc_enc_hash': arc1}, ckpt_out)
    out = {'stage': 280, 'overall': overall, 'hop': args.hop, 'no_hidden': args.no_hidden, 'value_head': use_v, 'subject_filter': args.subject_filter, 'smoke': args.smoke, 'seed': SEED, 'bc_episodes': n_bc, 'rl_episodes': n_rl, 'topk': k, 'min_per_family': args.min_per_family, 'run_tag': args.run_tag, 'checkpoint': str(ckpt_out), 'address': {'tau': args.address_tau, 'overlap': args.address_overlap, 'soft_match': args.soft_match, 'min_mentions': args.min_mentions}, 'reward': {'correct': 1.0, 'wrong': -args.wrong_cost, 'abstain': args.abstain_reward, 'read': -args.read_cost}, 'train_tape_shape': {'addresses': pack['n_addresses'], 'slots': pack['n_slots'], 'write_actions': pack['write_actions'], 'families': dict(fam)}, 'held_out_tape_shape': {'addresses': held['n_addresses'], 'slots': held['n_slots'], 'write_actions': held['write_actions'], 'families': dict(Counter((i['kind'] for i in held['items'])))}, 'teacher_ceiling_reward': ceiling, 'gates': {'G_arc_enc_frozen': g_arc, 'G_answer_is_slot': g_slot, 'G_all_families_present': g_families, 'G_teacher_usable': g_teacher_usable, 'G_reaches_teacher': g_reaches_teacher, 'G_answers_when_decidable': g_answers_when_decidable, 'G_abstain_on_tie': g_abstain_on_tie, 'G_acc_when_answering': g_acc_when_answering, 'G_generalises_to_held_out': g_generalises, 'G_hop_covers_silence': g_hop_used}, 'train_tape': {kk: vv for kk, vv in train_eval.items() if kk != 'traces'}, 'held_out': novel, 'fp_version': s271.fp_version(), 'arc_enc_hash_before': arc0, 'arc_enc_hash_after': arc1, 'curve': curve, 'reference_278_templates': {'reward_total': 0.7875, 'teacher': 0.85, 'acc_answered_all': 1.0, 'tie_abstain': 1.0}, 'note': "Everything that survived its own measurement, on text nobody wrote for it. The tape is 279's write decision - fp addressing checked by shared words, common-noun anchors excluded - and the families are not declared: an address with one value is clean, a leader beating the runner-up is decidable, a tie at the top is a tie, and which questions exist is a property of the corpus. The mind is 278's policy with the value baseline, the BC anchor kept on through RL and the exhaustive teacher. Two things are expected to be worse than on templates and both are why this runs: words go silent where a template guaranteed the subject was shared, which --hop fp answers or fails to; and the teacher can simply be wrong, because on wikitext a majority may be a popular error, so teacher_reward_total is reported as the ceiling this exam has and the policy is scored against that rather than against 1.0. The held-out tape is built from a disjoint 30% of the corpus.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'stage280_decision{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    (RES / f'stage280_mini{tag}.md').write_text(f"# Stage 280 the exam on raw text (hop {args.hop}{(', no hidden' if args.no_hidden else '')})\n\n**{overall}**{(' · SMOKE' if args.smoke else '')} · teacher ceiling **{ceiling:.3f}**\n\n| family (held out) | n | coverage | acc answered | abstain | teacher acc | reads |\n|---|---:|---:|---:|---:|---:|---:|\n" + ''.join((f"| {f} | {novel[f]['n']} | {novel[f]['coverage']:.2f} | {novel[f]['acc_answered']:.2f} | {novel[f]['abstain']:.2f} | {novel[f]['teacher_acc_all']:.2f} | {novel[f]['mean_reads']:.1f} |\n" for f in FAMILIES)) + f"\n- policy {novel['reward_total']:.3f} vs teacher {ceiling:.3f}; coverage {novel['coverage_all']:.2f} at accuracy {novel['acc_answered_all']:.2f}\n- words silent on {novel['words_silent_rate']:.2f} of episodes, {novel['hops_per_episode']:.2f} ink hops per episode\n- retrieval precision {novel['retrieval_precision']:.2f}, recall {novel['witness_recall']:.2f}\n\n## Gates\n\n" + ''.join((f'- {kk}: **{vv}**\n' for kk, vv in out['gates'].items())), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())