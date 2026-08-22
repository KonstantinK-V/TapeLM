"""297/298: write the tape by frames - exact count, no tau.

A place is a hole whose left+right frame the corpus writes at least twice. Width is the maximum
that still repeats. Fillers are literal. No parser, no stop-words, no merge threshold.

Interpolation (cosine) does NOT live here: each frame is its own address. Frame-frame cosine is
handed to Phi as the cos edge at read time, so "is this the same place" is the mind's job.

    pack_from_frames(...) -> same schema as pack_from_corpus, plus:
      frame_mode, frame_nfill[slot], frame_nfill_max, frame_fps[slot]
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from _inprint_glue import TapeView
from _tape_index import context_words
_HEADING = re.compile('^=+\\s*.*?\\s*=+$')
_AT = re.compile('\\s*@-@\\s*')

def hygiene_text(text: str) -> str:
    """Drop heading lines and BPE joint marks; collapse <unk>. Not a grammar."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or _HEADING.match(s):
            continue
        s = _AT.sub('-', s).replace('<unk>', ' ')
        if s.strip():
            out.append(s)
    return '\n'.join(out)

def tokenize(text: str) -> list[str]:
    return [t for t in text.split() if t]

def _frame_key(tokens, i, w) -> str:
    return ' '.join(tokens[i - w:i]) + '\x00' + ' '.join(tokens[i + 1:i + 1 + w])

def _repeating_at_width(tokens, w: int) -> set[str]:
    """Keys whose frame appears at least twice. Hapaxes never enter a list - that was the OOM."""
    n = len(tokens)
    first: dict[str, int] = {}
    multi: set[str] = set()
    for i in range(w, n - w):
        k = _frame_key(tokens, i, w)
        if k in multi:
            continue
        if k in first:
            del first[k]
            multi.add(k)
        else:
            first[k] = i
    return multi

def pack_from_frames(lines, *, bank, tok, pad_id, device, rng, n_addr, min_mentions, frame_max: int=12, **_ignored):
    """Build a pack whose addresses ARE frames. Signature mirrors pack_from_corpus extras."""
    if isinstance(lines, str):
        tokens = tokenize(hygiene_text(lines))
    else:
        tokens = []
        for ln in lines:
            t = hygiene_text(ln)
            if t:
                tokens.extend(tokenize(t))
    n = len(tokens)
    if n < 3:
        return _empty_pack(bank, tok, pad_id, device)
    repeat: list[set[str]] = []
    for w in range(1, frame_max + 1):
        repeat.append(_repeating_at_width(tokens, w))
    by_addr: dict[str, list[int]] = defaultdict(list)
    meta = {}
    for i in range(n):
        chosen = None
        for w in range(min(frame_max, i, n - i - 1), 0, -1):
            k = _frame_key(tokens, i, w)
            if k in repeat[w - 1]:
                left = tuple(tokens[i - w:i])
                right = tuple(tokens[i + 1:i + 1 + w])
                chosen = (w, left, right)
                break
        if chosen is None:
            continue
        w, left, right = chosen
        addr = _addr_key(w, left, right)
        by_addr[addr].append(i)
        meta[addr] = (w, left, right)
    del repeat
    addrs = [a for a, pos in by_addr.items() if len(pos) >= min_mentions]

    def _rank(a: str):
        pos = by_addr[a]
        k = len({tokens[i] for i in pos})
        m = len(pos)
        band = 0 if 2 <= k <= 20 else 1 if k == 1 else 2
        return (band, 0 if m <= 128 else 1, -min(m, 64), a)
    addrs.sort(key=_rank)
    addrs = addrs[:max(1, n_addr)]
    by_addr = {a: by_addr[a] for a in addrs}
    if not addrs:
        return _empty_pack(bank, tok, pad_id, device)
    row_cap = int(_ignored.get('frame_row_cap', 48))
    keys, akeys, ckeys, vals, texts, straddr = ([], [], [], [], [], [])
    frame_nfill, frame_fps = ([], [])
    slot_of_pos = {}
    flat_slots = []
    for addr in addrs:
        w, left, right = meta[addr]
        positions = by_addr[addr]
        nfill = len({tokens[i] for i in positions})
        if len(positions) > row_cap:
            step = len(positions) / row_cap
            positions = [positions[int(j * step)] for j in range(row_cap)]
        fl = bank.fp(list(left)).float()
        fr = bank.fp(list(right)).float()
        ffp = F.normalize(F.normalize(fl, dim=-1).mean(0) + F.normalize(fr, dim=-1).mean(0), dim=-1)
        for i in positions:
            filler = tokens[i]
            ctx = ' '.join(left) + ' ___ ' + ' '.join(right)
            c = bank.ctx_fp(ctx, exclude=filler)
            k = ffp
            keys.append(k)
            akeys.append(ffp)
            ckeys.append(F.normalize(c, dim=-1) if c is not None else ffp)
            vals.append(filler)
            texts.append(ctx)
            straddr.append(addr)
            frame_nfill.append(nfill)
            frame_fps.append(ffp)
            slot_of_pos[i] = len(vals) - 1
            flat_slots.append(len(vals) - 1)
    by_addr_slots = {a: [slot_of_pos[i] for i in by_addr[a] if i in slot_of_pos] for a in addrs}
    postings: dict[str, list[int]] = defaultdict(list)
    postings_probe: dict[str, list[int]] = defaultdict(list)
    for cid, t in enumerate(texts):
        for w in context_words(t, exclude=vals[cid]):
            postings[w].append(cid)
        for w in context_words(t):
            postings_probe[w].append(cid)
    n_slots = len(vals)
    idf = {w: math.log(max(2.0, n_slots / max(1, len(postings[w])))) for w in postings}
    idf_probe = {w: math.log(max(2.0, n_slots / max(1, len(postings_probe[w])))) for w in postings_probe}
    items = []
    for addr in addrs:
        sids = by_addr_slots.get(addr, [])
        if len(sids) < 2:
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
        w, left, right = meta[addr]
        subj = addr
        query = ' '.join(left) + ' ' + ' '.join(right)
        items.append({'S': subj, 'query': query, 'truth': truth, 'slots': sids, 'kind': kind, 'address': addr, 'frame_w': w, 'frame_nfill': len(cnt)})
    rng.shuffle(items)
    akey, aslots, slot_addr = ([], [], [0] * len(vals))
    for ai, addr in enumerate(addrs):
        sids = by_addr_slots.get(addr, [])
        if sids:
            akey.append(F.normalize(torch.stack([keys[i] for i in sids]).mean(0), dim=-1))
            aslots.append(sids)
            for i in sids:
                slot_addr[i] = len(aslots) - 1
    nfill_max = max(frame_nfill) if frame_nfill else 1
    return {'tape': TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id), 'texts': texts, 'items': items, 'postings': postings, 'idf': idf, 'straddr': straddr, 'postings_probe': postings_probe, 'idf_probe': idf_probe, 'texts_lc': [t.lower() for t in texts], 'addr_keys': F.normalize(torch.stack(akey).float(), dim=-1).to(device) if akey else None, 'addr_slots': aslots, 'slot_addr': slot_addr, 'addr_key': 'set', 'slot_keys': F.normalize(torch.stack([keys[i] for sids in aslots for i in sids]).float(), dim=-1).to(device) if aslots else None, 'slot_keys_slot': [i for sids in aslots for i in sids], 'anc_keys': torch.stack([akeys[i] for sids in aslots for i in sids]).float().to(device) if aslots else None, 'ctx_keys': torch.stack([ckeys[i] for sids in aslots for i in sids]).float().to(device) if aslots else None, 'bank': bank, 'write_actions': {'FRAME': n_slots, 'CONFIRM': sum((1 for c in Counter(straddr).values() if c > 1))}, 'n_addresses': len(addrs), 'n_slots': n_slots, 'frame_mode': True, 'frame_nfill': frame_nfill, 'frame_nfill_max': nfill_max, 'frame_fps': frame_fps}

def _addr_key(w, left, right) -> str:
    return f"f{w}:{' '.join(left)}|{' '.join(right)}"

def _empty_pack(bank, tok, pad_id, device):
    z = torch.zeros(1, 32, device=device)
    return {'tape': TapeView(z, ['?'], tok, pad_id), 'texts': ['?'], 'items': [], 'postings': {}, 'idf': {}, 'straddr': ['?'], 'postings_probe': {}, 'idf_probe': {}, 'texts_lc': ['?'], 'addr_keys': None, 'addr_slots': [], 'slot_addr': [0], 'addr_key': 'set', 'slot_keys': None, 'slot_keys_slot': [], 'anc_keys': None, 'ctx_keys': None, 'bank': bank, 'write_actions': {}, 'n_addresses': 0, 'n_slots': 0, 'frame_mode': True, 'frame_nfill': [1], 'frame_nfill_max': 1, 'frame_fps': [z[0]]}