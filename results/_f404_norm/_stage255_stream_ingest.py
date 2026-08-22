"""
Stage 255 тАФ Stream-ingest engine: chunked training with domain switching, bounded RAM.

North star: never hold the corpus in memory, never re-read it, switch domains mid-stream,
run for days on a small GPU, resume after a kill.

Loop per chunk:
    lines = reader.next_chunk()            # only this chunk is resident
    tape.append(gate(entities(lines)))     # knowledge grows, weights untouched
    trunk = joint_train(trunk, lines + reservoir_replay)   # CE + lam*CPC, ~1 epoch
    del lines                              # dropped; reservoir keeps a bounded sample

Domain switch is declared by --schedule (dom:chunks,...). Keys are always written with the
FROZEN canonical arc_enc. Queries use a trainable QueryAdapter (W_q) so understanding can
align reads to a growing bank without rewriting slot keys.

Probes (vs consumed tokens, not steps): exam next_tok, held-out CE for every domain seen
(carved once, never trained), probe-fact recall as the bank fills, uniformity, throughput.

  python _stage255_stream_ingest.py --smoke
  python _stage255_stream_ingest.py --schedule wiki:12 --chunk-lines 25000 --run-tag wiki12
  python _stage255_stream_ingest.py --schedule wiki:12 --lambda-admit --lambda-admit-alpha 0.015 --run-tag wiki12_lam
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import slot_query_words
from _tapelm_ext import DomainAdapter
RES = Path('results')
RUN = RES / 'stream255'
DECISION = RES / 'stage255_decision.json'
MINI = RES / 'stage255_mini.md'
LOG = RES / '_stage255_log.txt'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
SEED = 255
RECALL_SEED = SEED + 9000
LAM_DEFAULT = 0.2

def effective_lambda(base: float, n_admitted: int, alpha: float, dynamic: bool, entity_cap: int) -> float:
    """Lower CPC when ingest load is high; n_admitted normalized by entity_cap so lambda tracks fill rate."""
    if not dynamic or n_admitted <= 0 or entity_cap <= 0:
        return base
    load = float(n_admitted) / float(entity_cap)
    return base / (1.0 + alpha * load)
DOMAIN_PATHS = {'wiki': Path('data/_wikitext103_train.txt'), 'med': Path('data/_stage254_med.txt'), 'news': Path('data/_stage254_news.txt'), 'stories': Path('data/_tinystories_raw_100k.txt')}
MIN_LINE = 40

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def safe_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

class SegmentReader:
    """Walks a domain schedule, yielding chunks of lines; only the file handle is resident."""

    def __init__(self, schedule: list[tuple[str, int]], chunk_lines: int, positions: dict | None=None):
        self.schedule = schedule
        self.chunk_lines = chunk_lines
        self.positions = positions or {}
        self.seg_i = 0
        self.chunk_in_seg = 0

    def state(self) -> dict:
        return {'positions': self.positions, 'seg_i': self.seg_i, 'chunk_in_seg': self.chunk_in_seg}

    def load_state(self, st: dict) -> None:
        self.positions = st.get('positions', {})
        self.seg_i = st.get('seg_i', 0)
        self.chunk_in_seg = st.get('chunk_in_seg', 0)

    def next_chunk(self) -> tuple[str, list[str]] | None:
        while self.seg_i < len(self.schedule):
            dom, n_chunks = self.schedule[self.seg_i]
            if self.chunk_in_seg >= n_chunks:
                self.seg_i += 1
                self.chunk_in_seg = 0
                continue
            path = DOMAIN_PATHS[dom]
            pos = self.positions.get(dom, 0)
            lines: list[str] = []
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                f.seek(pos)
                while len(lines) < self.chunk_lines:
                    raw = f.readline()
                    if not raw:
                        break
                    s = raw.strip()
                    if len(s) >= MIN_LINE:
                        lines.append(s)
                self.positions[dom] = f.tell()
            if not lines:
                self.positions[dom] = 0
                self.seg_i += 1
                self.chunk_in_seg = 0
                continue
            self.chunk_in_seg += 1
            return (dom, lines)
        return None

class Tape:
    """Append-only canonical slot bank. fp16 on CPU; matmul in chunks so VRAM stays flat."""

    def __init__(self, dim: int, device):
        self.dim = dim
        self.device = device
        self._blocks: list[torch.Tensor] = []
        self._k_cache: torch.Tensor | None = None
        self.values: list[str] = []
        self.meta: list[dict] = []
        self._value_set: set[str] = set()
        self._ctxw: list[list[str]] = []
        self._postings = None

    def _sync_postings(self) -> None:
        from _inprint_glue import SlotPostings
        if len(self._ctxw) == len(self.values) and self._ctxw:
            self._postings = SlotPostings.from_ctxw(self._ctxw, torch.device('cpu'))
        else:
            self._postings = None

    @property
    def postings(self):
        return self._postings

    def __len__(self) -> int:
        return len(self.values)

    @property
    def K(self) -> torch.Tensor:
        if self._k_cache is None:
            self._k_cache = torch.cat(self._blocks, 0) if self._blocks else torch.zeros(0, self.dim, dtype=torch.float16)
        return self._k_cache

    def has_value(self, value: str) -> bool:
        return value in self._value_set

    def append(self, keys: torch.Tensor, values: list[str], meta: list[dict], ctxw: list[list[str]] | None=None) -> int:
        if not values:
            return 0
        self._blocks.append(keys.detach().to('cpu', torch.float16))
        self._k_cache = None
        self.values.extend(values)
        self.meta.extend(meta)
        self._value_set.update(values)
        if ctxw is not None:
            self._ctxw.extend(ctxw)
        self._sync_postings()
        return len(values)

    def scores(self, q: torch.Tensor, block: int=200000) -> torch.Tensor:
        K = self.K
        if K.numel() == 0:
            return torch.zeros(0)
        qq = q.detach().to('cpu', torch.float16)
        out = []
        for i in range(0, len(self.values), block):
            out.append((K[i:i + block] @ qq).float())
        return torch.cat(out) if out else torch.zeros(0)

    def max_score_for(self, q: torch.Tensor, value: str) -> float:
        idxs = [j for j, v in enumerate(self.values) if v == value]
        if not idxs:
            return -1.0
        return float(self.scores(q)[idxs].max())

    def nbytes(self) -> int:
        return self.K.numel() * 2

    def save(self, path: Path) -> None:
        torch.save({'K': self.K, 'values': self.values, 'meta': self.meta}, path)

    def load(self, path: Path) -> None:
        d = torch.load(path, map_location='cpu', weights_only=False)
        self._blocks = [d['K']] if d['K'].numel() else []
        self._k_cache = None
        self.values, self.meta = (d['values'], d['meta'])
        self._value_set = set(self.values)

class Reservoir:
    """Uniform sample of documents seen so far; bounded RAM, survives chunk deletion."""

    def __init__(self, cap: int, seed: int):
        self.cap = cap
        self.items: list[np.ndarray] = []
        self.n_seen = 0
        self.rng = random.Random(seed)

    def offer(self, doc: np.ndarray) -> None:
        self.n_seen += 1
        if len(self.items) < self.cap:
            self.items.append(doc)
        else:
            j = self.rng.randrange(self.n_seen)
            if j < self.cap:
                self.items[j] = doc

    def as_flat(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.items:
            return None
        offs = [0]
        for d in self.items:
            offs.append(offs[-1] + len(d))
        return (np.concatenate(self.items), np.asarray(offs, dtype=np.int64))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, n_seen=self.n_seen, cap=self.cap, items=np.array(self.items, dtype=object))

    def load(self, path: Path) -> None:
        d = np.load(path, allow_pickle=True)
        self.n_seen = int(d['n_seen'])
        self.cap = int(d['cap'])
        self.items = list(d['items'])

def chunk_to_flat(lines: list[str], tok: Tokenizer, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    return s213.build_flat_from_text('\n'.join(lines), tok, pad_id, max_lines=len(lines) + 8, min_line_len=20)

def merge_flats(a, b):
    """Concatenate two (flat, off) corpora into one doc-id space."""
    fa, oa = a
    fb, ob = b
    off = list(oa) + [int(oa[-1]) + int(x) for x in ob[1:]]
    return (np.concatenate([fa, fb]), np.asarray(off, dtype=np.int64))
CAP_WORD = re.compile('\\b([A-Z][a-z]{2,})\\b')
QPAIR_CAP = 40000

def ingest_entities(bank_can: FpBank, lines: list[str], cap: int, novelty: float, tape: Tape, dom: str, chunk_i: int, rng_scan: random.Random):
    """Real entities from the chunk -> canonical keys, novelty-gated so the tape does not bloat.

    Keys must use the SAME convention as probe facts (subject anchor + context). A context-only
    key is a generic direction that outscores anchored keys for any query and blinds the bank.
    """
    keys, vals, meta, qraw, ctxw_batch = ([], [], [], [], [])
    seen_local = set()
    line_order = list(range(len(lines)))
    rng_scan.shuffle(line_order)
    for li in line_order:
        ln = lines[li]
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in seen_local:
                continue
            lo, hi = (max(0, m.start() - 120), min(len(ln), m.end() + 120))
            c = bank_can.ctx_fp(ln[lo:hi], exclude=ent)
            if c is None:
                continue
            anchors = [w for w in CAP_WORD.findall(ln[lo:m.start()]) if w != ent]
            if not anchors:
                anchors = [w for w in CAP_WORD.findall(ln[lo:hi]) if w != ent]
            if not anchors:
                continue
            s_anchor = anchors[-1]
            seen_local.add(ent)
            a_fp = bank_can.fp([s_anchor])[0]
            keys.append(F.normalize(a_fp + c, dim=-1))
            cq = bank_can.ctx_fp(ln[lo:m.start()])
            qraw.append(F.normalize(a_fp + cq, dim=-1) if cq is not None else None)
            ctxw_batch.append(slot_query_words(ln[lo:hi], exclude=ent))
            vals.append(ent)
            meta.append({'domain': dom, 'chunk': chunk_i, 'kind': 'entity', 'anchor': s_anchor})
            if len(keys) >= cap:
                break
        if len(keys) >= cap:
            break
    if not keys:
        return (0, 0, [])
    Kn = torch.stack(keys, 0)
    kept_k, kept_v, kept_m, kept_ctxw, pairs = ([], [], [], [], [])
    dropped = 0
    for i in range(len(vals)):
        if tape.has_value(vals[i]):
            dropped += 1
            continue
        if kept_k:
            sim = float((torch.stack(kept_k, 0) @ Kn[i]).max())
            if sim > novelty:
                dropped += 1
                continue
        kept_k.append(Kn[i])
        kept_v.append(vals[i])
        kept_m.append(meta[i])
        kept_ctxw.append(ctxw_batch[i])
        if qraw[i] is not None:
            pairs.append({'q': qraw[i].detach().to('cpu', torch.float16), 'value': vals[i]})
    if kept_k:
        tape.append(torch.stack(kept_k, 0), kept_v, kept_m, ctxw=kept_ctxw)
    return (len(kept_v), dropped, pairs)

def make_probe_facts(bank_can: FpBank, values_pool: list[str], n: int, dom: str, rng: random.Random):
    """Controlled facts written to the tape only. Half fit W_q, half are held out for recall."""
    subs = [w for w in gen_fakes(set(values_pool), rng, n + 20) if len(w) >= 5][:n]
    facts, keys, ctxw_list = ([], [], [])
    for i, S in enumerate(subs):
        Vv = values_pool[rng.randrange(len(values_pool))]
        sent = f'{S} was appointed director of {Vv} in the {dom} chronicle of 1987 .'
        f = {'S': S, 'value': Vv, 'sent': sent, 'domain': dom, 'fid': f'{dom}_probe_{i}', 'wq_train': i % 2 == 0}
        k = bank_can.fp([S])[0]
        c = bank_can.ctx_fp(sent, exclude=Vv)
        keys.append(F.normalize(k + c, dim=-1) if c is not None else k)
        ctxw_list.append(slot_query_words(sent, exclude=Vv))
        facts.append(f)
    return (facts, torch.stack(keys, 0) if keys else torch.zeros(0, 256), ctxw_list)

def probe_bank_metrics(bank_q, tape: Tape, facts, all_values, W=None) -> dict:
    return L.tape_recall_metrics(facts, all_values, bank_q, tape.K, tape.values, RECALL_SEED, W_bwd=W, postings=tape.postings)

def parse_schedule(s: str) -> list[tuple[str, int]]:
    out = []
    for part in s.split(','):
        dom, _, n = part.partition(':')
        dom = dom.strip()
        if dom not in DOMAIN_PATHS:
            raise SystemExit(f'unknown domain {dom}; known: {list(DOMAIN_PATHS)}')
        out.append((dom, int(n or 1)))
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--schedule', type=str, default='wiki:6,med:4,news:4')
    ap.add_argument('--chunk-lines', type=int, default=0)
    ap.add_argument('--epochs-per-chunk', type=float, default=1.0)
    ap.add_argument('--reservoir', type=int, default=0)
    ap.add_argument('--replay-frac', type=float, default=0.2)
    ap.add_argument('--arc', choices=['frozen', 'adapt'], default='frozen')
    ap.add_argument('--entity-cap', type=int, default=0, help='entities ingested per chunk')
    ap.add_argument('--novelty', type=float, default=0.97)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--ckpt-every', type=int, default=2)
    ap.add_argument('--lambda-base', type=float, default=LAM_DEFAULT, help='CPC weight when dynamic=off or n_admit=0')
    ap.add_argument('--lambda-admit', action='store_true', help='lambda_eff = lambda_base / (1 + alpha * n_admitted_this_chunk)')
    ap.add_argument('--lambda-admit-alpha', type=float, default=0.35, help='at full entity_cap load: lambda*=1/(1+alpha)')
    ap.add_argument('--run-tag', type=str, default='', help='subdir under results/stream255/ for ckpts')
    ap.add_argument('--no-query-train', action='store_true', help='disable W_q contrastive training (ablation)')
    ap.add_argument('--query-steps', type=int, default=0, help='W_q steps per chunk (0=auto)')
    args = ap.parse_args()
    run_name = args.run_tag or ('smoke' if args.smoke else 'default')
    global RUN
    RUN = RES / 'stream255' / run_name
    RUN.mkdir(parents=True, exist_ok=True)
    lam_base = args.lambda_base
    if not args.resume:
        LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    schedule = parse_schedule('wiki:2,med:2' if args.smoke else args.schedule)
    chunk_lines = args.chunk_lines or (400 if args.smoke else 25000)
    reservoir_cap = args.reservoir or (200 if args.smoke else 4000)
    entity_cap = args.entity_cap or (40 if args.smoke else 400)
    n_probe_facts = 12 if args.smoke else 32
    n_exam = 40 if args.smoke else 120
    n_hold = 4 if args.smoke else 12
    arc_steps = 30 if args.smoke else 250
    w_steps = 30 if args.smoke else 350
    core_n = 40 if args.smoke else 200
    query_steps = args.query_steps or (25 if args.smoke else 150)
    use_query_train = not args.no_query_train
    log(f'Stage255 stream start {datetime.now(timezone.utc).isoformat()} device={device} schedule={schedule} chunk_lines={chunk_lines} arc={args.arc} resume={args.resume} run={run_name} lambda_base={lam_base} lambda_admit={args.lambda_admit} alpha={args.lambda_admit_alpha} query_train={use_query_train} query_steps={query_steps}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    Vtok = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, Vtok).to(device)
    model_can = SelfModelXL(n_char, Vtok).to(device)
    model_can.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)
    with DOMAIN_PATHS['wiki'].open('r', encoding='utf-8', errors='ignore') as f:
        seed_text = f.read(1000000 if args.smoke else 4000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(seed_text) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', seed_text) if len(w) <= 14)))[:core_n]
    F_can = s221.fp_matrix(bank_can, core)
    items = s251.load_exam_next(n_exam)
    tape = Tape(256, device)
    reservoir = Reservoir(reservoir_cap, SEED)
    reader = SegmentReader(schedule, chunk_lines)
    trunk = model_can
    probe_facts: dict[str, list] = {}
    holdouts: dict[str, list] = {}
    W_era: dict[str, object] = {}
    banks_q: dict[str, FpBank] = {}
    history: list[dict] = []
    tokens_train = 0
    tokens_unique = 0
    chunk_i = 0
    baseline_hold_ce: dict[str, float] = {}
    baseline_exam = s251.next_tok_acc(model_can, char_table, pad_id, items, device)
    W_query = L.init_query_adapter(device) if use_query_train else None
    q_pairs: list[dict] = []
    st_path = RUN / 'state.json'
    if args.resume and st_path.exists():
        st = json.loads(st_path.read_text(encoding='utf-8'))
        reader.load_state(st['reader'])
        tokens_train = st.get('tokens_train', st.get('tokens_seen', 0))
        tokens_unique = st.get('tokens_unique', 0)
        chunk_i, history = (st['chunk_i'], st['history'])
        probe_facts = {k: v for k, v in st['probe_facts'].items()}
        baseline_hold_ce = st.get('baseline_hold_ce', {})
        baseline_exam = st.get('baseline_exam', baseline_exam)
        tape.load(RUN / 'tape.pt')
        res_path = RUN / 'reservoir.npz'
        if res_path.exists():
            reservoir.load(res_path)
        trunk = SelfModelXL(n_char, Vtok).to(device)
        trunk.load_state_dict(torch.load(RUN / 'trunk.pt', map_location=device, weights_only=False)['model'])
        trunk.eval()
        for p in trunk.parameters():
            p.requires_grad_(False)
        hb = torch.load(RUN / 'holdouts.pt', map_location='cpu', weights_only=False)
        holdouts = {k: [t for t in v] for k, v in hb.items()}
        w_path = RUN / 'w_era.pt'
        if w_path.exists() and args.arc == 'adapt':
            wst = torch.load(w_path, map_location=device, weights_only=False)
            for dom, sd in wst.items():
                W = DomainAdapter(256).to(device)
                W.load_state_dict(sd)
                W_era[dom] = W
                sp = RUN / f'shift_{dom}.pt'
                if sp.exists():
                    sh = SelfModelXL(n_char, Vtok).to(device)
                    sh.load_state_dict(torch.load(sp, map_location=device, weights_only=False)['model'])
                    sh.eval()
                    for p in sh.parameters():
                        p.requires_grad_(False)
                    banks_q[dom] = FpBank(sh, stoi, device)
        qp = RUN / 'query_adapter.pt'
        if W_query is not None and qp.exists():
            qck = torch.load(qp, map_location=device, weights_only=False)
            W_query.load_state_dict(qck['W_q_stream'] if isinstance(qck, dict) and 'W_q_stream' in qck else qck)
        pp = RUN / 'qpairs.pt'
        if pp.exists():
            d = torch.load(pp, map_location='cpu', weights_only=False)
            q_pairs = [{'q': q, 'value': v} for q, v in zip(d['q'], d['values'])]
        log(f'resumed at chunk {chunk_i}, unique_tok={tokens_unique}, train_tok={tokens_train}, tape {len(tape)}')
    stop_file = RUN / 'STOP'
    while True:
        if stop_file.exists():
            log('STOP file present тАФ halting cleanly')
            break
        nxt = reader.next_chunk()
        if nxt is None:
            log('schedule exhausted')
            break
        dom, lines = nxt
        chunk_i += 1
        flat_c, off_c = chunk_to_flat(lines, tok, pad_id)
        n_docs = len(off_c) - 1
        if n_docs < 8:
            log(f'chunk {chunk_i} ({dom}) too small ({n_docs} docs) тАФ skipped')
            continue
        first_time = dom not in holdouts
        if first_time:
            n_h = max(2, int(n_docs * 0.05))
            hb = []
            for d in range(n_docs - n_h, n_docs):
                seg = flat_c[off_c[d]:off_c[d + 1]][:MAX_ARCS]
                row = np.full((1, MAX_ARCS), pad_id, np.int64)
                row[0, :len(seg)] = seg
                hb.append(torch.from_numpy(row))
            holdouts[dom] = hb[:n_hold] if len(hb) >= n_hold else hb
            baseline_hold_ce[dom] = s252.fixed_hold_ce(model_can, holdouts[dom], char_table, pad_id, device)
            train_docs = list(range(0, n_docs - n_h))
            pf, pk, pctx = make_probe_facts(bank_can, values_pool, n_probe_facts, dom, rng)
            probe_facts[dom] = pf
            tape.append(pk, [f['value'] for f in pf], [{'domain': dom, 'kind': 'probe'} for _ in pf], ctxw=pctx)
            log(f'  [{dom}] holdout={len(holdouts[dom])} probe_facts={len(pf)} P1_hold_ce={baseline_hold_ce[dom]:.3f}')
        else:
            train_docs = list(range(0, n_docs))
        scan_rng = random.Random(SEED + chunk_i * 7919)
        n_new, n_drop, new_pairs = ingest_entities(bank_can, lines, entity_cap, args.novelty, tape, dom, chunk_i, scan_rng)
        q_pairs.extend(new_pairs)
        if len(q_pairs) > QPAIR_CAP:
            q_pairs = random.Random(SEED + chunk_i).sample(q_pairs, QPAIR_CAP)
        n_admit_lam = n_new + (len(probe_facts.get(dom, [])) if first_time else 0)
        lam_eff = effective_lambda(lam_base, n_admit_lam, args.lambda_admit_alpha, args.lambda_admit, entity_cap)
        if args.arc == 'adapt' and dom not in W_era:
            shifted = s221.finetune_arc_enc(model_can, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + chunk_i)
            bq = FpBank(shifted, stoi, device)
            W, align = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bq, core), F_can, rng, w_steps, device)
            W_era[dom], banks_q[dom] = (W, bq)
            torch.save({'model': shifted.state_dict()}, RUN / f'shift_{dom}.pt')
            log(f'  [{dom}] W era align={align:.3f}')
        flat_t, off_t = (flat_c, off_c)
        res = reservoir.as_flat()
        if res is not None and args.replay_frac > 0:
            flat_t, off_t = merge_flats((flat_c, off_c), res)
            n_res = len(res[1]) - 1
            n_mix = min(n_res, int(len(train_docs) * args.replay_frac / max(1e-06, 1 - args.replay_frac)))
            pool = list(range(n_docs, n_docs + n_res))
            train_docs = train_docs + random.Random(SEED + chunk_i).sample(pool, n_mix)
        budget = int(args.epochs_per_chunk * int(off_c[-1]))
        hold_all = [b for d in holdouts for b in holdouts[d]]
        trunk, meta = s252.train_joint(trunk, flat_t, off_t, char_table, pad_id, device, budget, lam_eff, SEED + 1000 + chunk_i, f'c{chunk_i}:{dom}', train_docs, hold_all, items[:24], early_stop=False, n_probes=2)
        tokens_unique += int(off_c[-1])
        tokens_train += meta['tokens_ce']
        q_loss = float('nan')
        if W_query is not None and q_pairs:
            q_loss = L.train_query_adapter_pairs(W_query, q_pairs, tape.K, tape.values, device, query_steps, SEED + chunk_i)
        doc_ids = list(range(n_docs))
        if len(doc_ids) > 800:
            step = max(1, len(doc_ids) // 800)
            doc_ids = doc_ids[::step]
        for d in doc_ids:
            reservoir.offer(flat_c[off_c[d]:off_c[d + 1]].copy())
        del lines, flat_c, off_c, flat_t, off_t
        all_vals = list(dict.fromkeys([f['value'] for fs in probe_facts.values() for f in fs] + tape.values))
        W_read = W_query if W_query is not None else W_era.get(dom)
        eval_facts = {d: fs for d, fs in probe_facts.items() if fs}
        bank_frozen = {d: probe_bank_metrics(banks_q.get(d, bank_can), tape, eval_facts[d], all_vals, W=None) for d in eval_facts}
        bank_adapt = {d: probe_bank_metrics(banks_q.get(d, bank_can), tape, eval_facts[d], all_vals, W=W_read) for d in eval_facts}
        row = {'chunk': chunk_i, 'domain': dom, 'tokens_unique': tokens_unique, 'tokens_train': tokens_train, 'tape_slots': len(tape), 'tape_mb': tape.nbytes() / 1000000.0, 'entities_added': n_new, 'entities_dropped': n_drop, 'n_admit_lambda': n_admit_lam, 'lambda_eff': lam_eff, 'query_loss': q_loss, 'exam_next_tok': s251.next_tok_acc(trunk, char_table, pad_id, items, device), 'hold_ce': {d: s252.fixed_hold_ce(trunk, holdouts[d], char_table, pad_id, device) for d in holdouts}, 'probe_bank_frozen': bank_frozen, 'probe_bank': bank_adapt, 'wall_s': time.time() - t0}
        history.append(row)
        rec_s = ' '.join((f"{d}:ce={row['hold_ce'][d]:.2f}" + (f"/top1={bank_adapt[d]['top1']:.2f}(f={bank_frozen[d]['top1']:.2f})/mrr={bank_adapt[d]['mrr']:.2f}" if d in bank_adapt else '') for d in holdouts))
        log(f"  chunk {chunk_i} [{dom}] uniq_tok={tokens_unique} train_tok={tokens_train} slots={len(tape)} (+{n_new}/-{n_drop}) lam={lam_eff:.4f} (n_admit={n_admit_lam}) q_loss={q_loss:.3f} exam={row['exam_next_tok']:.3f} {rec_s} tape={row['tape_mb']:.1f}MB ({row['wall_s']:.0f}s)")
        if chunk_i % args.ckpt_every == 0:
            safe_torch_save({'model': trunk.state_dict()}, RUN / 'trunk.pt')
            tape.save(RUN / 'tape.pt')
            reservoir.save(RUN / 'reservoir.npz')
            safe_torch_save({k: v for k, v in holdouts.items()}, RUN / 'holdouts.pt')
            if W_era:
                safe_torch_save({k: v.state_dict() for k, v in W_era.items()}, RUN / 'w_era.pt')
            if W_query is not None:
                safe_torch_save({'W_q_stream': W_query.state_dict(), 'W_query': W_query.state_dict()}, RUN / 'query_adapter.pt')
            if q_pairs:
                safe_torch_save({'q': torch.stack([p['q'] for p in q_pairs]), 'values': [p['value'] for p in q_pairs]}, RUN / 'qpairs.pt')
            st_path.write_text(json.dumps({'reader': reader.state(), 'tokens_unique': tokens_unique, 'tokens_train': tokens_train, 'chunk_i': chunk_i, 'history': history, 'probe_facts': probe_facts, 'baseline_hold_ce': baseline_hold_ce, 'baseline_exam': baseline_exam}, indent=2), encoding='utf-8')
    if not history:
        log('no chunks processed')
        return 1
    first, last = (history[0], history[-1])
    doms = list(last['hold_ce'].keys())
    first_ce = {}
    for r in history:
        for d, v in r['hold_ce'].items():
            first_ce.setdefault(d, v)
    forget_vs_first = {d: last['hold_ce'][d] - first_ce[d] for d in doms}
    forget_vs_p1 = {d: last['hold_ce'][d] - baseline_hold_ce.get(d, first_ce[d]) for d in doms}
    rec_curve = [(r['tape_slots'], min((m['top1'] for m in r['probe_bank'].values()))) for r in history if r.get('probe_bank')]
    rec_frozen_curve = [(r['tape_slots'], min((m['top1'] for m in r.get('probe_bank_frozen', {}).values()))) for r in history if r.get('probe_bank_frozen')]
    mrr_curve = [(r['tape_slots'], min((m['mrr'] for m in r['probe_bank'].values()))) for r in history if r.get('probe_bank')]
    last_bank = last.get('probe_bank', {})
    last_frozen = last.get('probe_bank_frozen', {})
    last_top1 = min((v['top1'] for v in last_bank.values())) if last_bank else 0.0
    last_mrr = min((v['mrr'] for v in last_bank.values())) if last_bank else 0.0
    last_top1_frozen = min((v['top1'] for v in last_frozen.values())) if last_frozen else last_top1
    g_stream = len(history) >= 2
    g_no_forget_p1 = max(forget_vs_p1.values()) <= 0.15
    g_peak_regress = max(forget_vs_first.values()) <= 0.15
    g_grow = last['exam_next_tok'] >= baseline_exam - 0.01
    g_recall_adapt = last_top1 >= last_top1_frozen + 0.02 or last_mrr >= 0.1
    g_recall_floor = last_top1 >= 0.1 if use_query_train else last_top1_frozen >= 0.04
    g_bounded = last['tape_mb'] < 2000
    if g_stream and g_no_forget_p1 and g_grow and g_recall_adapt and g_recall_floor:
        overall = 'STREAM_INGEST_OK'
    elif g_stream and g_no_forget_p1 and g_grow and (g_recall_adapt or g_recall_floor):
        overall = 'STREAM_INGEST_PARTIAL'
    else:
        overall = 'STREAM_INGEST_NO'
    out = {'stage': 255, 'overall': overall, 'lambda_base': lam_base, 'lambda_admit': args.lambda_admit, 'lambda_admit_alpha': args.lambda_admit_alpha, 'run_tag': run_name, 'chunk_lines': chunk_lines, 'entity_cap': entity_cap, 'query_train': use_query_train, 'query_steps': query_steps, 'arc_mode': args.arc, 'epochs_per_chunk': args.epochs_per_chunk, 'replay_frac': args.replay_frac, 'reservoir_cap': reservoir_cap, 'gates': {'G_streamed': g_stream, 'G_no_forget_vs_P1': g_no_forget_p1, 'G_peak_hold_regress': g_peak_regress, 'G_understanding_holds': g_grow, 'G_recall_query_beats_frozen': g_recall_adapt, 'G_recall_adapt_top1_floor': g_recall_floor, 'G_tape_bounded': g_bounded}, 'summary': {'chunks': len(history), 'tokens_unique': last.get('tokens_unique', 0), 'tokens_train': last.get('tokens_train', 0), 'tape_slots': last['tape_slots'], 'tape_mb': last['tape_mb'], 'baseline_exam': baseline_exam, 'exam_first_chunk': history[0]['exam_next_tok'], 'exam_last': last['exam_next_tok'], 'forget_hold_ce_vs_first_chunk': forget_vs_first, 'forget_hold_ce_vs_P1': forget_vs_p1, 'baseline_hold_ce': baseline_hold_ce, 'recall_top1_vs_bank': rec_curve, 'recall_top1_frozen_vs_bank': rec_frozen_curve, 'recall_mrr_vs_bank': mrr_curve, 'recall_final_top1': last_top1, 'recall_final_top1_frozen': last_top1_frozen, 'recall_final_mrr': last_mrr}, 'history': history, 'note': 'Canonical frozen keys on tape; trainable W_q (QueryAdapter) aligns queries to keys — understanding→read, not re-indexing. W_q trains on ingested-entity contrastive pairs per chunk; recall gates score held-out probe facts only. Hold CE vs P1 is primary no-forget gate. Recall gates use W_q-adapted top1/MRR, not frozen-query-only.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    dec_path = RES / (f'stage255_decision_{run_name}.json' if run_name != 'default' else 'stage255_decision.json')
    mini_path = RES / (f'stage255_mini_{run_name}.md' if run_name != 'default' else 'stage255_mini.md')
    dec_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    mini_path.write_text(f"# Stage 255 stream ingest\n\n**{overall}** chunks={len(history)} uniq_tok={last.get('tokens_unique', 0)} train_tok={last.get('tokens_train', 0)} slots={last['tape_slots']} ({last['tape_mb']:.1f} MB)\n\n- exam P1={baseline_exam:.3f} last={last['exam_next_tok']:.3f}\n- forget vs P1: {json.dumps({k: round(v, 3) for k, v in forget_vs_p1.items()})}\n- recall top1 vs bank: {rec_curve}\n- recall mrr vs bank: {mrr_curve}\n", encoding='utf-8')
    log(json.dumps({'overall': overall, 'chunks': len(history), 'slots': last['tape_slots']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())