"""
Stage 246 — Multi-domain sequential retention (full TapeLM stack vs GPT).

Curriculum (default order):
  wiki → tinystories → med → news(ag-like)
Each phase trains for --steps (smoke small; full default 3000; use --steps 30000 for paper-scale).

TapeLM (product-faithful, full stack):
  - frozen canonical P1 arc_enc + shared slot bank
  - per domain: learn W_bwd (qmap), train head_domain (225, arc frozen), write domain facts into slots
  - after each phase: for EVERY past domain measure
      gen = window next_tok with matched head
      mem = slot recall with matched W @ shifted query encoder

GPT (parametric continuum):
  - sequential CE on the same domain flats
  - after each phase: for EVERY past domain measure window CE→PPL and planted-fact recall

Output: retention matrix phase × domain × {tape_gen, tape_mem, gpt_ppl, gpt_fact}.

  python _stage246_domain_curriculum.py [--smoke] [--steps N]
"""
from __future__ import annotations
import argparse
import copy
import json
import math
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
import _stage224_far_shift as s224
import _stage225_family_fork as s225
import _stage24x_lib as L
from _stage191_night import MICRO, PAD, SelfModelXL, load_data, sample_windows
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import load_gpt
from _tapelm_ext import DomainAdapter
RES = Path('results')
DATA = Path('data')
DECISION = RES / 'stage246_decision.json'
MINI = RES / 'stage246_mini.md'
LOG = RES / '_stage246_log.txt'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
STORIES = Path('data/external_tinystories_100k_85.txt')
SEED = 246
NEWS_RE = re.compile('\\b(said|reuters|reported|minister|election|president|government|officials|according to|announced|senate|parliament|campaign)\\b', re.I)
DOMAINS = ('wiki', 'stories', 'med', 'news')

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def ensure_news(max_lines: int) -> str:
    path = DATA / '_stage246_news_corpus.txt'
    if path.exists() and path.stat().st_size > 5000:
        return path.read_text(encoding='utf-8')
    lines: list[str] = []
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) < 48 or not NEWS_RE.search(line):
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
    if len(lines) < 80:
        raise RuntimeError('news corpus too small')
    path.write_text('\n'.join(lines), encoding='utf-8')
    log(f'news corpus lines={len(lines)}')
    return path.read_text(encoding='utf-8')

def domain_text(name: str, max_lines: int, smoke: bool, rng: random.Random) -> str:
    if name == 'wiki':
        with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
            return f.read(2000000 if smoke else 12000000)
    if name == 'stories':
        return STORIES.read_text(encoding='utf-8', errors='ignore')
    if name == 'med':
        return s224.ensure_med_corpus(max_lines=max_lines)
    if name == 'news':
        return ensure_news(max_lines)
    raise KeyError(name)

@torch.no_grad()
def window_ce(model, flat, off, char_table, pad_id, device, rng, n_batches=12) -> float:
    losses = []
    for _ in range(n_batches):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        pad = ids == pad_id
        logits, _, _ = model.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        if valid.sum() == 0:
            continue
        losses.append(float(F.cross_entropy(logits[:, :-1][valid], target[valid])))
    return float(np.mean(losses)) if losses else float('nan')

@torch.no_grad()
def gpt_window_ce(gm, flat, off, pad_id, device, rng, n_batches=12) -> float:
    losses = []
    for _ in range(n_batches):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        out = gm(input_ids=ids, labels=ids)
        losses.append(float(out.loss))
    return float(np.mean(losses)) if losses else float('nan')

def train_gpt_domain(gm, flat, off, pad_id, device, steps, lr, seed, log_every, tag):
    opt = torch.optim.AdamW(gm.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    gm.train()
    for step in range(1, steps + 1):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        loss = gm(input_ids=ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0:
            log(f'  gpt {tag} step {step}: loss={float(loss):.3f}')
    gm.eval()

def train_head(model, flat, off, char_table, pad_id, device, steps, seed):
    return s225.train_upper(model, flat, off, char_table, pad_id, device, steps, seed)

def plant_domain_facts(bank, values_pool, n_facts, rng, domain: str, seed_off: int):
    r = random.Random(SEED + seed_off)
    subs = [w for w in gen_fakes(set(values_pool), r, n_facts + 20) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[(i + seed_off) % len(values_pool)]
        sent = f'{S} was appointed director of {Vv} in the {domain} chronicle of 1987 .'
        facts.append({'S': S, 'value': Vv, 'sent': sent, 'domain': domain, 'fid': f'{domain}_{i}'})
    keys, vals = ([], [])
    for f in facts:
        k = bank.fp([f['S']])[0]
        c = bank.ctx_fp(f['sent'], exclude=f['value'])
        keys.append(F.normalize(k + c, dim=-1) if c is not None else k)
        vals.append(f['value'])
    return (facts, torch.stack(keys, 0), vals)

def mem_recall(facts, bank_q, K, V, W_bwd, seed: int) -> float:
    all_v = list(dict.fromkeys(V))
    return L.tape_recall(facts, all_v + all_v, bank_q, K, V, seed, W_bwd=W_bwd)

def gpt_fact_acc(gm, tok, pad_id, facts, device, seed: int) -> float:
    all_v = [f['value'] for f in facts]
    return L.gpt_fact_recall(gm, tok, pad_id, facts, all_v + all_v, device, seed)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0, help='steps/domain (0 => smoke 120 / full 3000)')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = args.steps or (120 if args.smoke else 3000)
    arc_steps = 40 if args.smoke else min(400, max(80, steps // 4))
    w_steps = 40 if args.smoke else min(600, max(100, steps // 3))
    head_steps = 40 if args.smoke else min(800, max(100, steps // 2))
    gpt_lr = 0.0003
    n_facts = 8 if args.smoke else 24
    max_lines = 250 if args.smoke else 8000
    core_n = 50 if args.smoke else 300
    eval_batches = 6 if args.smoke else 16
    log_every = max(20, steps // 4)
    log(f'Stage246 start {datetime.now(timezone.utc).isoformat()} device={device} steps/domain={steps} domains={list(DOMAINS)}')
    from _stage191_night import load_data
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
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_bit = f.read(4000000 if args.smoke else 20000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wiki_bit) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', wiki_bit) if len(w) <= 14)))[:core_n]
    F_can = s221.fp_matrix(bank_can, core)
    corpora = {}
    flats = {}
    for i, d in enumerate(DOMAINS):
        text = domain_text(d, max_lines, args.smoke, rng)
        flat, off = s213.build_flat_from_text(text, tok, pad_id, max_lines=max_lines, min_line_len=20)
        corpora[d] = text
        flats[d] = (flat, off)
        log(f'domain {d}: tokens={len(flat)} docs={len(off) - 1}')
    gm = copy.deepcopy(load_gpt(device))
    gm.eval()
    heads = {}
    W_bwd = {}
    banks_q = {}
    slot_K = {}
    slot_V = {}
    facts_d = {}
    matrix = []
    for phase_i, d in enumerate(DOMAINS):
        log(f'\n=== PHASE {phase_i + 1}/{len(DOMAINS)}: {d} ({steps} steps) ===')
        flat, off = flats[d]
        model_shift = s221.finetune_arc_enc(model_can, flat, off, char_table, pad_id, device, arc_steps, SEED + 10 + phase_i)
        bank_q = FpBank(model_shift, stoi, device)
        banks_q[d] = bank_q
        W, align = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_q, core), F_can, rng, w_steps, device)
        W_bwd[d] = W
        heads[d] = train_head(model_can, flat, off, char_table, pad_id, device, head_steps, SEED + 50 + phase_i)
        facts, K, V = plant_domain_facts(bank_can, values_pool, n_facts, rng, d, seed_off=100 + phase_i * 17)
        facts_d[d], slot_K[d], slot_V[d] = (facts, K, V)
        log(f'  tape {d}: W_align={align:.3f} facts={len(facts)}')
        train_gpt_domain(gm, flat, off, pad_id, device, steps, gpt_lr, SEED + 200 + phase_i, log_every, d)
        seen = DOMAINS[:phase_i + 1]
        row = {'after_phase': d, 'domains': {}}
        for e in seen:
            flat_e, off_e = flats[e]
            tape_ce = window_ce(heads[e], flat_e, off_e, char_table, pad_id, device, random.Random(SEED + 7), eval_batches)
            tape_nt = s225.window_next_tok_acc(heads[e], flat_e, off_e, char_table, pad_id, device, random.Random(SEED + 8), eval_batches)
            tape_mem = mem_recall(facts_d[e], banks_q[e], slot_K[e], slot_V[e], W_bwd[e], SEED + phase_i)
            g_ce = gpt_window_ce(gm, flat_e, off_e, pad_id, device, random.Random(SEED + 9), eval_batches)
            g_ppl = math.exp(min(20.0, g_ce)) if g_ce == g_ce else float('nan')
            g_fact = gpt_fact_acc(gm, tok, pad_id, facts_d[e], device, SEED + phase_i)
            row['domains'][e] = {'tape_ce': tape_ce, 'tape_next_tok': tape_nt, 'tape_mem': tape_mem, 'gpt_ce': g_ce, 'gpt_ppl': g_ppl, 'gpt_fact': g_fact}
            log(f'  eval[{e}]: tape_nt={tape_nt:.3f} tape_mem={tape_mem:.3f} | gpt_ppl={g_ppl:.2f} gpt_fact={g_fact:.3f}')
        matrix.append(row)
        log(f'phase {d} done ({time.time() - t0:.0f}s)')
    final = matrix[-1]['domains']
    first_seen = {}
    for row in matrix:
        for e, m in row['domains'].items():
            if e not in first_seen:
                first_seen[e] = m
    drops = {}
    for e in DOMAINS:
        if e not in final or e not in first_seen:
            continue
        drops[e] = {'tape_mem_drop': first_seen[e]['tape_mem'] - final[e]['tape_mem'], 'tape_nt_drop': first_seen[e]['tape_next_tok'] - final[e]['tape_next_tok'], 'gpt_fact_drop': first_seen[e]['gpt_fact'] - final[e]['gpt_fact'], 'gpt_ppl_rise': final[e]['gpt_ppl'] - first_seen[e]['gpt_ppl']}
    early = DOMAINS[0]
    g_tape_mem = final.get(early, {}).get('tape_mem', 0) >= 0.7
    g_tape_gen = final.get(early, {}).get('tape_next_tok', 0) >= 0.5
    g_gpt_fact_drop = drops.get(early, {}).get('gpt_fact_drop', 0) >= 0.1
    g_gpt_ppl_rise = drops.get(early, {}).get('gpt_ppl_rise', 0) >= 0.5
    g_gap_mem = final.get(early, {}).get('tape_mem', 0) - final.get(early, {}).get('gpt_fact', 0) >= 0.15
    if g_tape_mem and g_tape_gen and (g_gpt_fact_drop or g_gpt_ppl_rise) and g_gap_mem:
        overall = 'DOMAIN_CURRICULUM_OK'
    elif g_tape_mem and (g_gpt_fact_drop or g_gap_mem):
        overall = 'DOMAIN_CURRICULUM_PARTIAL'
    else:
        overall = 'DOMAIN_CURRICULUM_NO'
    out = {'stage': 246, 'overall': overall, 'steps_per_domain': steps, 'domains': list(DOMAINS), 'gates': {'G_tape_keeps_wiki_mem_ge_0p70': g_tape_mem, 'G_tape_keeps_wiki_gen_ge_0p50': g_tape_gen, 'G_gpt_wiki_fact_drop_ge_0p10': g_gpt_fact_drop, 'G_gpt_wiki_ppl_rise_ge_0p5': g_gpt_ppl_rise, 'G_gap_mem_ge_0p15': g_gap_mem}, 'matrix': matrix, 'drops_first_to_final': drops, 'note': 'TapeLM = frozen P1 + per-domain {W, head, slots}. GPT = single weight trajectory. news = AG-like wiki filter (not HF AG News download). Pass --steps 30000 for paper-scale curriculum.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    lines = [f'# Stage 246 domain curriculum\n\n**{overall}** steps/domain={steps}\n']
    lines.append('| after \\ domain | ' + ' | '.join(DOMAINS) + ' |')
    lines.append('|' + '---|' * (len(DOMAINS) + 1))
    for row in matrix:
        cells = []
        for e in DOMAINS:
            m = row['domains'].get(e)
            if not m:
                cells.append('-')
            else:
                cells.append(f"tMem{m['tape_mem']:.2f}/gPPL{m['gpt_ppl']:.1f}")
        lines.append(f"| {row['after_phase']} | " + ' | '.join(cells) + ' |')
    MINI.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log(json.dumps({'overall': overall, 'drops': drops}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())