"""
Stage 173 — Falsify: language signal vs orthographic trajectory.

Same frozen pen (170) + trained dyn (172). Do NOT retrain.
Eval dynamics on hold windows under corruptions that destroy content
but preserve / destroy form.

  python _stage173_curve_falsify.py
"""
from __future__ import annotations
import json
import random
import string
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import _stage170_curve_dynamics as s170
import _stage172_curve_scale as s172
RES = Path('results')
CKPT_PEN = Path('checkpoints/stage170_curve.pt')
CKPT_DYN = Path('checkpoints/stage172_curve.pt')
LOG = RES / '_stage173_log.txt'
DECISION = RES / 'stage173_falsify_decision.json'
MINI = RES / 'stage173_falsify_mini.md'
SEED = 173
N_WINDOWS = 96
K_REPORT = (1, 8, 16)

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)
LETTERS = string.ascii_letters

def corr_natural(s: str, rng: random.Random) -> str:
    return s

def corr_shuffle_all(s: str, rng: random.Random) -> str:
    chars = list(s)
    rng.shuffle(chars)
    return ''.join(chars)

def corr_shuffle_letters_keep_skeleton(s: str, rng: random.Random) -> str:
    """Keep spaces/punct/digits positions; shuffle only letters among themselves."""
    chars = list(s)
    idx = [i for i, c in enumerate(chars) if c.isalpha()]
    pool = [chars[i] for i in idx]
    rng.shuffle(pool)
    for i, ch in zip(idx, pool):
        chars[i] = ch
    return ''.join(chars)

def corr_random_letters_keep_skeleton(s: str, rng: random.Random) -> str:
    """Keep whitespace/punct skeleton; replace every letter/digit with random letter."""
    out = []
    for c in s:
        if c.isalpha():
            out.append(rng.choice(LETTERS))
        elif c.isdigit():
            out.append(rng.choice(LETTERS))
        else:
            out.append(c)
    return ''.join(out)

def corr_destroy_spaces(s: str, rng: random.Random) -> str:
    """Replace spaces with random letters — kill separator turns."""
    return ''.join((rng.choice(LETTERS) if c == ' ' else c for c in s))

def corr_only_spaces_punct(s: str, rng: random.Random) -> str:
    """Letters → 'a'; keep spaces/punct — almost pure skeleton rhythm."""
    return ''.join(('a' if c.isalnum() else c for c in s))

def corr_reverse(s: str, rng: random.Random) -> str:
    return s[::-1]
CORRUPTIONS = [('natural', corr_natural, 'real text'), ('shuffle_all', corr_shuffle_all, 'destroy order entirely'), ('shuffle_letters_keep_skel', corr_shuffle_letters_keep_skeleton, 'keep spaces/punct; scramble letters'), ('random_letters_keep_skel', corr_random_letters_keep_skeleton, 'same skeleton; random letters'), ('only_skel_flat_letters', corr_only_spaces_punct, "skeleton + all letters='a'"), ('destroy_spaces', corr_destroy_spaces, 'spaces → random letters'), ('reverse', corr_reverse, 'reversed string')]

@torch.no_grad()
def eval_condition(model, ids_np: np.ndarray, device, n_windows: int, seed: int) -> dict:
    """Same metrics spirit as 172 hold, on a full char-id stream."""
    model.pen.eval()
    model.dyn.eval()
    rng = random.Random(seed)
    n = len(ids_np)
    seq = s172.SEQ
    ctx = s172.CTX
    hold0 = int(0.85 * n)
    max_start = n - seq - 2
    if max_start <= hold0:
        hold0 = max(0, int(0.5 * n))
    mean_d = {k: [] for k in K_REPORT}
    for _ in range(24):
        s = rng.randint(0, max(1, hold0 - seq - 2))
        x = torch.tensor(ids_np[s:s + seq][None].astype(np.int64), device=device)
        z = model.encode(x)
        for k in K_REPORT:
            mean_d[k].append((z[:, k:] - z[:, :-k]).mean(dim=(0, 1)))
    mean_delta = {k: torch.stack(v, 0).mean(0) for k, v in mean_d.items()}
    acc = {k: {'cos': [], 'base_mean': [], 'base_copy': []} for k in K_REPORT}
    contrast_ok = []
    for _ in range(n_windows):
        s = hold0 + rng.randint(0, max(1, max_start - hold0))
        s = min(s, max_start)
        x = torch.tensor(ids_np[s:s + seq][None].astype(np.int64), device=device)
        z = model.encode(x)
        T = z.size(1)
        t = max(ctx, T - 1 - max(K_REPORT) - 1)
        z_ctx = z[:, t + 1 - ctx:t + 1]
        if z_ctx.size(1) < ctx:
            pad = z_ctx[:, :1].expand(1, ctx - z_ctx.size(1), -1)
            z_ctx = torch.cat([pad, z_ctx], dim=1)
        pred = model.dyn(z_ctx)
        for k in K_REPORT:
            if t + k >= T:
                continue
            delta = z[:, t + k] - z[:, t]
            d_hat = pred[f'delta_{k}']
            d_copy = z[:, t] - z[:, max(0, t - k)]
            acc[k]['cos'].append(float(F.cosine_similarity(d_hat, delta, dim=-1).mean()))
            acc[k]['base_mean'].append(float(F.cosine_similarity(mean_delta[k].unsqueeze(0), delta, dim=-1).mean()))
            acc[k]['base_copy'].append(float(F.cosine_similarity(d_copy, delta, dim=-1).mean()))
        L = 8
        true_f = F.normalize(z[:, t + 1:t + 1 + L].mean(dim=1), dim=-1)
        wrong_f = F.normalize(z[:, ctx:ctx + L].mean(dim=1), dim=-1)
        arc = pred['arc']
        m = min(arc.size(-1), true_f.size(-1))
        arc, true_f, wrong_f = (F.normalize(arc[:, :m], dim=-1), true_f[:, :m], wrong_f[:, :m])
        contrast_ok.append(float((F.cosine_similarity(arc, true_f) > F.cosine_similarity(arc, wrong_f)).float().mean()))

    def avg(xs):
        return sum(xs) / max(len(xs), 1)
    out = {'contrast_pref': avg(contrast_ok)}
    for k in K_REPORT:
        cos = avg(acc[k]['cos'])
        bm = avg(acc[k]['base_mean'])
        bc = avg(acc[k]['base_copy'])
        out[f'k{k}'] = {'cos_delta': cos, 'lift_mean': cos - bm, 'lift_copy': cos - bc}
    out['score'] = out['k1']['cos_delta'] + 0.5 * out['k16']['cos_delta'] + 0.3 * out['contrast_pref']
    return out

def encode_stream(text: str, stoi: dict) -> np.ndarray:
    return np.fromiter((stoi.get(c, 0) for c in text), dtype=np.int32, count=len(text))

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage173 falsify start {datetime.now(timezone.utc).isoformat()}')
    log('Q: language vs orthographic trajectory? (no retrain; frozen pen + 172 dyn)')
    if not CKPT_PEN.exists() or not CKPT_DYN.exists():
        log(f'FATAL missing ckpt pen={CKPT_PEN.exists()} dyn={CKPT_DYN.exists()}')
        return 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pen_blob = torch.load(CKPT_PEN, map_location='cpu', weights_only=False)
    dyn_blob = torch.load(CKPT_DYN, map_location='cpu', weights_only=False)
    stoi, itos = (pen_blob['stoi'], pen_blob['itos'])
    model = s172.ScaleModel(len(itos)).to(device)
    sd = dyn_blob['model']
    model.load_state_dict(sd, strict=False)
    pen_sd = {k[len('pen.'):]: v for k, v in pen_blob['model'].items() if k.startswith('pen.')}
    model.pen.load_state_dict(pen_sd, strict=True)
    for p in model.pen.parameters():
        p.requires_grad_(False)
    model.pen.eval()
    model.dyn.eval()
    log(f"loaded pen@170 + dyn@172 step={dyn_blob.get('step')} device={device}")
    raw = s170.load_corpus(max_chars=5000000)
    base = raw[1000000:3000000]
    log(f'base slice chars={len(base)}')
    rng_root = random.Random(SEED)
    results = {}
    natural_score = None
    for name, fn, desc in CORRUPTIONS:
        rng = random.Random(SEED + hash(name) % 10000)
        corrupted = fn(base, rng)
        ids = encode_stream(corrupted, stoi)
        metrics = eval_condition(model, ids, device, N_WINDOWS, seed=SEED + len(name))
        results[name] = {'desc': desc, **metrics}
        log(f"  [{name}] k1={metrics['k1']['cos_delta']:.3f} k8={metrics['k8']['cos_delta']:.3f} k16={metrics['k16']['cos_delta']:.3f} contrast={metrics['contrast_pref']:.3f} score={metrics['score']:.3f} | {desc}")
        if name == 'natural':
            natural_score = metrics['score']
    nat = results['natural']
    skel = results['random_letters_keep_skel']
    flat = results['only_skel_flat_letters']
    shuf = results['shuffle_all']
    nosp = results['destroy_spaces']
    shuf_let = results['shuffle_letters_keep_skel']

    def close(a, b, tol=0.05):
        return abs(a - b) <= tol
    nat_k1 = nat['k1']['cos_delta']
    gaps = {'gap_shuffle_all': nat_k1 - shuf['k1']['cos_delta'], 'gap_random_letters_skel': nat_k1 - skel['k1']['cos_delta'], 'gap_flat_skel': nat_k1 - flat['k1']['cos_delta'], 'gap_destroy_spaces': nat_k1 - nosp['k1']['cos_delta'], 'gap_shuffle_letters': nat_k1 - shuf_let['k1']['cos_delta']}
    skel_close = close(nat_k1, skel['k1']['cos_delta'], 0.07) or close(nat_k1, flat['k1']['cos_delta'], 0.07)
    shuffle_kills = gaps['gap_shuffle_all'] > 0.15
    spaces_matter = gaps['gap_destroy_spaces'] > 0.1
    letters_matter = gaps['gap_random_letters_skel'] > 0.1 and gaps['gap_shuffle_letters'] > 0.08
    if skel_close and spaces_matter and (not letters_matter):
        reading = 'MOSTLY_ORTHOGRAPHIC_SKELETON'
        detail = 'Dynamics nearly as good on skeleton/random letters; spaces matter; letter identity weak.'
    elif letters_matter and shuffle_kills:
        reading = 'LETTER_SEQUENCE_SENSITIVE'
        detail = 'Real letter order helps beyond skeleton — still may be orthographic n-grams, not semantics.'
    elif skel_close and (not spaces_matter):
        reading = 'GENERIC_SMOOTH_TRAJECTORY'
        detail = 'Even skeleton/space ablations stay high — likely easy local curve predictability.'
    else:
        reading = 'MIXED_FORM_SIGNAL'
        detail = 'Partial drops under corruptions; form-heavy with some letter-path sensitivity.'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'curve_falsify_ortho_vs_language', 'verdict_reading': reading, 'detail': detail, 'gaps_k1_vs_natural': gaps, 'conditions': results, 'natural_k1': nat_k1, 'flags': {'skel_close_to_natural': skel_close, 'shuffle_kills': shuffle_kills, 'spaces_matter': spaces_matter, 'letters_matter': letters_matter}, 'caveat': 'LETTER_SEQUENCE_SENSITIVE ≠ language understanding. It only means char-order of real text affects Δ predictability under this pen.', 'next': 'If MOSTLY_ORTHOGRAPHIC_*: need non-char pen or semantic probes. If LETTER_SEQUENCE_SENSITIVE: still need meaning falsify (paraphrase/cross-lingual) later.'}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    bullets = [f'reading `{reading}`', detail, f"natural k1={nat_k1:.3f} | random_skel k1={skel['k1']['cos_delta']:.3f} | flat_skel k1={flat['k1']['cos_delta']:.3f}", f"shuffle_all k1={shuf['k1']['cos_delta']:.3f} | destroy_spaces k1={nosp['k1']['cos_delta']:.3f}", f"gaps={{{', '.join((f'{k}={v:+.3f}' for k, v in gaps.items()))}}}", out['caveat']]
    MINI.write_text('\n'.join(['# Stage173 — orthography vs language falsify', '', f'**Reading:** `{reading}`', ''] + [f'- {b}' for b in bullets] + ['']), encoding='utf-8')
    log(f'[173] {reading}')
    log(detail)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())