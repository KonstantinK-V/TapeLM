"""
Residual audit for Stage 7.8c WordComposer.

Asks: after you remove letter-basin content from a word fingerprint,
what remains — and do related morphological diffs cluster?

Residuals
  R_span  = normalize( compose - proj onto span(letter basins in word) )
  R_mean  = normalize( compose - normalize(mean letter basins) )
  delta   = normalize( compose(form) - compose(stem) )   # morphology residual

Loads:
  checkpoints/alphabet_full_frozen.pt
  checkpoints/word_memory_78c.pt

Writes:
  results/residual_audit_report.txt
  results/residual_audit.json

Run:
  python residual_audit.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from train import (
    ALL_WORDS,
    ALPHABET_PATH,
    CHARS,
    HOLD_WORDS,
    LETTER2ID,
    MAX_WORD_LEN,
    MEMORY_CKPT_PATH,
    RELATED_FORMS,
    RESULTS_DIR,
    TRAIN_WORDS,
    Config,
    LetterEncoder,
    SoftWordMemory,
    WordComposer,
    encode_words_text,
    load_frozen_alphabet,
    pick_device,
)


# Explicit morphology / compound probes beyond RELATED_FORMS
EXTRA_PAIRS = [
    ("jump", "jumping", "ing"),
    ("walk", "walking", "ing"),
    ("read", "reading", "ing"),
    ("write", "writing", "ing"),
    ("quick", "quickly", "ly"),
    ("slow", "slowly", "ly"),
    ("happy", "happily", "ly"),
    ("friend", "friendship", "ship"),
    ("card", "cards", "s"),
    ("book", "books", "s"),
]

COMPOUND_PROBES = [
    ("friend", "ship", "friendship"),
    ("sun", "shine", "sunshine"),
    ("moon", "light", "moonlight"),
    ("play", "ground", "playground"),
    ("apple", "sauce", "applesauce"),
    ("bird", "house", "birdhouse"),
    ("note", "book", "notebook"),
    ("back", "pack", "backpack"),
]


def load_78c(device: torch.device):
    if not MEMORY_CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing {MEMORY_CKPT_PATH}")
    ckpt = torch.load(MEMORY_CKPT_PATH, map_location=device, weights_only=False)
    dim = int(ckpt.get("dim", Config.dim))
    max_len = int(ckpt.get("max_word_len", MAX_WORD_LEN))
    encoder = LetterEncoder(len(CHARS), dim).to(device)
    composer = WordComposer(dim, max_len).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    composer.load_state_dict(ckpt["composer"])
    encoder.eval()
    composer.eval()
    memory = SoftWordMemory(dim, float(ckpt.get("ema_momentum", 0.85)))
    if "memory" in ckpt:
        memory.load_state_dict(ckpt["memory"])
    basins = load_frozen_alphabet(device)
    return encoder, composer, basins, memory, ckpt


@torch.no_grad()
def compose_word(composer, encoder, word: str, device):
    fp = composer(*encode_words_text(encoder, [word], device)).squeeze(0)
    return F.normalize(fp, dim=-1)


@torch.no_grad()
def letter_basin_stack(basins, word: str):
    codebook = basins.normalized()
    return torch.stack([codebook[LETTER2ID[c]] for c in word], 0)


def project_onto_span(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Stable projection of vec onto row-span of basis (n, d) via truncated SVD."""
    # basis: (n, d) letter basins; work in ambient dim with B^T (d, n)
    b = basis.T.float()  # (d, n)
    try:
        u, s, vh = torch.linalg.svd(b, full_matrices=False)
    except RuntimeError:
        return torch.zeros_like(vec)
    # keep singular values above relative floor
    floor = float(s[0].clamp(min=1e-8)) * 1e-4 if s.numel() else 1e-6
    keep = s > floor
    if not bool(keep.any()):
        return torch.zeros_like(vec)
    u_k = u[:, keep]
    return u_k @ (u_k.T @ vec.float())


def residuals_for_word(composer, encoder, basins, word: str, device):
    composed = compose_word(composer, encoder, word, device)
    letter_rows = letter_basin_stack(basins, word)
    proj = project_onto_span(composed, letter_rows)
    diff = composed - proj
    r_span = F.normalize(diff, dim=-1)
    mean_letters = F.normalize(letter_rows.mean(0), dim=-1)
    r_mean = F.normalize(composed - mean_letters, dim=-1)
    codebook = basins.normalized()
    traces = {ch: float((composed * codebook[LETTER2ID[ch]]).sum()) for ch in sorted(set(word))}
    proj_n = torch.norm(proj)
    proj_cos = float((F.normalize(proj, dim=-1) * composed).sum()) if float(proj_n) > 1e-6 else 0.0
    return {
        "word": word,
        "compose": composed,
        "proj_span": proj,
        "r_span": r_span,
        "r_mean": r_mean,
        "proj_cos": proj_cos,
        "mean_cos": float((mean_letters * composed).sum()),
        "r_span_norm_pre": float(torch.norm(diff)),
        "min_trace": min(traces.values()) if traces else 0.0,
        "traces": traces,
    }


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum())


def pairwise_mean_cos(vecs: list[torch.Tensor]) -> float | None:
    if len(vecs) < 2:
        return None
    s = n = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            s += cos(vecs[i], vecs[j])
            n += 1
    return s / n


def infer_suffix_tag(stem: str, form: str) -> str:
    if form.startswith(stem) and len(form) > len(stem):
        return form[len(stem) :] or "id"
    # common stem trim: happy->happily, write->writing
    for k in range(len(stem), 0, -1):
        if form.startswith(stem[:k]):
            return f"~{stem[k:]}+{form[k:]}"
    return "irreg"


def build_pair_list():
    pairs = []
    for a, b in RELATED_FORMS:
        if a in ALL_WORDS and b in ALL_WORDS:
            pairs.append((a, b, infer_suffix_tag(a, b)))
    for a, b, tag in EXTRA_PAIRS:
        if a in ALL_WORDS and b in ALL_WORDS:
            pairs.append((a, b, tag))
    # dedupe by (a,b)
    seen = set()
    out = []
    for a, b, tag in pairs:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b, tag))
    return out


@torch.no_grad()
def run_audit():
    device = pick_device()
    encoder, composer, basins, memory, ckpt = load_78c(device)
    print(f"Device: {device}")
    print(f"Checkpoint: {MEMORY_CKPT_PATH}")
    print(f"Alphabet: {ALPHABET_PATH}")

    word_cache = {}
    for w in ALL_WORDS:
        word_cache[w] = residuals_for_word(composer, encoder, basins, w, device)

    pairs = build_pair_list()
    pair_rows = []
    by_tag: dict[str, list[torch.Tensor]] = defaultdict(list)
    by_tag_rspan: dict[str, list[torch.Tensor]] = defaultdict(list)

    for stem, form, tag in pairs:
        cs = word_cache[stem]["compose"]
        cf = word_cache[form]["compose"]
        delta = F.normalize(cf - cs, dim=-1)
        # residual-space delta: does morphology live outside letter span?
        d_r = F.normalize(word_cache[form]["r_span"] - word_cache[stem]["r_span"], dim=-1)
        row = {
            "stem": stem,
            "form": form,
            "tag": tag,
            "compose_cos": cos(cs, cf),
            "delta_norm": float(torch.norm(cf - cs)),
            "delta_vs_stem_r": cos(delta, word_cache[stem]["r_span"]),
            "r_span_cos": cos(word_cache[stem]["r_span"], word_cache[form]["r_span"]),
            "stem_proj_cos": word_cache[stem]["proj_cos"],
            "form_proj_cos": word_cache[form]["proj_cos"],
            "stem_min_trace": word_cache[stem]["min_trace"],
            "form_min_trace": word_cache[form]["min_trace"],
            "stem_set": "hold" if stem in HOLD_WORDS else "train",
            "form_set": "hold" if form in HOLD_WORDS else "train",
        }
        pair_rows.append(row)
        by_tag[tag].append(delta.cpu())
        by_tag_rspan[tag].append(d_r.cpu())

    tag_summary = []
    for tag in sorted(by_tag.keys(), key=lambda t: (-len(by_tag[t]), t)):
        tag_summary.append(
            {
                "tag": tag,
                "n": len(by_tag[tag]),
                "delta_cluster": pairwise_mean_cos(by_tag[tag]),
                "rspan_delta_cluster": pairwise_mean_cos(by_tag_rspan[tag]),
            }
        )

    # Cross-tag confusion: mean cos of deltas between different tags (sample)
    cross = []
    tags = [t for t in by_tag if len(by_tag[t]) >= 2]
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1 :]:
            sims = []
            for u in by_tag[t1]:
                for v in by_tag[t2]:
                    sims.append(cos(u, v))
            if sims:
                cross.append(
                    {
                        "tag_a": t1,
                        "tag_b": t2,
                        "mean_cross_cos": sum(sims) / len(sims),
                    }
                )
    cross.sort(key=lambda r: -r["mean_cross_cos"])

    compound_rows = []
    for left, right, whole in COMPOUND_PROBES:
        if whole not in word_cache:
            continue
        # only if parts are real words we can compose; else use letter-basin compose of parts
        parts_ok = []
        part_fps = []
        for p in (left, right):
            if p in word_cache:
                part_fps.append(word_cache[p]["compose"])
                parts_ok.append(p)
            else:
                # synthetic: compose from frozen letter basins only
                codebook = basins.normalized()
                fps = torch.stack([codebook[LETTER2ID[c]] for c in p], 0).unsqueeze(0)
                lengths = torch.tensor([len(p)], device=device)
                part_fps.append(F.normalize(composer(fps, lengths).squeeze(0), dim=-1))
                parts_ok.append(f"{p}*")
        mean_parts = F.normalize(torch.stack(part_fps).mean(0), dim=-1)
        # additive bind approx
        add_parts = F.normalize(part_fps[0] + part_fps[1], dim=-1)
        cw = word_cache[whole]["compose"]
        compound_rows.append(
            {
                "whole": whole,
                "parts": "+".join(parts_ok),
                "cos_mean_parts": cos(cw, mean_parts),
                "cos_sum_parts": cos(cw, add_parts),
                "whole_proj_cos": word_cache[whole]["proj_cos"],
                "whole_min_trace": word_cache[whole]["min_trace"],
                "set": "hold" if whole in HOLD_WORDS else "train",
            }
        )

    # Global stats
    train_proj = [word_cache[w]["proj_cos"] for w in TRAIN_WORDS]
    hold_proj = [word_cache[w]["proj_cos"] for w in HOLD_WORDS]
    train_r = [word_cache[w]["r_span"] for w in TRAIN_WORDS]
    hold_r = [word_cache[w]["r_span"] for w in HOLD_WORDS]

    # Do HOLD residuals look like TRAIN residuals? (collapse check)
    rng = torch.Generator().manual_seed(0)
    def sample_mean_pair_cos(vecs, n_pairs=200):
        if len(vecs) < 2:
            return None
        s = 0.0
        for _ in range(n_pairs):
            i = int(torch.randint(0, len(vecs), (1,), generator=rng).item())
            j = int(torch.randint(0, len(vecs), (1,), generator=rng).item())
            while j == i:
                j = int(torch.randint(0, len(vecs), (1,), generator=rng).item())
            s += cos(vecs[i], vecs[j])
        return s / n_pairs

    payload = {
        "stage": "residual_audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(MEMORY_CKPT_PATH),
        "alphabet": str(ALPHABET_PATH),
        "summary": {
            "train_mean_proj_cos": sum(train_proj) / len(train_proj),
            "hold_mean_proj_cos": sum(hold_proj) / len(hold_proj),
            "train_rspan_self_sim": sample_mean_pair_cos(train_r),
            "hold_rspan_self_sim": sample_mean_pair_cos(hold_r),
            "n_pairs": len(pair_rows),
            "n_tags": len(tag_summary),
        },
        "tag_clusters": tag_summary,
        "cross_tag": cross[:12],
        "pairs": pair_rows,
        "compounds": compound_rows,
        "hold_words": [
            {
                "word": w,
                "proj_cos": word_cache[w]["proj_cos"],
                "mean_cos": word_cache[w]["mean_cos"],
                "min_trace": word_cache[w]["min_trace"],
                "r_span_norm_pre": word_cache[w]["r_span_norm_pre"],
                "mem_count": memory.count(w),
            }
            for w in HOLD_WORDS
        ],
    }

    report = format_report(payload)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "residual_audit_report.txt"
    json_path = RESULTS_DIR / "residual_audit.json"
    # JSON: drop nothing heavy — compose tensors not included
    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"Saved: {report_path}")
    print(f"Saved: {json_path}")
    return payload


def format_report(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        "SOTE residual audit (compose minus letter basins)",
        f"timestamp: {payload['timestamp']}",
        f"checkpoint: {payload['checkpoint']}",
        "",
        "How to read:",
        "- proj_cos ~1  => word fp lives mostly in letter-basin span (little residual)",
        "- proj_cos low => residual carries order/morphology (interesting)",
        "- delta_cluster high within tag, low across tags => morphology pattern discovered",
        "",
        f"Train mean proj_cos: {s['train_mean_proj_cos']:.3f}",
        f"HOLD  mean proj_cos: {s['hold_mean_proj_cos']:.3f}",
        f"Train R_span self-sim (sample): {s['train_rspan_self_sim']:.3f}",
        f"HOLD  R_span self-sim (sample): {s['hold_rspan_self_sim']:.3f}",
        "",
        "Morphology delta clusters (compose(form)-compose(stem)):",
    ]
    for row in payload["tag_clusters"]:
        dc = row["delta_cluster"]
        rc = row["rspan_delta_cluster"]
        dc_s = f"{dc:.3f}" if dc is not None else "n/a"
        rc_s = f"{rc:.3f}" if rc is not None else "n/a"
        lines.append(f"  [{row['tag']:8s}] n={row['n']:2d}  delta_cluster={dc_s}  rspan_delta={rc_s}")

    lines.append("")
    lines.append("Top cross-tag delta similarity (want LOW if tags are distinct):")
    for row in payload["cross_tag"][:8]:
        lines.append(
            f"  {row['tag_a']:8s} vs {row['tag_b']:8s}  mean_cos={row['mean_cross_cos']:.3f}"
        )

    lines.append("")
    lines.append("Pair details (stem -> form):")
    for row in sorted(payload["pairs"], key=lambda r: (r["tag"], r["form"])):
        mark = "H" if row["form_set"] == "hold" or row["stem_set"] == "hold" else "T"
        lines.append(
            f"  [{mark}] {row['stem']:10s}->{row['form']:12s} tag={row['tag']:6s} "
            f"cos={row['compose_cos']:.2f} d|={row['delta_norm']:.2f} "
            f"Rcos={row['r_span_cos']:.2f} tr={row['stem_min_trace']:.2f}/{row['form_min_trace']:.2f}"
        )

    lines.append("")
    lines.append("Compound probes (whole vs parts):")
    for row in payload["compounds"]:
        lines.append(
            f"  {row['whole']:12s} = {row['parts']:20s} "
            f"mean={row['cos_mean_parts']:.2f} sum={row['cos_sum_parts']:.2f} "
            f"proj={row['whole_proj_cos']:.2f} tr={row['whole_min_trace']:.2f}"
        )

    lines.append("")
    lines.append("HOLD word residuals:")
    for row in payload["hold_words"]:
        lines.append(
            f"  {row['word']:12s} proj={row['proj_cos']:.2f} mean={row['mean_cos']:.2f} "
            f"tr={row['min_trace']:.2f} |r|={row['r_span_norm_pre']:.2f} mem_n={row['mem_count']}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_audit()
