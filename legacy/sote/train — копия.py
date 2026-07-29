"""
Stage 28+: light dirty-augment fine-tune.

Warm-start Hop1PartialBinder from 27b+; ADD Hop2LeftBinder for dirty bridge.
Short FT on TRAIN controlled pairs (clean+dirty mix). Re-eval dirty stress.

Frozen: CueBinder / stack / thought_scratch_27c contract
Trainable: Hop1PartialBinder (FT) + Hop2LeftBinder (new)

Protocol same as 28; hop2 can use Hop2LeftBinder instead of probe when trained.

Gates (same soft stress + no clean collapse):
  dirty_bind joint >= 30%, lift vs raw >= 10pp, drop vs clean <= 20pp
  clean_bind not below baseline-5pp (~46%)
  dirty_both_bind target >= 40% (stretch)
  >=2/3 seeds

Results:
  results/stage28_plus_report.txt
  results/stage28_plus_metrics.json
  checkpoints/stage28_plus_dirty_ft.pt

Run:
  python train.py
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

PARENT16 = CKPT / "bridge_qa_binder_16_FROZEN.pt"
SCRATCH26 = CKPT / "thought_scratch_26_FROZEN.pt"
SCRATCH27C = CKPT / "thought_scratch_27c_FROZEN.pt"
HOP25 = CKPT / "hop_bank_25.pt"
PARENT27B = CKPT / "stage27b_plus_hop1_partial.pt"
PARENT27C = CKPT / "stage27c_both_partial.pt"
BASELINE28 = CKPT / "stage28_dirty_partial.pt"
WORD_PATH = CKPT / "word_memory_9_2.pt"
MORPH_PATH = CKPT / "morph_letter_9f_plusplus_FROZEN.pt"
OUT_CKPT = CKPT / "stage28_plus_dirty_ft.pt"
OUT_TXT = RES / "stage28_plus_report.txt"
OUT_JSON = RES / "stage28_plus_metrics.json"

EVAL_SEEDS = (272, 273, 274)
HOLD_EVAL_SEED = 10027
TRAIN_EVAL_SEED = 10026
DIRTY_KINDS = ("drop", "swap", "trunc", "sub")

LETTER_CHARS = [chr(c) for c in range(ord("a"), ord("z") + 1)]
CHARS = LETTER_CHARS + [" "]
CHAR2ID = {c: i for i, c in enumerate(CHARS)}
AFFIXES = ("ing", "ed", "s")
AFFIX2ID = {a: i for i, a in enumerate(AFFIXES)}
RELS = ("on", "to")
PTYPE2ID = {"obj_only": 0, "rel_only": 1, "left_only": 2}

PLAIN_EXTRA = [
    "cat", "dog", "car", "top", "go", "on", "to", "no", "not", "good",
    "spot", "stop", "post", "act", "pot", "card", "bird", "book", "fish",
    "run", "play", "make", "take", "give", "house", "tree", "water",
    "ball", "door", "hand", "land", "sand", "ship", "shop", "rock", "road",
]
MORPH_STEMS = [
    "play", "go", "act", "fish", "post", "book", "make", "take", "give",
    "run", "stop", "spot", "top", "cat", "dog", "look", "talk", "open",
    "help", "move", "call", "work", "jump", "walk", "read", "write",
    "live", "pull", "push", "wash",
]


def _cvc_double(stem: str) -> bool:
    if len(stem) < 3 or len(stem) > 4:
        return False
    return stem[-1] not in "aeiouwy" and stem[-2] in "aeiou" and stem[-3] not in "aeiou"


def surface(stem: str, affix: str) -> str:
    if affix == "ing":
        if stem.endswith("e") and not stem.endswith(("ee", "ye", "oe")):
            return stem[:-1] + "ing"
        if _cvc_double(stem):
            return stem + stem[-1] + "ing"
        return stem + "ing"
    if affix == "ed":
        if stem.endswith("e"):
            return stem + "d"
        if len(stem) > 1 and stem.endswith("y") and stem[-2] not in "aeiou":
            return stem[:-1] + "ied"
        if _cvc_double(stem):
            return stem + stem[-1] + "ed"
        return stem + "ed"
    if affix == "s":
        if stem.endswith(("s", "x", "z", "ch", "sh")) or stem in {"go", "do"}:
            return stem + "es"
        if len(stem) > 1 and stem.endswith("y") and stem[-2] not in "aeiou":
            return stem[:-1] + "ies"
        return stem + "s"
    raise ValueError(affix)


def affix_chars(stem: str, affix: str) -> str:
    form = surface(stem, affix)
    if affix == "ing":
        return "ing"
    if affix == "ed":
        if form.endswith("ied"):
            return "ied"
        return "ed" if form.endswith("ed") else "d"
    if stem.endswith(("s", "x", "z", "ch", "sh")) or stem in {"go", "do"}:
        return "es"
    if len(stem) > 1 and stem.endswith("y") and stem[-2] not in "aeiou":
        return "ies"
    return "s"


def orth_prefix(stem: str, affix: str) -> str:
    form = surface(stem, affix)
    suf = affix_chars(stem, affix)
    return form[: -len(suf)]


def build_morph_lexicon(stems):
    lex = {}
    for stem in stems:
        for aff in AFFIXES:
            lex[surface(stem, aff)] = (stem, aff)
    return lex


MORPH_LEX = build_morph_lexicon(list(dict.fromkeys(MORPH_STEMS + PLAIN_EXTRA)))


@dataclass
class Config:
    dim: int = 64
    seed: int = 272
    episode_k: int = 3
    steps: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    dirty_p: float = 0.55
    clean_w: float = 0.6
    teach_w: float = 0.7
    bridge_w: float = 0.35
    bridge_mix: float = 0.35
    neg_w: float = 0.75
    n_hold_episodes: int = 80
    n_train_episodes: int = 48
    create_below: float = 0.55
    stabilize_lo: float = 0.55
    lock_above: float = 0.92
    ema: float = 0.85
    retrieve_margin: float = 0.05
    max_phrase_len: int = 5
    max_word_len: int = 11
    dirty_joint_min: float = 0.30
    dirty_lift_vs_raw_pp: float = 0.10
    dirty_drop_vs_clean_max: float = 0.20
    dirty_both_min: float = 0.40
    clean_floor: float = 0.46
    pass_seeds_min: int = 2


# --------------- frozen modules ---------------


class LetterEncoder(nn.Module):
    def __init__(self, n_chars: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_chars, dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(self.emb(char_ids)), dim=-1)


class WordComposer(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pos_learned = nn.Embedding(max_len, dim)
        self.content_bind = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.order_bind = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim)
        )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pos_sin", pe)

    def forward(self, char_fps: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        bsz, max_l, dim = char_fps.shape
        device = char_fps.device
        pos_ids = torch.arange(max_l, device=device).unsqueeze(0).expand(bsz, -1)
        pos = F.normalize(self.pos_learned(pos_ids) + self.pos_sin[:max_l].unsqueeze(0), dim=-1)
        x = torch.cat([char_fps, pos], dim=-1)
        content = self.content_bind(x)
        order = self.order_bind(x)
        g = self.gate(torch.cat([content, order], dim=-1))
        slots = g * content + (1.0 - g) * order
        mask = (torch.arange(max_l, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        pooled = (slots * mask.unsqueeze(-1)).sum(1) / lengths.clamp(min=1).float().unsqueeze(1)
        return F.normalize(self.out(pooled), dim=-1)


class MorphModPlus(nn.Module):
    def __init__(self, dim: int, n_affix: int, max_suffix: int):
        super().__init__()
        self.dim = dim
        self.max_suffix = max_suffix
        self.basins = nn.Parameter(torch.randn(n_affix, dim))
        self.pos_emb = nn.Embedding(max_suffix, dim)
        self.letter_scale = nn.Parameter(torch.tensor(1.0))
        self.morph_scale = nn.Parameter(torch.tensor(0.7))
        self.pos_gate_net = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid()
        )
        self.content_gate = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid()
        )
        self.delta = nn.Sequential(nn.Linear(dim * 3, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.delta_scale = nn.Parameter(torch.tensor(0.05))
        self.res = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.res_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, suf_fps: torch.Tensor, affix_ids: torch.Tensor):
        bsz, slen, _ = suf_fps.shape
        device = suf_fps.device
        morph = F.normalize(self.basins[affix_ids], dim=-1).unsqueeze(1).expand(bsz, slen, -1)
        pos = self.pos_emb(torch.arange(slen, device=device) % self.max_suffix)
        pos = pos.unsqueeze(0).expand(bsz, -1, -1)
        x = torch.cat([suf_fps, morph, pos], dim=-1)
        pos_gate = self.pos_gate_net(x)
        c_gate = self.content_gate(x)
        delta = self.delta_scale * self.delta(x)
        morph_part = self.morph_scale * (c_gate * morph) + delta
        base = self.letter_scale * suf_fps + pos_gate * morph_part
        return F.normalize(base + self.res_scale * self.res(base), dim=-1)


class PhraseComposer(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.end = nn.Parameter(F.normalize(torch.randn(dim), dim=0))
        self.pos_learned = nn.Embedding(max_len, dim)
        self.content_bind = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.order_bind = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim)
        )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pos_sin", pe)

    def forward(self, word_fps: torch.Tensor, lengths: torch.Tensor):
        bsz, max_l, dim = word_fps.shape
        device = word_fps.device
        max_out = max_l + 1
        out = torch.zeros(bsz, max_out, dim, device=device)
        new_len = lengths + 1
        idx = torch.arange(max_l, device=device).unsqueeze(0)
        valid = idx < lengths.unsqueeze(1)
        out[:, :max_l] = torch.where(valid.unsqueeze(-1), word_fps, out[:, :max_l])
        end_idx = lengths.view(bsz, 1, 1).expand(bsz, 1, dim)
        out.scatter_(1, end_idx, self.end.view(1, 1, dim).expand(bsz, 1, dim))
        pos_ids = torch.arange(max_out, device=device).unsqueeze(0).expand(bsz, -1)
        pos = F.normalize(self.pos_learned(pos_ids) + self.pos_sin[:max_out].unsqueeze(0), dim=-1)
        x = torch.cat([out, pos], dim=-1)
        content = self.content_bind(x)
        order = self.order_bind(x)
        g = self.gate(torch.cat([content, order], dim=-1))
        slots = g * content + (1.0 - g) * order
        mask = (torch.arange(max_out, device=device).unsqueeze(0) < new_len.unsqueeze(1)).float()
        pooled = (slots * mask.unsqueeze(-1)).sum(1) / new_len.clamp(min=1).float().unsqueeze(1)
        return F.normalize(self.out(pooled), dim=-1), slots, new_len


class CueBinder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.kind_emb = nn.Embedding(2, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, cue_fp: torch.Tensor, kind_ids: torch.Tensor) -> torch.Tensor:
        k = self.kind_emb(kind_ids)
        h = self.net(torch.cat([cue_fp, k], dim=-1))
        return F.normalize(cue_fp + self.res_scale * h, dim=-1)


class Hop1PartialBinder(nn.Module):
    """ADD-on for WHAT hop1 partial cues (CueBinder stays frozen)."""

    def __init__(self, dim: int, n_ptype: int = 2):
        super().__init__()
        self.ptype_emb = nn.Embedding(n_ptype, dim)
        self.kind_emb = nn.Embedding(2, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, partial_fp: torch.Tensor, ptype_ids: torch.Tensor, kind_ids: torch.Tensor):
        x = torch.cat([partial_fp, self.ptype_emb(ptype_ids), self.kind_emb(kind_ids)], dim=-1)
        h = self.net(x)
        return F.normalize(partial_fp + self.res_scale * h, dim=-1)


class Hop2LeftBinder(nn.Module):
    """ADD-on for WHERE hop2 left_only cues (esp. dirty bridge)."""

    def __init__(self, dim: int):
        super().__init__()
        self.ptype_emb = nn.Embedding(1, dim)  # left_only only
        self.kind_emb = nn.Embedding(2, dim)  # where=1
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, partial_fp: torch.Tensor):
        bsz = partial_fp.size(0)
        device = partial_fp.device
        pid = torch.zeros(bsz, dtype=torch.long, device=device)
        kid = torch.ones(bsz, dtype=torch.long, device=device)
        x = torch.cat([partial_fp, self.ptype_emb(pid), self.kind_emb(kid)], dim=-1)
        h = self.net(x)
        return F.normalize(partial_fp + self.res_scale * h, dim=-1)


@dataclass
class MemSlot:
    key: str
    fp: torch.Tensor
    n: int = 1
    locked: bool = False


class SoftPhraseMemory:
    def __init__(self, dim: int, cfg: Config, device):
        self.dim = dim
        self.cfg = cfg
        self.device = device
        self.slots: dict[str, MemSlot] = {}

    def top2(self, fp):
        if not self.slots:
            return None, 0.0, None, 0.0
        names = list(self.slots.keys())
        sims = fp @ torch.stack([self.slots[n].fp for n in names], 0).T
        if len(names) == 1:
            return names[0], float(sims[0]), None, 0.0
        vals, idx = torch.topk(sims, k=2)
        return names[int(idx[0])], float(vals[0]), names[int(idx[1])], float(vals[1])

    def max_sim(self, fp):
        if not self.slots:
            return 0.0
        names = list(self.slots.keys())
        return float((fp @ torch.stack([self.slots[n].fp for n in names], 0).T).max().item())

    def retrieve(self, fp):
        fp = F.normalize(fp.detach(), dim=-1)
        if not self.slots:
            return None, 0.0, 0.0, False
        best, s1, second, s2 = self.top2(fp)
        gap = s1 - s2 if second is not None else s1
        if second is not None and gap < self.cfg.retrieve_margin:
            return None, s1, gap, False
        return best, s1, gap, True

    def observe(self, key, fp):
        fp = F.normalize(fp.detach(), dim=-1)
        if key in self.slots:
            slot = self.slots[key]
            if slot.locked:
                return "lock"
            slot.fp = F.normalize(self.cfg.ema * slot.fp + (1 - self.cfg.ema) * fp, dim=-1)
            slot.n += 1
            if float((slot.fp * fp).sum()) >= self.cfg.lock_above and slot.n >= 3:
                slot.locked = True
                return "lock"
            return "stabilize"
        if self.max_sim(fp) < self.cfg.create_below or not self.slots:
            self.slots[key] = MemSlot(key, fp.clone(), n=1, locked=False)
            return "create"
        best, s1, _, _ = self.top2(fp)
        if s1 >= self.cfg.stabilize_lo:
            slot = self.slots[best]
            if not slot.locked:
                slot.fp = F.normalize(self.cfg.ema * slot.fp + (1 - self.cfg.ema) * fp, dim=-1)
                slot.n += 1
            return "stabilize"
        self.slots[key] = MemSlot(key, fp.clone(), n=1, locked=False)
        return "create"


class ThoughtScratch:
    def __init__(self, device):
        self.device = device
        self.stack = []
        self.cells = {}
        self.meta = {}

    def write(self, tag, fp, cell=None, meta=None):
        fp = F.normalize(fp.detach(), dim=-1)
        self.stack.append({"tag": tag, "fp": fp.clone()})
        if cell is not None:
            self.cells[cell] = fp.clone()
        if meta is not None and cell is not None:
            self.meta[cell] = meta
        return "scratch_write"

    def depth(self):
        return len(self.stack)


def encode_chars(encoder, text, device):
    if not text:
        return torch.zeros(0, 64, device=device)
    return encoder(torch.tensor([CHAR2ID[c] for c in text], device=device))


@torch.no_grad()
def compose_plain(encoder, composer, word, device):
    fps = encode_chars(encoder, word, device)
    return composer(fps.unsqueeze(0), torch.tensor([fps.size(0)], device=device)).squeeze(0)


def reattach_morph(morph, encoder, composer, stem, affix, device):
    pref = orth_prefix(stem, affix)
    suf = affix_chars(stem, affix)
    pref_fps = encode_chars(encoder, pref, device)
    suf_fps = encode_chars(encoder, suf, device)
    aid = torch.tensor([AFFIX2ID[affix]], device=device)
    suf_mod = morph(suf_fps.unsqueeze(0), aid).squeeze(0)
    full = torch.cat([pref_fps, suf_mod], 0) if pref_fps.numel() else suf_mod
    return composer(full.unsqueeze(0), torch.tensor([full.size(0)], device=device)).squeeze(0)


def word_fp(encoder, composer, morph, word, device):
    if word in MORPH_LEX:
        stem, aff = MORPH_LEX[word]
        return reattach_morph(morph, encoder, composer, stem, aff, device)
    return compose_plain(encoder, composer, word, device)


class Stack:
    def __init__(self, encoder, composer, morph, phrase, binder, device):
        self.encoder = encoder
        self.composer = composer
        self.morph = morph
        self.phrase = phrase
        self.binder = binder
        self.device = device
        self._w = {}

    def w(self, word):
        if word not in self._w:
            with torch.no_grad():
                self._w[word] = word_fp(self.encoder, self.composer, self.morph, word, self.device).detach()
        return self._w[word]

    @torch.no_grad()
    def phrase_fp(self, words):
        fps = torch.stack([self.w(x) for x in words], 0)
        out, _, _ = self.phrase(fps.unsqueeze(0), torch.tensor([fps.size(0)], device=self.device))
        return out.squeeze(0).detach()


def load_stack(device, cfg):
    assert SCRATCH26.exists() and HOP25.exists() and PARENT27B.exists()
    contr = torch.load(PARENT16, map_location="cpu", weights_only=False)
    w = torch.load(WORD_PATH, map_location="cpu", weights_only=False)
    encoder = LetterEncoder(len(CHARS), cfg.dim).to(device)
    composer = WordComposer(cfg.dim, int(w.get("max_word_len", cfg.max_word_len))).to(device)
    encoder.load_state_dict(w["encoder"], strict=False)
    composer.load_state_dict(w["composer"], strict=False)
    mck = torch.load(MORPH_PATH, map_location="cpu", weights_only=False)
    morph = MorphModPlus(cfg.dim, len(AFFIXES), int(mck.get("config", {}).get("max_suffix", 4))).to(device)
    morph.load_state_dict(mck["morph"], strict=True)
    phrase = PhraseComposer(cfg.dim, int(contr.get("config", {}).get("max_phrase_len", cfg.max_phrase_len))).to(device)
    phrase.load_state_dict(contr["phrase_composer"], strict=False)
    binder = CueBinder(cfg.dim).to(device)
    binder.load_state_dict(contr["cue_binder"], strict=True)
    for mod in (encoder, composer, morph, phrase, binder):
        for p in mod.parameters():
            p.requires_grad_(False)
        mod.eval()
    return Stack(encoder, composer, morph, phrase, binder, device), torch.load(HOP25, map_location="cpu", weights_only=False)


def load_hop1_binders(device, cfg):
    ck = torch.load(PARENT27B, map_location="cpu", weights_only=False)
    out = {}
    by = ck.get("binders_by_seed") or {}
    if by:
        for s, sd in by.items():
            m = Hop1PartialBinder(cfg.dim).to(device)
            m.load_state_dict(sd, strict=True)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            out[int(s)] = m
    else:
        m = Hop1PartialBinder(cfg.dim).to(device)
        m.load_state_dict(ck["hop1_partial_binder"], strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        out[int(ck.get("best_seed", cfg.seed))] = m
    return out, ck


def parse_rel(phrase: str):
    ws = phrase.split()
    for i, w in enumerate(ws):
        if w in RELS and i > 0 and i + 1 < len(ws):
            return {"phrase": phrase, "left": ws[i - 1], "rel": w, "right": ws[i + 1]}
    return None


def make_rel_bank(phrases):
    return [p for p in (parse_rel(x) for x in phrases) if p]


def full_what(fact):
    return f"{fact['rel']} {fact['right']}"


def hop1_partial(fact, which: str):
    if which == "first":
        return fact["right"], "obj_only"
    return fact["rel"], "rel_only"


def dirty_token(tok: str, kind: str, rng: random.Random):
    """Apply mild orthographic noise; keep at least 1 letter."""
    tok = "".join(c for c in tok.lower() if c in CHAR2ID and c != " ")
    if not tok:
        return "x", "empty"
    if len(tok) == 1:
        alt = rng.choice([c for c in LETTER_CHARS if c != tok])
        return (alt if kind == "sub" else tok), kind if kind == "sub" else "noop"
    if kind == "drop":
        i = rng.randrange(len(tok))
        out = tok[:i] + tok[i + 1 :]
        return (out if out else tok[0]), "drop"
    if kind == "swap":
        i = rng.randrange(len(tok) - 1)
        return tok[:i] + tok[i + 1] + tok[i] + tok[i + 2 :], "swap"
    if kind == "trunc":
        keep = max(1, len(tok) - 1)
        return tok[:keep], "trunc"
    if kind == "sub":
        i = rng.randrange(len(tok))
        alt = rng.choice([c for c in LETTER_CHARS if c != tok[i]])
        return tok[:i] + alt + tok[i + 1 :], "sub"
    return tok, "noop"


def safe_phrase_fp(stack, text: str):
    words = [w for w in text.split() if w]
    if not words:
        words = ["x"]
    # letter-compose unknown / dirty tokens via stack.w (morph or plain)
    return stack.phrase_fp(words)


def candidates(partial, ptype, facts):
    lefts = sorted({f["left"] for f in facts})
    rights = sorted({f["right"] for f in facts})
    out = []
    if ptype == "obj_only":
        for rel in RELS:
            out.append((f"{rel} {partial}", "what"))
    elif ptype == "left_only":
        for rel in RELS:
            out.append((f"{partial} {rel}", "where"))
    else:  # rel_only
        for r in rights:
            out.append((f"{partial} {r}", "what"))
        for l in lefts:
            out.append((f"{l} {partial}", "where"))
    seen, uniq = set(), []
    for t, k in out:
        if t not in seen:
            seen.add(t)
            uniq.append((t, k))
    return uniq


def probe_complete(stack, mem, scratch, partial, ptype, facts, tag="h"):
    best = None
    scratch.write(
        f"{tag}_partial:{partial}",
        safe_phrase_fp(stack, partial),
        cell="partial_cue",
        meta=f"{ptype}:{partial}",
    )
    for text, ck in candidates(partial, ptype, facts):
        bound = bind_full(stack, ck, text)
        scratch.write(f"{tag}_cand:{text}", bound)
        key, sim, gap, conf = mem.retrieve(bound)
        score = sim + (0.05 if conf else 0) + 0.02 * gap
        if key is not None and (best is None or score > best["score"]):
            best = {"key": key, "sim": sim, "text": text, "score": score, "bound": bound}
    if best is None:
        bound = bind_full(stack, "where" if ptype == "left_only" else "what", partial)
        key, sim, gap, conf = mem.retrieve(bound)
        return key, sim, partial, bound
    scratch.write(f"{tag}_complete:{best['text']}", best["bound"], cell="complete_cue", meta=best["text"])
    return best["key"], best["sim"], best["text"], best["bound"]


def answer_right(key):
    f = parse_rel(key) if key else None
    return f["right"] if f else None


def sample_episode(pairs, bank, k, rng):
    a_p, b_p = pairs[rng.randrange(len(pairs))]
    a, b = bank[a_p], bank[b_p]
    facts = [a, b]
    seen = {a_p, b_p}
    pool = list(bank.values())
    while len(facts) < k:
        d = pool[rng.randrange(len(pool))]
        if d["phrase"] not in seen:
            facts.append(d)
            seen.add(d["phrase"])
    rng.shuffle(facts)
    return facts, a, b


def bind_full(stack, kind, text):
    cue = safe_phrase_fp(stack, text)
    kid = torch.tensor([0 if kind == "what" else 1], device=stack.device)
    return stack.binder(cue.unsqueeze(0), kid).squeeze(0)


@torch.no_grad()
def run_episode(
    stack, cfg, facts, fact_a, fact_b, mode: str, which: str, hop1_binder=None, hop2_binder=None, dirty_rng=None
):
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    scratch = ThoughtScratch(stack.device)
    for f in facts:
        mem.observe(f["phrase"], stack.phrase_fp(f["phrase"].split()))

    dirty = mode.startswith("dirty")
    dirty_bridge = mode == "dirty_both_bind"
    dkind = None
    partial_clean, ptype = hop1_partial(fact_a, which)
    partial = partial_clean
    if dirty:
        assert dirty_rng is not None
        dkind = DIRTY_KINDS[dirty_rng.randrange(len(DIRTY_KINDS))]
        partial, dkind = dirty_token(partial_clean, dkind, dirty_rng)

    # ----- hop1 -----
    if mode == "clean_bind":
        assert hop1_binder is not None
        pid = torch.tensor([PTYPE2ID[ptype]], device=stack.device)
        kid = torch.tensor([0], device=stack.device)
        bound = hop1_binder(stack.phrase_fp(partial_clean.split()).unsqueeze(0), pid, kid).squeeze(0)
        scratch.write(f"h1_bind:{partial_clean}", bound, cell="complete_cue", meta=partial_clean)
        key1, sim1, gap1, conf1 = mem.retrieve(bound)
        used1 = partial_clean
    elif mode == "dirty_raw":
        scratch.write(
            f"h1_dirty:{partial}",
            safe_phrase_fp(stack, partial),
            cell="partial_cue",
            meta=f"{dkind}:{partial_clean}->{partial}",
        )
        bound = bind_full(stack, "what", partial)
        scratch.write("h1_raw", bound, cell="complete_cue", meta=partial)
        key1, sim1, gap1, conf1 = mem.retrieve(bound)
        used1 = partial
    elif mode == "dirty_probe":
        key1, sim1, used1, bound = probe_complete(stack, mem, scratch, partial, ptype, facts, "h1")
    else:  # dirty_bind | dirty_both_bind
        assert hop1_binder is not None
        scratch.write(
            f"h1_dirty:{partial}",
            safe_phrase_fp(stack, partial),
            cell="partial_cue",
            meta=f"{dkind}:{partial_clean}->{partial}",
        )
        pid = torch.tensor([PTYPE2ID[ptype]], device=stack.device)
        kid = torch.tensor([0], device=stack.device)
        bound = hop1_binder(safe_phrase_fp(stack, partial).unsqueeze(0), pid, kid).squeeze(0)
        scratch.write(f"h1_bind:{partial}", bound, cell="complete_cue", meta=partial)
        key1, sim1, gap1, conf1 = mem.retrieve(bound)
        used1 = partial

    ok1 = key1 == fact_a["phrase"] and answer_right(key1) == fact_a["right"]
    bridge = answer_right(key1) if key1 else None
    if bridge is None:
        bridge = fact_a["right"]
        bridge_src = "fallback_gold"
    else:
        bridge_src = "hop1_ans"
    scratch.write(f"bridge:{bridge}", stack.w(bridge), cell="bridge_entity", meta=bridge)
    if key1 and key1 in mem.slots:
        scratch.write("attended_1", mem.slots[key1].fp, cell="attended_1")

    hop2_cue = bridge
    dkind2 = None
    if dirty_bridge:
        dkind2 = DIRTY_KINDS[dirty_rng.randrange(len(DIRTY_KINDS))]
        hop2_cue, dkind2 = dirty_token(bridge, dkind2, dirty_rng)
        scratch.write(
            f"bridge_dirty:{hop2_cue}",
            safe_phrase_fp(stack, hop2_cue),
            cell="partial_cue",
            meta=f"{dkind2}:{bridge}->{hop2_cue}",
        )

    # ----- hop2 -----
    use_h2_binder = hop2_binder is not None and mode in ("clean_bind", "dirty_bind", "dirty_both_bind")
    if mode == "dirty_raw":
        bound = bind_full(stack, "where", hop2_cue)
        scratch.write("h2_raw", bound, cell="complete_cue", meta=hop2_cue)
        key2, sim2, gap2, conf2 = mem.retrieve(bound)
        used2 = hop2_cue
    elif use_h2_binder:
        bound = hop2_binder(safe_phrase_fp(stack, hop2_cue).unsqueeze(0)).squeeze(0)
        scratch.write(f"h2_bind:{hop2_cue}", bound, cell="complete_cue", meta=hop2_cue)
        key2, sim2, gap2, conf2 = mem.retrieve(bound)
        used2 = hop2_cue
    else:
        key2, sim2, used2, bound = probe_complete(stack, mem, scratch, hop2_cue, "left_only", facts, "h2")

    if key2 and key2 in mem.slots:
        scratch.write("attended_2", mem.slots[key2].fp, cell="attended_2")
    ans2 = answer_right(key2)
    ok2 = key2 == fact_b["phrase"] and ans2 == fact_b["right"]
    return {
        "ok1": ok1,
        "ok2": ok2,
        "joint": bool(ok1 and ok2),
        "used1": used1,
        "used2": used2,
        "bridge": bridge,
        "bridge_src": bridge_src,
        "dirty_kind": dkind,
        "dirty_kind2": dkind2,
        "partial_clean": partial_clean,
        "partial_dirty": partial if dirty else partial_clean,
        "depth": scratch.depth(),
        "a": fact_a["phrase"],
        "b": fact_b["phrase"],
        "mode": mode,
    }


@torch.no_grad()
def eval_modes(stack, pairs, bank, cfg, rng, n_eps, modes, hop1_binder, hop2_binder=None):
    stats = {m: {"ok1": 0, "ok2": 0, "joint": 0, "depth": 0} for m in modes}
    by_kind = {k: {"n": 0, "joint": 0} for k in DIRTY_KINDS}
    samples = []
    for i in range(n_eps):
        facts, a, b = sample_episode(pairs, bank, cfg.episode_k, rng)
        which = "first" if (i % 2 == 0) else "second"
        ep_seed = HOLD_EVAL_SEED * 10007 + i * 17 + (0 if which == "first" else 1)
        row = {"a": a["phrase"], "b": b["phrase"], "which": which}
        last_r = None
        for m in modes:
            r = run_episode(
                stack,
                cfg,
                facts,
                a,
                b,
                m,
                which,
                hop1_binder,
                hop2_binder=hop2_binder,
                dirty_rng=random.Random(ep_seed),
            )
            last_r = r
            stats[m]["ok1"] += int(r["ok1"])
            stats[m]["ok2"] += int(r["ok2"])
            stats[m]["joint"] += int(r["joint"])
            stats[m]["depth"] += r["depth"]
            row[f"{m}_j"] = r["joint"]
            row[f"{m}_1"] = r["ok1"]
            if m == "dirty_bind" and r.get("dirty_kind") in by_kind:
                by_kind[r["dirty_kind"]]["n"] += 1
                by_kind[r["dirty_kind"]]["joint"] += int(r["joint"])
        if i < 8 and last_r is not None:
            row["dirty_ex"] = (
                f"{last_r.get('partial_clean')}->{last_r.get('partial_dirty')}[{last_r.get('dirty_kind')}]"
            )
            samples.append(row)
    n = max(n_eps, 1)
    out = {"n": n, "n_pairs": len(pairs), "samples": samples, "modes": {}, "by_kind": {}}
    for m in modes:
        out["modes"][m] = {
            "hop1": stats[m]["ok1"] / n,
            "hop2": stats[m]["ok2"] / n,
            "joint": stats[m]["joint"] / n,
            "mean_depth": stats[m]["depth"] / n,
        }
    for k, v in by_kind.items():
        out["by_kind"][k] = (v["joint"] / v["n"]) if v["n"] else None
    return out


def seed_ok(hold, cfg):
    clean = hold["modes"]["clean_bind"]["joint"]
    raw = hold["modes"]["dirty_raw"]["joint"]
    bind = hold["modes"]["dirty_bind"]["joint"]
    both = hold["modes"]["dirty_both_bind"]["joint"]
    lift = bind - raw
    drop = clean - bind
    ok = (
        lift >= cfg.dirty_lift_vs_raw_pp
        and drop <= cfg.dirty_drop_vs_clean_max
        and bind >= cfg.dirty_joint_min
        and clean >= cfg.clean_floor
        and both >= cfg.dirty_both_min
    )
    return ok, clean, raw, bind, lift, drop, both


def build_ft_examples(pairs, bank, rng):
    """Hop1 (A) + Hop2 (B via bridge=A.right) supervised examples."""
    h1, h2 = [], []
    all_facts = list(bank.values())
    for a_p, b_p in pairs:
        if a_p not in bank or b_p not in bank:
            continue
        a, b = bank[a_p], bank[b_p]
        others = [f for f in all_facts if f["phrase"] not in (a_p, b_p)]
        for which in ("first", "second"):
            partial, ptype = hop1_partial(a, which)
            negs = [others[rng.randrange(len(others))] for _ in range(min(3, len(others)))] if others else []
            h1.append(
                {
                    "fact": a,
                    "partial": partial,
                    "ptype": ptype,
                    "negs": negs,
                    "full": full_what(a),
                }
            )
        # hop2: bridge entity -> B
        negs2 = [others[rng.randrange(len(others))] for _ in range(min(3, len(others)))] if others else []
        h2.append({"fact": b, "bridge": a["right"], "negs": negs2})
    return h1, h2


def train_dirty_ft(hop1, hop2, stack, h1_ex, h2_ex, cfg, rng, device):
    opt = torch.optim.Adam(list(hop1.parameters()) + list(hop2.parameters()), lr=cfg.lr)
    hist = []
    hop1.train()
    hop2.train()
    for step in range(1, cfg.steps + 1):
        loss = torch.tensor(0.0, device=device)
        # --- hop1 batch ---
        batch1 = [h1_ex[rng.randrange(len(h1_ex))] for _ in range(cfg.batch_size)]
        for ex in batch1:
            use_dirty = rng.random() < cfg.dirty_p
            tok = ex["partial"]
            if use_dirty:
                kind = DIRTY_KINDS[rng.randrange(len(DIRTY_KINDS))]
                tok, _ = dirty_token(tok, kind, rng)
            cue = safe_phrase_fp(stack, tok)
            pid = torch.tensor([PTYPE2ID[ex["ptype"]]], device=device)
            kid = torch.tensor([0], device=device)
            pred = hop1(cue.unsqueeze(0), pid, kid).squeeze(0)
            pos = stack.phrase_fp(ex["fact"]["phrase"].split())
            right_fp = stack.w(ex["fact"]["right"])
            anchor = F.normalize(pos + cfg.bridge_mix * right_fp, dim=-1)
            with torch.no_grad():
                teacher = bind_full(stack, "what", ex["full"])
            loss_pos = 1.0 - (pred * pos).sum()
            loss_teach = 1.0 - (pred * teacher).sum()
            loss_bridge = 1.0 - (pred * anchor).sum()
            loss_neg = torch.tensor(0.0, device=device)
            for n in ex["negs"]:
                nfp = stack.phrase_fp(n["phrase"].split())
                loss_neg = loss_neg + F.relu((pred * nfp).sum() - (pred * pos).sum() + 0.1)
            w = cfg.clean_w if not use_dirty else 1.0
            loss = loss + w * (
                loss_pos + cfg.teach_w * loss_teach + cfg.bridge_w * loss_bridge
                + cfg.neg_w * loss_neg / max(len(ex["negs"]), 1)
            )
        # --- hop2 batch ---
        batch2 = [h2_ex[rng.randrange(len(h2_ex))] for _ in range(cfg.batch_size)]
        for ex in batch2:
            use_dirty = rng.random() < cfg.dirty_p
            tok = ex["bridge"]
            if use_dirty:
                kind = DIRTY_KINDS[rng.randrange(len(DIRTY_KINDS))]
                tok, _ = dirty_token(tok, kind, rng)
            cue = safe_phrase_fp(stack, tok)
            pred = hop2(cue.unsqueeze(0)).squeeze(0)
            pos = stack.phrase_fp(ex["fact"]["phrase"].split())
            with torch.no_grad():
                teacher = bind_full(stack, "where", f"{ex['bridge']} {ex['fact']['rel']}")
            loss_pos = 1.0 - (pred * pos).sum()
            loss_teach = 1.0 - (pred * teacher).sum()
            loss_neg = torch.tensor(0.0, device=device)
            for n in ex["negs"]:
                nfp = stack.phrase_fp(n["phrase"].split())
                loss_neg = loss_neg + F.relu((pred * nfp).sum() - (pred * pos).sum() + 0.1)
            w = cfg.clean_w if not use_dirty else 1.0
            loss = loss + w * (loss_pos + cfg.teach_w * loss_teach + cfg.neg_w * loss_neg / max(len(ex["negs"]), 1))
        loss = loss / (2 * cfg.batch_size)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.detach())})
            print(f"  [28+] step {step:4d}/{cfg.steps} loss={float(loss.detach()):.4f}", flush=True)
    hop1.eval()
    hop2.eval()
    return hist


def format_report(hold, train, cfg, device, seed_rows, baseline=None):
    lines = [
        "SOTE Stage 28+ report (dirty-augment FT + Hop2LeftBinder)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"device: {device}",
        f"parents: {PARENT27B.name} + {SCRATCH27C.name} + {HOP25.name}",
        f"knobs: steps={cfg.steps} lr={cfg.lr} dirty_p={cfg.dirty_p} clean_w={cfg.clean_w}",
        f"HOLD pairs={hold['n_pairs']} episodes={hold['n']} HOLD_EVAL_SEED={HOLD_EVAL_SEED}",
        "",
        f"{'mode':<18} {'H1':>7} {'H2':>7} {'JOINT':>8} {'depth':>7}",
        "-" * 52,
    ]
    for m, h in hold["modes"].items():
        lines.append(
            f"{m:<18} {h['hop1']*100:6.1f}% {h['hop2']*100:6.1f}% {h['joint']*100:7.1f}% {h['mean_depth']:7.2f}"
        )
    if hold.get("by_kind"):
        lines += ["", "dirty_bind joint by kind:"]
        for k, v in hold["by_kind"].items():
            lines.append(f"  {k:<8} {('n/a' if v is None else f'{v*100:.1f}%')}")
    if baseline:
        lines += ["", "vs Stage28 baseline (seed-mean approx):"]
        for m in ("clean_bind", "dirty_bind", "dirty_both_bind"):
            if m in baseline and m in hold["modes"]:
                b, n = baseline[m], hold["modes"][m]["joint"]
                lines.append(f"  {m:<18} {b*100:.1f}% -> {n*100:.1f}%  ({(n-b)*100:+.1f}pp)")
    lines += ["", "samples:"]
    for s in hold["samples"][:6]:
        lines.append(
            f"  [{s['which']}] {s['a']} => {s['b']}  {s.get('dirty_ex','')}  "
            f"clean={int(s.get('clean_bind_j',0))} raw={int(s.get('dirty_raw_j',0))} "
            f"probe={int(s.get('dirty_probe_j',0))} bind={int(s.get('dirty_bind_j',0))} "
            f"both={int(s.get('dirty_both_bind_j',0))}"
        )
    ok, clean, raw, bind, lift, drop, both = seed_ok(hold, cfg)
    probe = hold["modes"]["dirty_probe"]["joint"]
    lines += ["", "=== Best-seed detail ==="]
    lines.append(
        f"  clean={clean*100:.1f}% dirty_raw={raw*100:.1f}% probe={probe*100:.1f}% "
        f"dirty_bind={bind*100:.1f}% dirty_both={both*100:.1f}% "
        f"lift={lift*100:+.1f}pp drop_vs_clean={drop*100:+.1f}pp"
    )
    lines += ["", "=== Multi-seed ==="]
    n_pass = sum(1 for r in seed_rows if r["ok"])
    mean_j = sum(r["bind_joint"] for r in seed_rows) / len(seed_rows)
    mean_both = sum(r["both_joint"] for r in seed_rows) / len(seed_rows)
    mean_clean = sum(r["clean_joint"] for r in seed_rows) / len(seed_rows)
    for r in seed_rows:
        tag = "PASS" if r["ok"] else "MISS"
        lines.append(
            f"  seed {r['seed']}: {tag}  dirty_bind={r['bind_joint']*100:.1f}% "
            f"dirty_both={r['both_joint']*100:.1f}% clean={r['clean_joint']*100:.1f}% "
            f"lift={r['lift']*100:+.1f}pp drop={r['drop']*100:+.1f}pp"
        )
    overall = n_pass >= cfg.pass_seeds_min
    lines.append(
        f"  aggregate: {n_pass}/{len(seed_rows)} seeds pass; "
        f"mean_dirty={mean_j*100:.1f}% mean_both={mean_both*100:.1f}% mean_clean={mean_clean*100:.1f}%"
    )
    lines += ["", "=== Verdict ==="]
    if overall:
        lines.append("  PASS: dirty FT keeps clean floor and lifts dirty_both toward gate.")
    elif mean_both >= cfg.dirty_both_min - 0.05 and mean_clean >= cfg.clean_floor:
        lines.append("  PARTIAL: dirty_both improved but multi-seed / gates soft-miss.")
    else:
        lines.append("  FAIL / PARTIAL: FT did not clear dirty_both or hurt clean.")
    lines.append("")
    return "\n".join(lines), overall


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Device:", device, f"({torch.cuda.get_device_name(device)})")
    else:
        print("Device: cpu")

    print("Stage 28+: dirty-augment FT (Hop1 FT + new Hop2LeftBinder)")
    stack, hop = load_stack(device, cfg)
    parent_binders, parent_ck = load_hop1_binders(device, cfg)
    print(f"warm-start Hop1 seeds={sorted(parent_binders)}")

    baseline = None
    if BASELINE28.exists():
        b28 = torch.load(BASELINE28, map_location="cpu", weights_only=False)
        rows = b28.get("seed_rows") or []
        if rows:
            # mean of hold modes from first seed's hold if present, else seed_rows fields
            baseline = {
                "clean_bind": sum(r.get("clean_joint", 0) for r in rows) / len(rows),
                "dirty_bind": sum(r.get("bind_joint", 0) for r in rows) / len(rows),
                "dirty_both_bind": None,
            }
            # try pull dirty_both from hold_modes
            boths = []
            for r in rows:
                hm = r.get("hold_modes") or {}
                if "dirty_both_bind" in hm:
                    boths.append(hm["dirty_both_bind"]["joint"])
            if boths:
                baseline["dirty_both_bind"] = sum(boths) / len(boths)
            both_s = (
                f" both={baseline['dirty_both_bind']*100:.1f}%"
                if baseline.get("dirty_both_bind") is not None
                else ""
            )
            print(
                f"baseline28 mean clean={baseline['clean_bind']*100:.1f}% "
                f"dirty={baseline['dirty_bind']*100:.1f}%{both_s}"
            )

    hold_pairs = [(a, b) for a, b in hop["hold_controlled_pairs"]]
    train_pairs = [(a, b) for a, b in hop["train_controlled_pairs"]]
    hold_phrases = list(dict.fromkeys(list(hop["hold_extra"]) + [x for ab in hold_pairs for x in ab]))
    train_phrases = list(dict.fromkeys(list(hop["train_extra"]) + [x for ab in train_pairs for x in ab]))
    hold_bank = {f["phrase"]: f for f in make_rel_bank(hold_phrases)}
    train_bank = {f["phrase"]: f for f in make_rel_bank(train_phrases)}
    hold_pairs = [(a, b) for a, b in hold_pairs if a in hold_bank and b in hold_bank]
    train_pairs = [(a, b) for a, b in train_pairs if a in train_bank and b in train_bank]
    print(f"pairs HOLD={len(hold_pairs)} TRAIN={len(train_pairs)}")

    modes = ("clean_bind", "dirty_raw", "dirty_probe", "dirty_bind", "dirty_both_bind")
    seed_rows = []
    trained = {}
    for seed in EVAL_SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)
        rng = random.Random(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        hop1 = Hop1PartialBinder(cfg.dim).to(device)
        hop1.load_state_dict(parent_binders[seed].state_dict(), strict=True)
        hop2 = Hop2LeftBinder(cfg.dim).to(device)
        for p in hop1.parameters():
            p.requires_grad_(True)
        for p in hop2.parameters():
            p.requires_grad_(True)

        h1_ex, h2_ex = build_ft_examples(train_pairs, train_bank, rng)
        print(f"\n=== Seed {seed}: FT hop1={len(h1_ex)} hop2={len(h2_ex)} ===", flush=True)
        hist = train_dirty_ft(hop1, hop2, stack, h1_ex, h2_ex, cfg, rng, device)
        trained[seed] = {"hop1": hop1, "hop2": hop2, "hist": hist}

        print(f"Eval HOLD seed={seed}...", flush=True)
        hold_res = eval_modes(
            stack,
            hold_pairs,
            hold_bank,
            cfg,
            random.Random(HOLD_EVAL_SEED),
            cfg.n_hold_episodes,
            modes,
            hop1,
            hop2_binder=hop2,
        )
        train_res = eval_modes(
            stack,
            train_pairs,
            train_bank,
            cfg,
            random.Random(TRAIN_EVAL_SEED),
            cfg.n_train_episodes,
            modes,
            hop1,
            hop2_binder=hop2,
        )
        for m, h in hold_res["modes"].items():
            print(f"  {m:<18} H1={h['hop1']*100:5.1f}% H2={h['hop2']*100:5.1f}% joint={h['joint']*100:5.1f}%")
        ok, clean, raw, bind, lift, drop, both = seed_ok(hold_res, cfg)
        seed_rows.append(
            {
                "seed": seed,
                "ok": ok,
                "bind_joint": bind,
                "both_joint": both,
                "clean_joint": clean,
                "raw_joint": raw,
                "lift": lift,
                "drop": drop,
                "hold": hold_res,
                "train": train_res,
                "hist": hist,
            }
        )

    best = max(seed_rows, key=lambda r: (r["ok"], r["both_joint"], r["bind_joint"], r["clean_joint"]))
    # filter baseline keys that are None for format
    base_for_report = None
    if baseline and baseline.get("dirty_both_bind") is not None:
        base_for_report = baseline
    report, overall_ok = format_report(
        best["hold"], best["train"], cfg, str(device), seed_rows, baseline=base_for_report
    )
    print()
    print(report)

    torch.save(
        {
            "stage": "28+",
            "pass": overall_ok,
            "best_seed": best["seed"],
            "hop1_partial_binder": trained[best["seed"]]["hop1"].state_dict(),
            "hop2_left_binder": trained[best["seed"]]["hop2"].state_dict(),
            "binders_by_seed": {
                str(s): {
                    "hop1": trained[s]["hop1"].state_dict(),
                    "hop2": trained[s]["hop2"].state_dict(),
                }
                for s in trained
            },
            "seed_rows": [
                {
                    "seed": r["seed"],
                    "ok": r["ok"],
                    "bind_joint": r["bind_joint"],
                    "both_joint": r["both_joint"],
                    "clean_joint": r["clean_joint"],
                    "raw_joint": r["raw_joint"],
                    "lift": r["lift"],
                    "drop": r["drop"],
                    "hold_modes": r["hold"]["modes"],
                    "by_kind": r["hold"].get("by_kind"),
                }
                for r in seed_rows
            ],
            "config": asdict(cfg),
            "parents": [PARENT27B.name, SCRATCH27C.name, HOP25.name, PARENT27C.name],
            "hold": best["hold"],
            "train": best["train"],
            "history": best["hist"],
            "note": "dirty-augment FT hop1 + new hop2 left binder",
        },
        OUT_CKPT,
    )
    OUT_TXT.write_text(report, encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "pass": overall_ok,
                "best_seed": best["seed"],
                "seeds": [
                    {
                        "seed": r["seed"],
                        "ok": r["ok"],
                        "bind_joint": r["bind_joint"],
                        "both_joint": r["both_joint"],
                        "clean_joint": r["clean_joint"],
                        "raw_joint": r["raw_joint"],
                        "lift": r["lift"],
                        "drop": r["drop"],
                        "hold": r["hold"],
                        "train": r["train"],
                    }
                    for r in seed_rows
                ],
                "config": asdict(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {OUT_CKPT}")


if __name__ == "__main__":
    main()
