"""Shared helpers for TapeLM extension stages 214+ (zero-train policies, adapters).

Canonical memory + decode (227 / 228c):
  - Write slot keys in **canonical** fp (frozen P1 `arc_enc`).
  - Read under domain shift: **qmap** query with `W_bwd` (domain → canonical).
  - Official decode: **4-way slot retrieve** → score candidates with `cos(fp(c), fp(retrieved))`.
  Do **not** use global argmax retrieve + fp (228b anti-pattern).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank

if TYPE_CHECKING:
    pass

W_REGISTRY_DIR = Path("checkpoints/w_registry")
W_REGISTRY_MANIFEST = W_REGISTRY_DIR / "w_registry.json"

WORD_RE_CTX = WORD_RE


class RecencyFpBank(FpBank):
    """ctx_fp with exponential decay by word index distance to entity (zero-train)."""

    def __init__(self, model, stoi, device, lam: float = 0.0, entity_at_end: bool = True):
        super().__init__(model, stoi, device)
        self.lam = lam
        self.entity_at_end = entity_at_end

    @torch.no_grad()
    def ctx_fp(self, text: str, exclude: str | None = None) -> torch.Tensor | None:
        if self.lam <= 0.0:
            return super().ctx_fp(text, exclude=exclude)
        words = [w for w in WORD_RE_CTX.findall(text) if w != exclude]
        if len(words) < 3:
            return None
        # entity anchor: last capitalized token in window (194-style entities)
        ent_idx = None
        for i in range(len(words) - 1, -1, -1):
            if ENT_RE.match(words[i]) and words[i] != exclude:
                ent_idx = i
                break
        if ent_idx is None:
            ent_idx = len(words) - 1 if self.entity_at_end else 0
        fps = self.fp(words)
        weights = []
        for i in range(len(words)):
            d = abs(i - ent_idx)
            weights.append(math.exp(-self.lam * d))
        w = torch.tensor(weights, device=fps.device, dtype=fps.dtype)
        w = w / w.sum().clamp(min=1e-9)
        return F.normalize((fps * w.unsqueeze(-1)).sum(0), dim=-1)


@torch.no_grad()
def slow_endpoint_vec(model, char_table, pad_id, ctx_ids: list[int], device: torch.device) -> torch.Tensor | None:
    """Last slow-channel state for a BPE token-id context window (frozen P1)."""
    from _stage191_night import MAX_ARCS

    seq = (ctx_ids)[-MAX_ARCS:]
    if not seq:
        return None
    x = torch.tensor([seq], dtype=torch.long, device=device)
    pad = x == pad_id
    arcs = model._arcs(char_table[x], ids=x)
    slow, _, _ = model.slow(arcs, pad)
    valid = (~pad[0]).nonzero(as_tuple=False)
    if len(valid) == 0:
        return None
    t = int(valid[-1].item())
    return F.normalize(slow[0, t], dim=-1)


class DomainAdapter(nn.Module):
    """Learnable warp in fp-space: fp' = normalize(W @ fp)."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.w(fp), dim=-1)

    def map_raw(self, fp: torch.Tensor) -> torch.Tensor:
        """Linear part before normalize (for slot keys)."""
        return self.w(fp)


class BottleneckRemap(nn.Module):
    """Tiny d -> r -> d remap (fewer params than full 256x256)."""

    def __init__(self, d: int = 256, r: int = 32):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.up(self.down(fp)), dim=-1)

    def map_raw(self, fp: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(fp))


class WFamilyPolicy:
    """
    L2 migration policy: Identity vs registry[family] vs learn-new.
    Thresholds seeded by stage 224 (knobs, not laws).
    """

    def __init__(
        self,
        registry: dict[str, DomainAdapter] | None = None,
        cos_identity: float = 0.85,
        cos_family_floor: float = 0.65,
    ):
        self.registry = dict(registry or {})
        self.cos_identity = cos_identity
        self.cos_family_floor = cos_family_floor

    def decide(self, mean_cos_core: float, family: str | None) -> dict:
        if mean_cos_core > self.cos_identity:
            return {"action": "identity", "family": family, "mean_cos": mean_cos_core}
        if family and family in self.registry:
            return {"action": "use_registry", "family": family, "mean_cos": mean_cos_core}
        if mean_cos_core < self.cos_family_floor:
            return {"action": "learn_family_W", "family": family or "outlier", "mean_cos": mean_cos_core}
        return {"action": "learn_or_attach_family", "family": family or "prose", "mean_cos": mean_cos_core}

    def get(self, family: str) -> DomainAdapter | None:
        return self.registry.get(family)

    def set(self, family: str, W: DomainAdapter) -> None:
        self.registry[family] = W

    @staticmethod
    def should_fork(matched_recall: float, reuse_recall: float, drop_tol: float = 0.05) -> bool:
        """True if existing family W is not good enough for a new corpus."""
        return (matched_recall - reuse_recall) >= drop_tol


def fp_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular-convolution-style bind in fp-space (elementwise product, normalized)."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return F.normalize(a * b, dim=-1)


def weighted_slot_sims(
    sims: torch.Tensor,
    ages: list[int],
    w_versions: list[str],
    active_w_version: str,
    tau: float,
    version_penalty: float = 0.25,
) -> torch.Tensor:
    """L3 read: age decay × penalty when slot `w_version` ≠ active registry era."""
    out = sims.clone()
    for j, (age, wv) in enumerate(zip(ages, w_versions)):
        w = math.exp(-age / max(tau, 1e-6))
        if wv != active_w_version:
            w *= version_penalty
        out[j] = out[j] * w
    return out


def pick_w_bwd_for_era(
    registry: dict[str, DomainAdapter],
    era: str,
    fallback: str = "prose_bwd",
) -> DomainAdapter | None:
    """Temporal W: map logical era label → persisted adapter key (e.g. prose_v2_bwd)."""
    key = f"{era}_bwd" if not era.endswith("_bwd") else era
    if key in registry:
        return registry[key]
    return registry.get(fallback)


@torch.no_grad()
def mean_core_cos(bank_a: FpBank, bank_b: FpBank, core: list[str]) -> float:
    Fa = bank_a.fp(core)
    Fb = bank_b.fp(core)
    return float((Fa * Fb).sum(-1).mean())


def compose_w_bwd(W_outer: DomainAdapter, W_inner: DomainAdapter) -> DomainAdapter:
    """Compose qmap adapters: ``normalize(W_outer @ W_inner @ q)`` (227 qmap chain)."""
    d = W_outer.w.weight.shape[0]
    W = DomainAdapter(d).to(W_outer.w.weight.device)
    with torch.no_grad():
        W.w.weight.copy_(W_outer.w.weight @ W_inner.w.weight)
    W.eval()
    return W


def lexicon_nearest(fp: torch.Tensor, lex_fps: torch.Tensor) -> torch.Tensor:
    """Snap batch of fp vectors to nearest row in lex_fps."""
    idx = (fp @ lex_fps.T).argmax(dim=-1)
    return lex_fps[idx]


# --- Canonical read / official fp decode (227 + 228c) ---


def apply_qmap(W_bwd: DomainAdapter, q: torch.Tensor) -> torch.Tensor:
    """Map domain query fp → canonical key space (227 P_qmap)."""
    if q.dim() == 1:
        return F.normalize(W_bwd.map_raw(q.unsqueeze(0)), dim=-1)[0]
    return F.normalize(W_bwd.map_raw(q), dim=-1)


def slot_retrieve_4way(
    K: torch.Tensor,
    V: list[str],
    qq: torch.Tensor,
    candidates: list[str],
) -> str:
    """Among `candidates`, pick value with best max slot-key cosine to `qq` (227 exam protocol)."""
    best_sc, best_c = -1.0, candidates[0]
    for c in candidates:
        idxs = [i for i, v in enumerate(V) if v == c]
        if not idxs:
            sc = -1.0
        else:
            sc = float((K[idxs] @ qq).max())
        if sc > best_sc:
            best_sc, best_c = sc, c
    return best_c


def slot_retrieve_global(K: torch.Tensor, V: list[str], qq: torch.Tensor) -> str:
    """Global argmax over all slots — 228b broken protocol; contrast only."""
    return V[int((K @ qq).argmax())]


@torch.no_grad()
def fp_cos_scores(bank: FpBank, anchor_word: str, candidates: list[str]) -> list[float]:
    """cos(fp(word), fp(c)) for each candidate c."""
    anchor = bank.fp([anchor_word])[0]
    fps = bank.fp(candidates)
    return [float((fps[i] * anchor).sum()) for i in range(len(candidates))]


@torch.no_grad()
def fp_decode_pick_retrieved_4way(
    bank_can: FpBank,
    K_can: torch.Tensor,
    V: list[str],
    W_bwd: DomainAdapter,
    bank_query: FpBank,
    ctx: str,
    exclude: str | None,
    candidates: list[str],
) -> tuple[str, str]:
    """
    Official TapeLM memory decode API (228c).

    Returns (retrieved_slot_value, chosen_candidate).
    """
    q = bank_query.ctx_fp(ctx, exclude=exclude)
    if q is None:
        raise ValueError("ctx_fp returned None for decode context")
    qq = apply_qmap(W_bwd, q)
    retrieved = slot_retrieve_4way(K_can, V, qq, candidates)
    scores = fp_cos_scores(bank_can, retrieved, candidates)
    pick = candidates[int(max(range(len(scores)), key=lambda i: scores[i]))]
    return retrieved, pick


def save_w_family(path: Path, W: DomainAdapter, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "state_dict": W.state_dict()}
    torch.save(payload, path)


def load_w_family(path: Path, device: torch.device | str = "cpu") -> tuple[DomainAdapter, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    W = DomainAdapter(256).to(device)
    W.load_state_dict(blob["state_dict"])
    W.eval()
    return W, dict(blob.get("meta") or {})


def load_w_registry(
    registry_dir: Path | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, DomainAdapter], dict]:
    """
    Load manifest + all family adapters listed in `w_registry.json`.

    Expects keys like `prose_bwd`, `code_bwd` (qmap) and optional `*_fwd` (keylift).
    """
    root = registry_dir or W_REGISTRY_DIR
    manifest_path = root / "w_registry.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run artifact/scripts/export_w_registry.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapters: dict[str, DomainAdapter] = {}
    for family, spec in manifest.get("families", {}).items():
        for direction, rel in spec.get("files", {}).items():
            key = f"{family}_{direction}"
            path = root / rel
            W, _ = load_w_family(path, device)
            adapters[key] = W
    return adapters, manifest


# --- Contradiction resolution (229 → policy layer) ---


@dataclass(frozen=True)
class AnnotatedSlotHit:
    value: str
    score: float
    provenance: str
    year: int


def subject_slot_hits(
    K: torch.Tensor,
    V: list[str],
    qq: torch.Tensor,
    slot_indices: list[int],
    meta: list[dict],
) -> list[AnnotatedSlotHit]:
    """Cosine scores for a subject's slot rows, highest first."""
    hits: list[AnnotatedSlotHit] = []
    for i in slot_indices:
        m = meta[i]
        hits.append(
            AnnotatedSlotHit(
                value=V[i],
                score=float(K[i] @ qq),
                provenance=str(m.get("provenance", "unknown")),
                year=int(m.get("year", 0)),
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def _query_wants_revision(query: str) -> bool | None:
    q = query.lower()
    if re.search(r"\b1999\b|revision|later claim|updated record", q):
        return True
    if re.search(r"\b1987\b|original record|official records|as filed", q):
        return False
    return None


def resolve_slot_contradiction(
    hits: list[AnnotatedSlotHit],
    query: str,
    policy: str = "composite",
    gap_tau: float = 0.05,
) -> str:
    """
    Pick one value among contradictory slot hits (229 upper layer).

    Policies: ``argmax``, ``recency``, ``query_cue``, ``composite`` (default).
    """
    if not hits:
        raise ValueError("no slot hits to resolve")
    if len(hits) == 1:
        return hits[0].value

    if policy == "argmax":
        return hits[0].value
    if policy == "recency":
        return max(hits, key=lambda h: h.year).value

    want_rev = _query_wants_revision(query)
    if policy == "query_cue":
        if want_rev is True:
            for h in hits:
                if h.provenance == "revision":
                    return h.value
        if want_rev is False:
            for h in hits:
                if h.provenance == "official":
                    return h.value
        return max(hits, key=lambda h: h.year).value

    # composite: tight score gap → trust metadata + query; else argmax
    gap = abs(hits[0].score - hits[1].score)
    if gap >= gap_tau:
        return hits[0].value
    if want_rev is True:
        for h in hits:
            if h.provenance == "revision":
                return h.value
    if want_rev is False:
        for h in hits:
            if h.provenance == "official":
                return h.value
    return max(hits, key=lambda h: h.year).value

