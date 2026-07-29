"""
Stage 164 — 0L controls (after 163).

Question: how much of 1L d128 STORY (~21.6%) is just emb→linear, with no attention?

Arms (N≈460k, word_fp, S★ ritual, 50k steps, hops OUT):
  1. zero_last_d128      — logits = W @ emb[last_token]   (bigram-like)
  2. zero_meanpool_d128  — logits = W @ mean(emb[prefix]) (bag)
  3. zero_last_d256      — same as (1) at d256 (width ctrl)

Waits for stage163_onelayer_floor_heads_decision.json

Run:
  python _stage164_zerolayer_control.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    RES,
    _warmup_then_constant,
    collate_word_id_batch,
    eval_id_capacity_suite,
    lines_to_pairs,
)
from _stage109_110_slot_baseline import REF100, _subsample  # noqa: E402
from _stage150_155_clean_compare_pipeline import (  # noqa: E402
    EV_SEEN_N,
    EV_STORY_N,
    FIN_SEEN_N,
    FIN_STORY_N,
    WARMUP,
    EVAL_EVERY_WORD,
    load_phrases,
    setup_stack,
    make_mix,
    make_opt,
    metrics_pack,
)
from _stage160_geometry_500k import RITUAL  # noqa: E402

DEC163 = RES / "stage163_onelayer_floor_heads_decision.json"
DEC162 = RES / "stage162_onelayer_d128_50k_decision.json"
DEC160 = RES / "stage160_geometry_500k_decision.json"
DEC = RES / "stage164_zerolayer_control_decision.json"
LOG = RES / "_stage164_zerolayer_control_log.txt"
N_PHRASES = 460_000
STEPS = 50_000

ARMS = {
    "zero_last_d128": {
        "d": 128,
        "mode": "last",
        "note": "0L last-token linear (bigram-like)",
    },
    "zero_meanpool_d128": {
        "d": 128,
        "mode": "meanpool",
        "note": "0L mean-pool prefix linear (bag)",
    },
    "zero_last_d256": {
        "d": 256,
        "mode": "last",
        "note": "0L last-token @ d256 width ctrl",
    },
}


class ZeroLayerWordLM(nn.Module):
    """No attention: emb → (last | meanpool) → linear → vocab logits.

    Same eval API as WordIdTransformer: pad_id, max_len, pred_id,
    logits_last_from_batch, init_from_fps.
    """

    def __init__(self, n_vocab: int, d_model: int = 128, max_len: int = 16, mode: str = "last"):
        super().__init__()
        assert mode in ("last", "meanpool")
        self.n_vocab = n_vocab
        self.pad_id = n_vocab
        self.d_model = d_model
        self.max_len = max_len
        self.mode = mode
        self.tok = nn.Embedding(n_vocab + 1, d_model, padding_idx=self.pad_id)
        self.head = nn.Linear(d_model, n_vocab, bias=True)

    @torch.no_grad()
    def init_from_fps(self, word_fps: torch.Tensor):
        V, d_fp = word_fps.shape
        assert V == self.n_vocab
        w = torch.zeros(V + 1, self.d_model, device=word_fps.device, dtype=self.tok.weight.dtype)
        d = min(self.d_model, d_fp)
        w[:V, :d] = word_fps[:, :d].to(dtype=w.dtype)
        if self.d_model > d_fp:
            w[:V, d_fp:].normal_(std=0.02)
        w[:V] = F.normalize(w[:V], dim=-1)
        self.tok.weight.copy_(w)
        with torch.no_grad():
            hw = torch.zeros(V, self.d_model, device=word_fps.device, dtype=self.head.weight.dtype)
            hw[:, :d] = word_fps[:, :d].to(dtype=hw.dtype)
            if self.d_model > d_fp:
                hw[:, d_fp:].normal_(std=0.02)
            hw = F.normalize(hw, dim=-1)
            self.head.weight.copy_(hw)
            self.head.bias.zero_()

    def _pool(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """ids [B,T] → pooled [B,d]. Left-padded batches: last col is prefix end."""
        emb = self.tok(ids)  # [B,T,d]
        if self.mode == "last":
            return emb[:, -1, :]
        # meanpool over non-pad
        if key_padding_mask is None:
            # True = pad in HF convention used by collate
            key_padding_mask = ids == self.pad_id
        valid = (~key_padding_mask).unsqueeze(-1).to(emb.dtype)  # [B,T,1]
        s = (emb * valid).sum(dim=1)
        z = valid.sum(dim=1).clamp(min=1.0)
        return s / z

    def forward(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        h = self._pool(ids, key_padding_mask)
        return self.head(h)

    def logits_last_from_batch(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        return self.forward(ids, key_padding_mask=key_padding_mask)

    def logits_from_prefix(self, id_list) -> torch.Tensor:
        if not id_list:
            raise ValueError("empty prefix")
        ids = torch.tensor(
            [id_list[-self.max_len :]], dtype=torch.long, device=self.tok.weight.device
        )
        return self.forward(ids)[0]

    def pred_id(self, id_list) -> int:
        return int(self.logits_from_prefix(id_list).argmax())


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF


def wait_163(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage163_onelayer_floor_heads_decision.json ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC163.exists():
            d = json.loads(DEC163.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 163 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 163 ... {int(time.time()-t0)}s")
    raise TimeoutError("163 not ready")


def already_done(path: Path) -> dict | None:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("verdict"):
            log(f"[skip] {path.name}")
            return d
    return None


def write_dec(path: Path, out: dict):
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[write] {path.name}")


def train_zero(tag, spec, phrases, device, cfg, stack, seed):
    trunk = dict(RITUAL)
    d = spec["d"]
    mode = spec["mode"]
    log(
        f"\n======== ZERO {tag} mode={mode} d={d} batch={trunk['batch']} "
        f"lr={trunk['lr']} fat={trunk['fat_frac']} steps={STEPS} ========"
    )
    train, hold_seen, hold_rare, hold_story, meta = make_mix(phrases, cfg)
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    model = ZeroLayerWordLM(len(words), d_model=d, max_len=trunk["word_max_len"], mode=mode).to(device)
    model.init_from_fps(fps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[zero] V={len(words)} params={n_params/1e6:.2f}M mode={mode}")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 60000, seed + 1), stoi)
    if not fat_p:
        fat_p = story_p

    ev_seen = _subsample(hold_seen, EV_SEEN_N, seed + 10)
    ev_story = _subsample(hold_story, EV_STORY_N, seed + 11)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), FIN_SEEN_N), seed + 20)
    fin_story = _subsample(hold_story, min(len(hold_story), FIN_STORY_N), seed + 21)

    opt = make_opt(model.parameters(), trunk)
    rr = random.Random(seed)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []
    batch = trunk["batch"]
    fat_frac = trunk["fat_frac"]

    def snap(step):
        model.eval()
        s = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        st = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
        obj = s["obj"]["hit1"]
        rel = s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(f"  [{tag}] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(
                story_all=sall,
                obj=obj,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        model.train()

    model.train()
    snap(0)
    for step in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, trunk["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac)))
        n_fat = min(n_fat, batch)
        batch_ex = [rr.choice(fat_p) for _ in range(n_fat)] + [
            rr.choice(story_p) for _ in range(batch - n_fat)
        ]
        packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY_WORD == 0 or step == STEPS:
            snap(step)

    last_story = curve[-1]["story_all"] if curve else -1.0
    last_obj = curve[-1]["obj"] if curve else -1.0
    model.load_state_dict(best["state"])
    model.eval()
    fs = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    ft = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    ck = CKPT / f"stage164_{tag}.pt"
    torch.save({"zero_lm": best["state"], "surfaces": words, "spec": spec}, ck)
    return {
        "arm": tag,
        "unit": "word_fp",
        "spec": spec,
        "trunk": {**{k: trunk[k] for k in ("batch", "lr", "opt", "wd", "fat_frac")}, "d": d, "n_layer": 0},
        "V": len(words),
        "params": n_params,
        "steps": STEPS,
        "curve": curve,
        "ckpt": str(ck),
        "story_all_last": last_story,
        "seen_obj_last": last_obj,
        "ckpt_select": "best_ev_story_then_obj",
        "n_phrases": len(phrases),
        **metrics_pack(fs, ft),
    }


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")

    import _stage150_155_clean_compare_pipeline as p150

    _orig = p150.log

    def _tee(msg):
        _orig(msg)
        line = msg if msg.endswith("\n") else msg + "\n"
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)

    p150.log = _tee

    log(f"164 zerolayer control start {datetime.now(timezone.utc).isoformat()}")
    if already_done(DEC):
        return 0
    try:
        up = wait_163()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg, stack = setup_stack(device)
        phrases = load_phrases(N_PHRASES)
        log(f"[data] n_phrases={len(phrases)} steps={STEPS}")

        arms = {}
        for name, spec in ARMS.items():
            seed = stable_seed(164, name)
            log(f"\n########## {name}: {spec['note']} ##########")
            arms[name] = train_zero(name, spec, phrases, device, cfg, stack, seed)
            log(
                f"[done] {name} STORY={100*arms[name]['story_all']:.1f}% "
                f"SEEN_obj={100*arms[name]['seen_obj']:.1f}% params={arms[name]['params']/1e6:.2f}M"
            )

        ref_1l = None
        if DEC162.exists():
            d162 = json.loads(DEC162.read_text(encoding="utf-8"))
            ref_1l = (d162.get("arms") or {}).get("shallow_1L_d128_50k", {}).get("story_all")
        ref_base = None
        if DEC160.exists():
            d160 = json.loads(DEC160.read_text(encoding="utf-8"))
            ref_base = (d160.get("arms") or {}).get("base_2L_d256", {}).get("story_all")
        if ref_1l is None:
            ref_1l = 0.0

        deltas = {k: arms[k]["story_all"] - ref_1l for k in arms}
        ranked = sorted(arms, key=lambda k: arms[k]["story_all"], reverse=True)
        best = ranked[0]
        gap_attn = ref_1l - arms["zero_last_d128"]["story_all"]

        # If 0L within 2pp of 1L → attention adds little; if gap >=5pp → attention carries
        if gap_attn >= 0.05:
            verdict = "ATTENTION_MATTERS"
        elif gap_attn >= 0.02:
            verdict = "ATTENTION_HELPS"
        elif gap_attn >= -0.02:
            verdict = "ATTENTION_REDUNDANT"
        else:
            verdict = "ZEROLAYER_WINS"  # shouldn't happen often

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "zerolayer_control_vs_1L",
            "framing": (
                "0L emb→linear controls: last-token (bigram-like) and meanpool (bag). "
                "Measures how much 1L attention adds over pure codebook projection."
            ),
            "n_phrases": len(phrases),
            "steps": STEPS,
            "ritual": RITUAL,
            "arms": arms,
            "ref_1L_d128_50k_story": ref_1l,
            "ref_160_base_story": ref_base,
            "deltas_vs_1L_d128": deltas,
            "attention_gap_1L_minus_0L_last": gap_attn,
            "best_arm": best,
            "ranked": ranked,
            "verdict": verdict,
            "upstream_163": up.get("verdict"),
            "ref100": REF100,
            "read": (
                "ATTENTION_MATTERS: gap>=5pp — keep 1L. "
                "ATTENTION_HELPS: 2–5pp. "
                "ATTENTION_REDUNDANT: 0L≈1L — emb table carries STORY."
            ),
        }
        write_dec(DEC, out)
        log(
            f"[164] {verdict} best={best} STORY={100*arms[best]['story_all']:.1f}% "
            f"gap_1L_minus_0Llast={100*gap_attn:+.1f}pp"
        )
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
