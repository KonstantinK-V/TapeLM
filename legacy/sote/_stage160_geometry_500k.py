"""
Stage 160 — layer geometry dig @ ~500k under locked S★ ritual.

Agreed after 140–159: soft morph/tails flat; next = arch *shape* at scale,
not more soft codebook rows. Ritual fixed to S★; only geometry varies.

Arms (word_fp only, N≈460k, hops OUT, exact@1):
  1. base_2L_d256     — S★ baseline (2L / 4H / d256)
  2. wide_2L_d512     — wide-shallow (2L / 8H / d512)
  3. deep_4L_d256     — thin-deep (4L / 4H / d256)
  4. pyramid_up       — 256→384→512 (3 stages)
  5. pyramid_down     — 512→384→256 (3 stages)

Ritual lock (all arms):
  batch=8  lr=1e-3  Adam wd=0  fat=0.75  max_len=16  steps=40k  warmup=200

Run:
  python _stage160_geometry_500k.py
"""
from __future__ import annotations

import json
import random
import sys
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
    WordIdTransformer,
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
    FT_STEPS,
    WARMUP,
    EVAL_EVERY_WORD,
    SEED_MIX,
    load_phrases,
    setup_stack,
    make_mix,
    make_opt,
    metrics_pack,
    log as pipe_log,
)

DEC = RES / "stage160_geometry_500k_decision.json"
LOG = RES / "_stage160_geometry_500k_log.txt"
N_PHRASES = 460_000
PLAN = RES / "sote_ideal_recipe_140_159.md"

# S★ ritual (geometry only changes)
RITUAL = {
    "batch": 8,
    "lr": 1e-3,
    "opt": "Adam",
    "wd": 0.0,
    "fat_frac": 0.75,
    "word_max_len": 16,
}

GEOMS = {
    "base_2L_d256": {
        "kind": "uniform",
        "d": 256,
        "n_layer": 2,
        "n_head": 4,
        "note": "S★ baseline",
    },
    "wide_2L_d512": {
        "kind": "uniform",
        "d": 512,
        "n_layer": 2,
        "n_head": 8,
        "note": "wide-shallow",
    },
    "deep_4L_d256": {
        "kind": "uniform",
        "d": 256,
        "n_layer": 4,
        "n_head": 4,
        "note": "thin-deep",
    },
    "pyramid_up": {
        "kind": "pyramid",
        "dims": (256, 384, 512),
        "heads": (4, 6, 8),
        "note": "expand 256→384→512",
    },
    "pyramid_down": {
        "kind": "pyramid",
        "dims": (512, 384, 256),
        "heads": (8, 6, 4),
        "note": "funnel 512→384→256",
    },
}


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


def done(path: Path) -> dict | None:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("verdict"):
            log(f"[skip] {path.name}")
            return d
    return None


def write_dec(path: Path, out: dict):
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[write] {path.name}")


class PyramidWordIdTransformer(nn.Module):
    """Word LM with per-stage width (simple pyramid / funnel).

    Same API surface as WordIdTransformer for collate/eval: pad_id, max_len,
    logits_last_from_batch, init_from_fps.
    """

    def __init__(
        self,
        n_vocab: int,
        dims: tuple[int, ...],
        heads: tuple[int, ...],
        max_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(dims) == len(heads) and len(dims) >= 1
        for d, h in zip(dims, heads):
            if d % h != 0:
                raise ValueError(f"d={d} not divisible by heads={h}")
        self.n_vocab = n_vocab
        self.pad_id = n_vocab
        self.dims = tuple(dims)
        self.d_model = dims[0]  # emb width = first stage
        self.d_out = dims[-1]
        self.max_len = max_len
        self.tok = nn.Embedding(n_vocab + 1, self.d_model, padding_idx=self.pad_id)
        self.pos = nn.Embedding(max_len, self.d_model)
        stages = []
        projs = []
        for i, (d, h) in enumerate(zip(dims, heads)):
            if i == 0:
                projs.append(nn.Identity())
                d_in = d
            else:
                projs.append(nn.Linear(dims[i - 1], d))
                d_in = d
            stages.append(
                nn.TransformerEncoderLayer(
                    d_model=d_in,
                    nhead=h,
                    dim_feedforward=4 * d_in,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.projs = nn.ModuleList(projs)
        self.stages = nn.ModuleList(stages)
        self.head = nn.Linear(self.d_out, n_vocab, bias=False)

    @torch.no_grad()
    def init_from_fps(self, word_fps: torch.Tensor):
        V, d_fp = word_fps.shape
        assert V == self.n_vocab
        w = torch.zeros(V + 1, self.d_model, device=word_fps.device, dtype=self.tok.weight.dtype)
        d = min(self.d_model, d_fp)
        w[:V, :d] = word_fps[:, :d].to(dtype=w.dtype)
        w[:V] = F.normalize(w[:V], dim=-1)
        self.tok.weight.copy_(w)
        # tie-ish: init head rows from fps projected/padded to d_out
        with torch.no_grad():
            hw = torch.zeros(V, self.d_out, device=word_fps.device, dtype=self.head.weight.dtype)
            dd = min(self.d_out, d_fp)
            hw[:, :dd] = word_fps[:, :dd].to(dtype=hw.dtype)
            if self.d_out > d_fp:
                hw[:, d_fp:].normal_(std=0.02)
            hw = F.normalize(hw, dim=-1)
            self.head.weight.copy_(hw)

    def forward(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        bsz, tmax = ids.shape
        device = ids.device
        pos = torch.arange(tmax, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.tok(ids) + self.pos(pos)
        for proj, layer, d in zip(self.projs, self.stages, self.dims):
            x = proj(x)
            mask = torch.triu(torch.ones(tmax, tmax, device=device, dtype=torch.bool), diagonal=1)
            x = layer(x, src_mask=mask, src_key_padding_mask=key_padding_mask)
        return self.head(x)

    def logits_last_from_batch(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        return self.forward(ids, key_padding_mask=key_padding_mask)[:, -1, :]

    def logits_from_prefix(self, id_list) -> torch.Tensor:
        if not id_list:
            raise ValueError("empty prefix")
        ids = torch.tensor(
            [id_list[-self.max_len :]], dtype=torch.long, device=self.tok.weight.device
        )
        return self.forward(ids)[0, -1]

    def pred_id(self, id_list) -> int:
        return int(self.logits_from_prefix(id_list).argmax())


def trunk_for(geom: dict) -> dict:
    t = dict(RITUAL)
    if geom["kind"] == "uniform":
        t.update(d=geom["d"], n_layer=geom["n_layer"], n_head=geom["n_head"])
    else:
        # placeholders for logging; real model uses dims/heads
        t.update(d=geom["dims"][0], n_layer=len(geom["dims"]), n_head=geom["heads"][0])
        t["pyramid_dims"] = list(geom["dims"])
        t["pyramid_heads"] = list(geom["heads"])
    return t


def train_uniform(tag, geom, phrases, device, cfg, stack, seed):
    trunk = trunk_for(geom)
    # reuse 150 train_word path
    from _stage150_155_clean_compare_pipeline import train_word

    return train_word(
        tag,
        trunk,
        phrases,
        device,
        cfg,
        stack,
        steps=FT_STEPS,
        seed=seed,
        ckpt_name=f"stage160_{tag}.pt",
    )


def train_pyramid(tag, geom, phrases, device, cfg, stack, seed):
    trunk = trunk_for(geom)
    dims, heads = geom["dims"], geom["heads"]
    log(
        f"\n======== WORD {tag} pyramid dims={dims} heads={heads} "
        f"batch={trunk['batch']} lr={trunk['lr']} fat={trunk['fat_frac']} steps={FT_STEPS} ========"
    )
    train, hold_seen, hold_rare, hold_story, meta = make_mix(phrases, cfg)
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    model = PyramidWordIdTransformer(
        len(words), dims, heads, max_len=trunk["word_max_len"], dropout=0.1
    ).to(device)
    model.init_from_fps(fps)
    if model.d_model > fps.shape[1]:
        with torch.no_grad():
            model.tok.weight[: len(words), fps.shape[1] :].normal_(std=0.02)
            model.tok.weight[: len(words)] = F.normalize(model.tok.weight[: len(words)], dim=-1)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[word] V={len(words)} params={n_params/1e6:.2f}M")

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
    for step in range(1, FT_STEPS + 1):
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
        if step % EVAL_EVERY_WORD == 0 or step == FT_STEPS:
            snap(step)

    last_story = curve[-1]["story_all"] if curve else -1.0
    last_obj = curve[-1]["obj"] if curve else -1.0
    model.load_state_dict(best["state"])
    model.eval()
    fs = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    ft = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    ck = CKPT / f"stage160_{tag}.pt"
    torch.save({"word_tf": best["state"], "surfaces": words, "geom": geom}, ck)
    out = {
        "arm": tag,
        "unit": "word_fp",
        "geom": geom,
        "trunk": trunk,
        "V": len(words),
        "params": n_params,
        "steps": FT_STEPS,
        "curve": curve,
        "ckpt": str(ck),
        "story_all_last": last_story,
        "seen_obj_last": last_obj,
        "ckpt_select": "best_ev_story_then_obj",
        "n_phrases": len(phrases),
        "mix_meta": {k: meta.get(k) for k in ("n_train", "n_hold_seen", "n_story_hold") if k in meta},
        **metrics_pack(fs, ft),
    }
    return out


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    # redirect pipeline logs into our log for train_word
    import _stage150_155_clean_compare_pipeline as p150

    _orig = p150.log

    def _tee(msg):
        _orig(msg)
        # also append to 160 log (pipe_log already writes 150 log)
        line = msg if msg.endswith("\n") else msg + "\n"
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)

    p150.log = _tee

    log(f"160 geometry@500k start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan_ref={PLAN}")
    prev = done(DEC)
    if prev:
        return 0
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg, stack = setup_stack(device)
        phrases = load_phrases(N_PHRASES)
        log(f"[data] n_phrases={len(phrases)} target={N_PHRASES}")

        arms = {}
        for name, geom in GEOMS.items():
            seed = stable_seed(160, name)
            log(f"\n########## GEOM {name}: {geom.get('note')} ##########")
            if geom["kind"] == "uniform":
                arms[name] = train_uniform(name, geom, phrases, device, cfg, stack, seed)
            else:
                arms[name] = train_pyramid(name, geom, phrases, device, cfg, stack, seed)
            log(
                f"[done] {name} STORY={100*arms[name]['story_all']:.1f}% "
                f"SEEN_obj={100*arms[name]['seen_obj']:.1f}% params={arms[name]['params']/1e6:.2f}M"
            )

        base = arms["base_2L_d256"]["story_all"]
        deltas = {k: arms[k]["story_all"] - base for k in arms}
        best_name = max(arms, key=lambda k: arms[k]["story_all"])
        best_lift = deltas[best_name]
        # promote if clear win vs S★
        if best_name != "base_2L_d256" and best_lift >= 0.03:
            verdict = "PASS_PROMOTE"
        elif best_name != "base_2L_d256" and best_lift >= 0.015:
            verdict = "PARTIAL"
        elif max(deltas.values()) >= -0.02 and min(deltas.values()) <= 0.02:
            verdict = "PARITY_TABLE"
        elif best_lift < -0.02 and best_name == "base_2L_d256":
            verdict = "BASE_WINS"
        else:
            verdict = "GEOMETRY_TABLE"

        # prefer BASE_WINS when base is strictly best by >=1.5pp
        ranked = sorted(arms, key=lambda k: arms[k]["story_all"], reverse=True)
        if ranked[0] == "base_2L_d256" and (arms[ranked[0]]["story_all"] - arms[ranked[1]]["story_all"]) >= 0.015:
            verdict = "BASE_WINS"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "geometry_at_500k_under_Sstar",
            "framing": (
                "Lock S★ ritual; vary only layer geometry at N≈460k. "
                "Tests wide-shallow / thin-deep / pyramid vs baseline."
            ),
            "n_phrases": len(phrases),
            "ritual": RITUAL,
            "geoms": GEOMS,
            "arms": arms,
            "deltas_story_vs_base": deltas,
            "best_arm": best_name,
            "best_lift_vs_base": best_lift,
            "ranked": ranked,
            "verdict": verdict,
            "ref100": REF100,
            "ref154_460k_hist_word_story": 0.194,
            "note": (
                "Promote into S★ only on PASS_PROMOTE (>=+3pp). "
                "Do not sum with 151 deltas; this is the geometry factorial at scale."
            ),
        }
        write_dec(DEC, out)
        log(
            f"[160] {verdict} best={best_name} "
            f"STORY={100*arms[best_name]['story_all']:.1f}% "
            f"lift_vs_base={100*best_lift:+.1f}pp"
        )
        log("deltas: " + ", ".join(f"{k}={100*v:+.1f}pp" for k, v in deltas.items()))
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
