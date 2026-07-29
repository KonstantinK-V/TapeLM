"""
Stages 141-147 — weird digs pipeline (~10 GPU-h on 3050).

Waits for Stage140, then runs sequentially. Each dig writes its own
decision JSON. Hops OUT. Gate = word exact@1 STORY ALL vs matched ctrl.

  141 reverse_lm_word
  142 masked_word_fill
  143 char_stream_word_confirm
  144 random_bigram_bank
  145 pos_shuffle_bag
  146 copy_token_noise_emb
  147 two_word_blank_space

Run:
  python _stage141_147_weird_pipeline.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CHAR2ID,
    CKPT,
    RES,
    Config,
    _role,
    _warmup_then_constant,
    build_ts_repeat_mix,
    line_subkind,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage137_dynamic_bigram_input import WordOutTransformer  # noqa: E402

DEC140 = RES / "stage140_bpe_budget_word_atoms_decision.json"
LOG = RES / "_stage141_147_log.txt"
PIPELINE_DEC = RES / "stage141_147_weird_pipeline_decision.json"

BATCH = 8
FAT_FRAC = 0.75
EVAL_EVERY = 2500


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_140(timeout_s=24 * 3600, poll=60):
    log(f"[wait] for {DEC140.name} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC140.exists():
            d = json.loads(DEC140.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 140 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("140 not ready")


def verdict_vs(ctrl_story, dig_story, ctrl_obj, dig_obj, ctrl_rel, dig_rel):
    lift = dig_story - ctrl_story
    obj_d = dig_obj - ctrl_obj
    rel_ok = dig_rel >= 0.70 and ctrl_rel >= 0.70
    if obj_d < -0.03:
        return "HARM", lift, obj_d
    if rel_ok and lift >= 0.03 and obj_d >= -0.02:
        return "PASS", lift, obj_d
    if rel_ok and lift >= 0.015:
        return "PARTIAL", lift, obj_d
    return "PARITY", lift, obj_d


def collate(prefs, targets, pad_id, device, max_len):
    prefs = [p[-max_len:] for p in prefs if p]
    if not prefs:
        return None
    tmax = max(len(p) for p in prefs)
    bsz = len(prefs)
    ids = torch.full((bsz, tmax), pad_id, dtype=torch.long, device=device)
    mask = torch.ones((bsz, tmax), dtype=torch.bool, device=device)
    for i, p in enumerate(prefs):
        ids[i, -len(p) :] = torch.tensor(p, dtype=torch.long, device=device)
        mask[i, -len(p) :] = False
    tgt = torch.tensor(targets, dtype=torch.long, device=device)
    return ids, mask, tgt


def setup_data(phrases, seed=272):
    cfg = Config()
    cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
    cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
    cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent.exists():
        parent = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent)
    for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
        for p in mod.parameters():
            p.requires_grad_(False)
        mod.eval()
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=seed)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    return cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps, meta


def word_eval(model, hold, stoi, words, n_word, prefix_fn):
    roles = defaultdict(lambda: {"n": 0, "h": 0})
    obj = {"n": 0, "h": 0}
    for ln in hold:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            if gold not in stoi:
                continue
            ids = prefix_fn(ws, t)
            if not ids:
                continue
            logits = model.logits_from_prefix(ids)
            pred = int(logits[:n_word].argmax())
            ok = int(pred == stoi[gold])
            fake = {"target_word": gold, "prefix_len": t}
            role = _role(fake, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h"] += ok
            if role == "right":
                obj["n"] += 1
                obj["h"] += ok
    pack = {k: {"n": v["n"], "hit1": v["h"] / max(v["n"], 1)} for k, v in roles.items()}
    return {"roles": pack, "obj": {"n": obj["n"], "hit1": obj["h"] / max(obj["n"], 1)}}


def train_loop(
    *,
    tag,
    model,
    sample_fn,
    eval_fn,
    steps,
    device,
    seed,
):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(seed)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []
    n_params = sum(p.numel() for p in model.parameters())
    log(f"\n======== {tag} params={n_params/1e6:.2f}M steps={steps} ========")

    def snap(step):
        model.eval()
        s, st = eval_fn("seen"), eval_fn("story")
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
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        batch = [sample_fn(rr) for _ in range(BATCH)]
        batch = [b for b in batch if b is not None]
        if not batch:
            continue
        prefs, tgts = zip(*batch)
        packed = collate(list(prefs), list(tgts), model.pad_id, device, model.max_len)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    return best, curve, n_params


def make_word_model(n_word, fps, device, max_len=16):
    m = WordOutTransformer(n_word, n_word, 256, 4, 2, max_len, 0.1).to(device)
    m.init_word_rows(fps)
    return m


def save_dig(stage, slug, verdict, payload):
    path = RES / f"stage{stage}_{slug}_decision.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "slug": slug,
        "verdict": verdict,
        **payload,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RES / f"stage{stage}_{verdict}.txt").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"[{stage}] {verdict} -> {path.name}")
    return payload


# ---------- digs ----------

def dig_141(ctx):
    """Reverse LM: train on reversed windows; eval original next-word."""
    steps = 30000
    cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps, meta = ctx
    n_word = len(words)
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = _subsample([ln for ln in train if ln.get("bucket") != "fat_train"] or train, 40000, 41)
    ev_s = _subsample(hold_seen, 400, 1411)
    ev_t = _subsample(hold_story, 300, 1413)
    fin_s = _subsample(hold_seen, min(len(hold_seen), 800), 2411)
    fin_t = _subsample(hold_story, min(len(hold_story), 500), 2413)

    def pools(lines, reverse=False):
        ex = []
        for ln in lines:
            ws = list(ln["words"])
            if reverse:
                ws = list(reversed(ws))
            ids = [stoi[w] for w in ws if w in stoi]
            for i in range(1, len(ids)):
                ex.append((ids[:i], ids[i]))
        return ex

    fat_f, st_f = pools(fat, False), pools(story, False)
    fat_r, st_r = pools(fat, True), pools(story, True)

    def run(tag, fat_ex, st_ex, reverse_eval):
        model = make_word_model(n_word, fps, device)

        def sample(rr):
            pool = fat_ex if rr.random() < FAT_FRAC else st_ex
            return rr.choice(pool) if pool else None

        def pref_fn(ws, t):
            # eval always in ORIGINAL order next-word
            if reverse_eval:
                # model was trained reverse; at eval we still ask original next-word
                # using forward prefix (transfer probe) — honest "does reverse help forward?"
                return [stoi[w] for w in ws[:t] if w in stoi][-model.max_len :]
            return [stoi[w] for w in ws[:t] if w in stoi][-model.max_len :]

        def ev(which):
            hold = ev_s if which == "seen" else ev_t
            return word_eval(model, hold, stoi, words, n_word, pref_fn)

        best, curve, n_params = train_loop(
            tag=tag, model=model, sample_fn=sample, eval_fn=ev, steps=steps, device=device, seed=141
        )
        model.load_state_dict(best["state"])
        fs, ft = (
            word_eval(model, fin_s, stoi, words, n_word, pref_fn),
            word_eval(model, fin_t, stoi, words, n_word, pref_fn),
        )
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": n_params,
        }

    ctrl = run("141_forward", fat_f, st_f, False)
    rev = run("141_reverse", fat_r, st_r, True)
    v, lift, obj_d = verdict_vs(
        ctrl["story_all"], rev["story_all"], ctrl["seen_obj"], rev["seen_obj"], ctrl["seen_rel"], rev["seen_rel"]
    )
    return save_dig(
        141,
        "reverse_lm_word",
        v,
        {
            "hypothesis": "reverse LM nearly matches forward next-word",
            "ctrl": ctrl,
            "dig": rev,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
            "note": "Reverse trained; eval=original-order next-word (transfer).",
        },
    )


def dig_142(ctx):
    """Masked fill + short causal; vs causal ctrl."""
    steps = 30000
    cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps, meta = ctx
    n_word = len(words)
    MASK = n_word  # extra emb row
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = _subsample([ln for ln in train if ln.get("bucket") != "fat_train"] or train, 40000, 42)
    ev_s = _subsample(hold_seen, 400, 1421)
    ev_t = _subsample(hold_story, 300, 1423)
    fin_s = _subsample(hold_seen, min(len(hold_seen), 800), 2421)
    fin_t = _subsample(hold_story, min(len(hold_story), 500), 2423)

    def causal_pool(lines):
        ex = []
        for ln in lines:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            for i in range(1, len(ids)):
                ex.append(("ce", ids[:i], ids[i]))
        return ex

    def mask_pool(lines):
        ex = []
        for ln in lines:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            if len(ids) < 3:
                continue
            for i in range(1, len(ids) - 1):
                pref = ids[:i] + [MASK]  # mask at i, predict ids[i] from left+mask only (causal-ish)
                # true bidirectional would need encoder; keep causal mask-fill: left context + MASK token
                ex.append(("mask", pref, ids[i]))
        return ex

    def run(tag, use_mask):
        model = WordOutTransformer(n_word, n_word + 1, 256, 4, 2, 16, 0.1).to(device)
        model.init_word_rows(fps)
        with torch.no_grad():
            model.tok.weight[MASK] = fps.mean(0)
            model.tok.weight[MASK] = F.normalize(model.tok.weight[MASK], dim=-1)
        # logits over words only: wrap
        class M(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.pad_id = base.pad_id
                self.max_len = base.max_len
                self.n_word = n_word

            def forward(self, ids, key_padding_mask=None):
                return self.base.forward(ids, key_padding_mask)  # already [:n_word]

            def logits_last_from_batch(self, ids, key_padding_mask=None):
                return self.base.logits_last_from_batch(ids, key_padding_mask)

            def logits_from_prefix(self, id_list):
                return self.base.logits_from_prefix(id_list)

        model = M(model)
        ce_fat, ce_st = causal_pool(fat), causal_pool(story)
        mk_fat, mk_st = mask_pool(fat), mask_pool(story)

        def sample(rr):
            if use_mask and rr.random() < 0.5:
                pool = mk_fat if rr.random() < FAT_FRAC else mk_st
            else:
                pool = ce_fat if rr.random() < FAT_FRAC else ce_st
            if not pool:
                return None
            _kind, pref, tgt = rr.choice(pool)
            return pref, tgt

        def pref_fn(ws, t):
            return [stoi[w] for w in ws[:t] if w in stoi][-model.max_len :]

        def ev(which):
            hold = ev_s if which == "seen" else ev_t
            return word_eval(model, hold, stoi, words, n_word, pref_fn)

        best, curve, n_params = train_loop(
            tag=tag, model=model, sample_fn=sample, eval_fn=ev, steps=steps, device=device, seed=142
        )
        model.load_state_dict(best["state"])
        fs = word_eval(model, fin_s, stoi, words, n_word, pref_fn)
        ft = word_eval(model, fin_t, stoi, words, n_word, pref_fn)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": n_params,
        }

    ctrl = run("142_causal", False)
    dig = run("142_maskmix", True)
    v, lift, obj_d = verdict_vs(
        ctrl["story_all"], dig["story_all"], ctrl["seen_obj"], dig["seen_obj"], ctrl["seen_rel"], dig["seen_rel"]
    )
    return save_dig(
        142,
        "masked_word_fill",
        v,
        {
            "hypothesis": "mask-fill (+causal mix) lifts causal exact@1",
            "ctrl": ctrl,
            "dig": dig,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
        },
    )


def dig_143(ctx):
    """Char stream CE; word confirm at boundaries."""
    steps = 25000
    cfg, device, stack, train, hold_seen, hold_story, words, stoi_w, fps_w, meta = ctx
    # char vocab from CHAR2ID letters present
    chars = sorted([c for c in CHAR2ID if c != " "]) + ["_"]  # _ = space/boundary
    ctoi = {c: i for i, c in enumerate(chars)}
    n_char = len(chars)
    n_word = len(words)
    # unified emb: words first? For char LM use char-only model; word confirm via separate map
    # Simpler: char LM only; at word boundary decode last-char-span... hard.
    # Practical: expand words to char ids with _ between; train next-char;
    # word confirm: after prefix words as char-stream ending with _, argmax among word rows
    # by scoring each candidate word's first-char? Too weak.
    # Better word confirm: maintain parallel word model head — skip.
    # Use: encode prefix as chars; append special WORD_PROBE; logits over n_word via extra head.

    class CharWord(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_word = n_word
            self.n_char = n_char
            self.pad_id = n_char
            self.max_len = 64
            self.tok = nn.Embedding(n_char + 1, 256, padding_idx=self.pad_id)
            self.pos = nn.Embedding(self.max_len, 256)
            layer = nn.TransformerEncoderLayer(256, 4, 1024, 0.1, "gelu", batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(layer, 2)
            self.word_head = nn.Linear(256, n_word, bias=False)
            self.char_head = nn.Linear(256, n_char, bias=False)

        def encode(self, ids, key_padding_mask=None):
            bsz, tmax = ids.shape
            pos = torch.arange(tmax, device=ids.device).unsqueeze(0).expand(bsz, -1)
            x = self.tok(ids) + self.pos(pos)
            mask = torch.triu(torch.ones(tmax, tmax, device=ids.device, dtype=torch.bool), diagonal=1)
            return self.tr(x, mask=mask, src_key_padding_mask=key_padding_mask)

        def logits_last_from_batch(self, ids, key_padding_mask=None):
            h = self.encode(ids, key_padding_mask)[:, -1, :]
            return self.char_head(h)

        def word_logits_from_prefix(self, id_list):
            ids = torch.tensor([id_list[-self.max_len :]], dtype=torch.long, device=self.tok.weight.device)
            h = self.encode(ids)[0, -1, :]
            return self.word_head(h)

        def logits_from_prefix(self, id_list):
            # for word_eval compatibility — word head
            return self.word_logits_from_prefix(id_list)

    def to_chars(ws):
        ids = []
        for w in ws:
            for ch in w:
                if ch in ctoi:
                    ids.append(ctoi[ch])
            ids.append(ctoi["_"])
        return ids

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = _subsample([ln for ln in train if ln.get("bucket") != "fat_train"] or train, 30000, 43)
    ev_s = _subsample(hold_seen, 350, 1431)
    ev_t = _subsample(hold_story, 250, 1433)
    fin_s = _subsample(hold_seen, min(len(hold_seen), 700), 2431)
    fin_t = _subsample(hold_story, min(len(hold_story), 400), 2433)

    # ctrl word
    def run_word():
        model = make_word_model(n_word, fps_w, device)
        fat_ex, st_ex = [], []
        for ln in fat:
            ids = [stoi_w[w] for w in ln["words"] if w in stoi_w]
            for i in range(1, len(ids)):
                fat_ex.append((ids[:i], ids[i]))
        for ln in story:
            ids = [stoi_w[w] for w in ln["words"] if w in stoi_w]
            for i in range(1, len(ids)):
                st_ex.append((ids[:i], ids[i]))

        def sample(rr):
            pool = fat_ex if rr.random() < FAT_FRAC else st_ex
            return rr.choice(pool) if pool else None

        def pref_fn(ws, t):
            return [stoi_w[w] for w in ws[:t] if w in stoi_w][-model.max_len :]

        def ev(which):
            return word_eval(model, ev_s if which == "seen" else ev_t, stoi_w, words, n_word, pref_fn)

        best, curve, n_params = train_loop(
            tag="143_word", model=model, sample_fn=sample, eval_fn=ev, steps=steps, device=device, seed=143
        )
        model.load_state_dict(best["state"])
        fs = word_eval(model, fin_s, stoi_w, words, n_word, pref_fn)
        ft = word_eval(model, fin_t, stoi_w, words, n_word, pref_fn)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": n_params,
        }

    def run_char():
        model = CharWord().to(device)
        # init char rows from letter fps when possible
        with torch.no_grad():
            for c, i in ctoi.items():
                if c == "_":
                    model.tok.weight[i].normal_(std=0.02)
                else:
                    try:
                        model.tok.weight[i] = F.normalize(stack.w(c).detach(), dim=-1)
                    except Exception:
                        model.tok.weight[i].normal_(std=0.02)
            model.word_head.weight.copy_(fps_w)

        char_fat, char_st = [], []
        for ln in fat:
            ids = to_chars(ln["words"])
            for i in range(1, len(ids)):
                char_fat.append((ids[:i], ids[i]))
        for ln in story:
            ids = to_chars(ln["words"])
            for i in range(1, len(ids)):
                char_st.append((ids[:i], ids[i]))

        # also word-boundary supervision: after _, predict next word
        wb_fat, wb_st = [], []
        for ln, bucket in ((fat, wb_fat), (story, wb_st)):
            for row in ln:
                ws = row["words"]
                for t in range(1, len(ws)):
                    if ws[t] not in stoi_w:
                        continue
                    ids = to_chars(ws[:t])  # ends with _
                    bucket.append((ids, stoi_w[ws[t]], "word"))

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rr = random.Random(143)
        best = {"story_all": -1.0, "obj": -1.0, "state": None}
        curve = []

        def pref_fn(ws, t):
            return to_chars(ws[:t])[-model.max_len :]

        def ev_hold(hold):
            return word_eval(model, hold, stoi_w, words, n_word, pref_fn)

        def snap(step):
            model.eval()
            s, stv = ev_hold(ev_s), ev_hold(ev_t)
            obj = s["obj"]["hit1"]
            rel = s["roles"].get("rel", {}).get("hit1", 0.0)
            sall = stv["roles"].get("ALL", {}).get("hit1", 0.0)
            curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
            log(f"  [143_char] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | STORY={100*sall:.1f}%")
            if (sall, obj) >= (best["story_all"], best["obj"]):
                best.update(
                    story_all=sall,
                    obj=obj,
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            model.train()

        model.train()
        snap(0)
        for step in range(1, steps + 1):
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, 1e-3, 200)
            losses = []
            for _ in range(BATCH):
                if rr.random() < 0.6:
                    pool = char_fat if rr.random() < FAT_FRAC else char_st
                    if not pool:
                        continue
                    pref, tgt = rr.choice(pool)
                    packed = collate([pref], [tgt], model.pad_id, device, model.max_len)
                    if packed is None:
                        continue
                    ids, mask, tg = packed
                    losses.append(F.cross_entropy(model.logits_last_from_batch(ids, mask), tg))
                else:
                    pool = wb_fat if rr.random() < FAT_FRAC else wb_st
                    if not pool:
                        continue
                    pref, wtgt, _ = rr.choice(pool)
                    packed = collate([pref], [0], model.pad_id, device, model.max_len)  # dummy
                    if packed is None:
                        continue
                    ids, mask, _ = packed
                    h = model.encode(ids, mask)[:, -1, :]
                    losses.append(F.cross_entropy(model.word_head(h), torch.tensor([wtgt], device=device)))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0 or step == steps:
                snap(step)

        model.load_state_dict(best["state"])
        model.eval()
        fs, ft = ev_hold(fin_s), ev_hold(fin_t)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": sum(p.numel() for p in model.parameters()),
        }

    ctrl = run_word()
    dig = run_char()
    v, lift, obj_d = verdict_vs(
        ctrl["story_all"], dig["story_all"], ctrl["seen_obj"], dig["seen_obj"], ctrl["seen_rel"], dig["seen_rel"]
    )
    return save_dig(
        143,
        "char_stream_word_confirm",
        v,
        {
            "hypothesis": "char CE + word-boundary head beats word-id CE",
            "ctrl": ctrl,
            "dig": dig,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
        },
    )


def dig_144(ctx):
    """Random bigram bank (structure ablated) vs word ctrl."""
    steps = 25000
    cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps, meta = ctx
    n_word = len(words)
    N_BG = 8000
    rr0 = random.Random(144)
    # random pairing table: id -> (a_idx, b_idx) scrambled
    pairs = [(rr0.randrange(n_word), rr0.randrange(n_word)) for _ in range(N_BG)]
    unk = N_BG
    n_emb = n_word + N_BG + 1

    def bg_id(a, b):
        # hash into random bank (not freq)
        return (hash((a, b)) % N_BG)

    def expand(ws):
        out = []
        for i, w in enumerate(ws):
            if w not in stoi:
                return []
            out.append(stoi[w])
            if i + 1 < len(ws) and ws[i + 1] in stoi:
                out.append(n_word + bg_id(w, ws[i + 1]))
        return out

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = _subsample([ln for ln in train if ln.get("bucket") != "fat_train"] or train, 40000, 44)
    ev_s = _subsample(hold_seen, 400, 1441)
    ev_t = _subsample(hold_story, 300, 1443)
    fin_s = _subsample(hold_seen, min(len(hold_seen), 800), 2441)
    fin_t = _subsample(hold_story, min(len(hold_story), 500), 2443)

    def run_word():
        model = make_word_model(n_word, fps, device)
        pools = []
        for lines in (fat, story):
            ex = []
            for ln in lines:
                ids = [stoi[w] for w in ln["words"] if w in stoi]
                for i in range(1, len(ids)):
                    ex.append((ids[:i], ids[i]))
            pools.append(ex)
        fat_ex, st_ex = pools

        def sample(rr):
            pool = fat_ex if rr.random() < FAT_FRAC else st_ex
            return rr.choice(pool) if pool else None

        def pref_fn(ws, t):
            return [stoi[w] for w in ws[:t] if w in stoi][-model.max_len :]

        def ev(which):
            return word_eval(model, ev_s if which == "seen" else ev_t, stoi, words, n_word, pref_fn)

        best, curve, n_params = train_loop(
            tag="144_word", model=model, sample_fn=sample, eval_fn=ev, steps=steps, device=device, seed=144
        )
        model.load_state_dict(best["state"])
        fs = word_eval(model, fin_s, stoi, words, n_word, pref_fn)
        ft = word_eval(model, fin_t, stoi, words, n_word, pref_fn)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": n_params,
        }

    def run_rand_bg():
        base = WordOutTransformer(n_word, n_emb, 256, 4, 2, 32, 0.1).to(device)
        base.init_word_rows(fps)
        with torch.no_grad():
            for bi, (ai, bj) in enumerate(pairs):
                base.tok.weight[n_word + bi] = F.normalize(fps[ai] + fps[bj], dim=-1)
            base.tok.weight[n_word + unk].normal_(std=0.02)

        class P(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.b = b
                self.pad_id = b.pad_id
                self.max_len = b.max_len

            def forward(self, ids, key_padding_mask=None):
                return self.b.forward(ids, key_padding_mask)

            def logits_last_from_batch(self, ids, key_padding_mask=None):
                return self.b.logits_last_from_batch(ids, key_padding_mask)

            def logits_from_prefix(self, id_list):
                return self.b.logits_from_prefix(id_list)

        model = P(base)
        fat_ex, st_ex = [], []
        for ln in fat:
            ids = expand(ln["words"])
            # targets only on word positions: rebuild with positions
            # simpler: next-id CE but only when target < n_word
            ws = ln["words"]
            for t in range(1, len(ws)):
                if ws[t] not in stoi:
                    continue
                pref = expand(ws[:t])
                if pref:
                    fat_ex.append((pref, stoi[ws[t]]))
        for ln in story:
            ws = ln["words"]
            for t in range(1, len(ws)):
                if ws[t] not in stoi:
                    continue
                pref = expand(ws[:t])
                if pref:
                    st_ex.append((pref, stoi[ws[t]]))

        def sample(rr):
            pool = fat_ex if rr.random() < FAT_FRAC else st_ex
            return rr.choice(pool) if pool else None

        def pref_fn(ws, t):
            return expand(ws[:t])[-model.max_len :]

        def ev(which):
            return word_eval(model, ev_s if which == "seen" else ev_t, stoi, words, n_word, pref_fn)

        best, curve, n_params = train_loop(
            tag="144_randbg", model=model, sample_fn=sample, eval_fn=ev, steps=steps, device=device, seed=144
        )
        model.load_state_dict(best["state"])
        fs = word_eval(model, fin_s, stoi, words, n_word, pref_fn)
        ft = word_eval(model, fin_t, stoi, words, n_word, pref_fn)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": n_params,
        }

    ctrl = run_word()
    dig = run_rand_bg()
    v, lift, obj_d = verdict_vs(
        ctrl["story_all"], dig["story_all"], ctrl["seen_obj"], dig["seen_obj"], ctrl["seen_rel"], dig["seen_rel"]
    )
    return save_dig(
        144,
        "random_bigram_bank",
        v,
        {
            "hypothesis": "137 lift was capacity not bigram structure",
            "ctrl": ctrl,
            "dig": dig,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
            "ref_137_lift_pp": 0.0167,
        },
    )


def dig_145(ctx):
    """Position shuffle bag on prefix (keep last token)."""
    steps = 25000
    cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps, meta = ctx
    n_word = len(words)
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = _subsample([ln for ln in train if ln.get("bucket") != "fat_train"] or train, 40000, 45)
    ev_s = _subsample(hold_seen, 400, 1451)
    ev_t = _subsample(hold_story, 300, 1453)
    fin_s = _subsample(hold_seen, min(len(hold_seen), 800), 2451)
    fin_t = _subsample(hold_story, min(len(hold_story), 500), 2453)

    def shuffle_pref(ids, rr):
        if len(ids) <= 2:
            return ids
        head, last = ids[:-1], ids[-1]
        head = list(head)
        rr.shuffle(head)
        return head + [last]

    def run(tag, do_shuffle):
        model = make_word_model(n_word, fps, device)
        fat_ex, st_ex = [], []
        for ln in fat:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            for i in range(1, len(ids)):
                fat_ex.append((ids[:i], ids[i]))
        for ln in story:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            for i in range(1, len(ids)):
                st_ex.append((ids[:i], ids[i]))

        def sample(rr):
            pool = fat_ex if rr.random() < FAT_FRAC else st_ex
            if not pool:
                return None
            pref, tgt = rr.choice(pool)
            if do_shuffle:
                pref = shuffle_pref(list(pref), rr)
            return pref, tgt

        def pref_fn(ws, t):
            ids = [stoi[w] for w in ws[:t] if w in stoi]
            if do_shuffle and len(ids) > 2:
                # deterministic shuffle for eval
                rr = random.Random(145 + t + hash(tuple(ws[:t])) % 10007)
                ids = shuffle_pref(ids, rr)
            return ids[-model.max_len :]

        def ev(which):
            return word_eval(model, ev_s if which == "seen" else ev_t, stoi, words, n_word, pref_fn)

        best, curve, n_params = train_loop(
            tag=tag, model=model, sample_fn=sample, eval_fn=ev, steps=steps, device=device, seed=145
        )
        model.load_state_dict(best["state"])
        fs = word_eval(model, fin_s, stoi, words, n_word, pref_fn)
        ft = word_eval(model, fin_t, stoi, words, n_word, pref_fn)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": n_params,
        }

    ctrl = run("145_ordered", False)
    dig = run("145_shuffle", True)
    v, lift, obj_d = verdict_vs(
        ctrl["story_all"], dig["story_all"], ctrl["seen_obj"], dig["seen_obj"], ctrl["seen_rel"], dig["seen_rel"]
    )
    return save_dig(
        145,
        "pos_shuffle_bag",
        v,
        {
            "hypothesis": "shuffling non-final prefix barely hurts exact@1",
            "ctrl": ctrl,
            "dig": dig,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
        },
    )


def dig_146(ctx):
    """COPY aux + emb noise; clean eval."""
    steps = 30000
    cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps, meta = ctx
    n_word = len(words)
    COPY = n_word
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = _subsample([ln for ln in train if ln.get("bucket") != "fat_train"] or train, 40000, 46)
    ev_s = _subsample(hold_seen, 400, 1461)
    ev_t = _subsample(hold_story, 300, 1463)
    fin_s = _subsample(hold_seen, min(len(hold_seen), 800), 2461)
    fin_t = _subsample(hold_story, min(len(hold_story), 500), 2463)

    def run(tag, use_copy_noise):
        base = WordOutTransformer(n_word, n_word + (1 if use_copy_noise else 0), 256, 4, 2, 20, 0.1).to(device)
        base.init_word_rows(fps)
        if use_copy_noise:
            with torch.no_grad():
                base.tok.weight[COPY] = fps.mean(0)
                base.tok.weight[COPY] = F.normalize(base.tok.weight[COPY], dim=-1)

        class M(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.b = b
                self.pad_id = b.pad_id
                self.max_len = b.max_len

            def logits_last_from_batch(self, ids, key_padding_mask=None):
                return self.b.logits_last_from_batch(ids, key_padding_mask)

            def logits_from_prefix(self, id_list):
                return self.b.logits_from_prefix(id_list)

            def forward(self, ids, key_padding_mask=None):
                return self.b.forward(ids, key_padding_mask)

            def parameters(self, recurse=True):
                return self.b.parameters(recurse)

            def state_dict(self, *a, **k):
                return self.b.state_dict(*a, **k)

            def load_state_dict(self, *a, **k):
                return self.b.load_state_dict(*a, **k)

            def train(self, mode=True):
                self.b.train(mode)
                return self

            def eval(self):
                self.b.eval()
                return self

        model = M(base)
        fat_ex, st_ex = [], []
        for ln in fat:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            for i in range(1, len(ids)):
                fat_ex.append((ids[:i], ids[i], "ce"))
                if use_copy_noise and i >= 2:
                    j = random.Random(i + len(ids)).randrange(0, i)
                    fat_ex.append((ids[:i] + [COPY], ids[j], "copy"))
        for ln in story:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            for i in range(1, len(ids)):
                st_ex.append((ids[:i], ids[i], "ce"))

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rr = random.Random(146)
        best = {"story_all": -1.0, "obj": -1.0, "state": None}
        curve = []

        def pref_fn(ws, t):
            return [stoi[w] for w in ws[:t] if w in stoi][-model.max_len :]

        def ev_hold(hold):
            return word_eval(model, hold, stoi, words, n_word, pref_fn)

        def snap(step):
            model.eval()
            s, stv = ev_hold(ev_s), ev_hold(ev_t)
            obj = s["obj"]["hit1"]
            rel = s["roles"].get("rel", {}).get("hit1", 0.0)
            sall = stv["roles"].get("ALL", {}).get("hit1", 0.0)
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
        for step in range(1, steps + 1):
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, 1e-3, 200)
            batch = []
            for _ in range(BATCH):
                pool = fat_ex if rr.random() < FAT_FRAC else st_ex
                if not pool:
                    continue
                pref, tgt, kind = rr.choice(pool)
                batch.append((pref, tgt, kind))
            if not batch:
                continue
            # noise on emb during CE steps
            if use_copy_noise:
                with torch.no_grad():
                    noise = torch.randn_like(base.tok.weight[:n_word]) * 0.05
                    base.tok.weight[:n_word].add_(noise)
                    base.tok.weight[:n_word].copy_(F.normalize(base.tok.weight[:n_word], dim=-1))
            packed = collate([b[0] for b in batch], [b[1] for b in batch], model.pad_id, device, model.max_len)
            if packed is None:
                continue
            ids, mask, tgt = packed
            loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0 or step == steps:
                # restore clean emb from fps for eval fairness? keep learned
                snap(step)

        model.load_state_dict(best["state"])
        model.eval()
        fs, ft = ev_hold(fin_s), ev_hold(fin_t)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": sum(p.numel() for p in model.parameters()),
        }

    ctrl = run("146_clean", False)
    dig = run("146_copy_noise", True)
    v, lift, obj_d = verdict_vs(
        ctrl["story_all"], dig["story_all"], ctrl["seen_obj"], dig["seen_obj"], ctrl["seen_rel"], dig["seen_rel"]
    )
    return save_dig(
        146,
        "copy_token_noise_emb",
        v,
        {
            "hypothesis": "COPY aux + emb noise strengthens next-word",
            "ctrl": ctrl,
            "dig": dig,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
        },
    )


def dig_147(ctx):
    """Blank space token + two-word CE; gate = first word exact@1."""
    steps = 25000
    cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps, meta = ctx
    n_word = len(words)
    BLANK = n_word
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = _subsample([ln for ln in train if ln.get("bucket") != "fat_train"] or train, 40000, 47)
    ev_s = _subsample(hold_seen, 400, 1471)
    ev_t = _subsample(hold_story, 300, 1473)
    fin_s = _subsample(hold_seen, min(len(hold_seen), 800), 2471)
    fin_t = _subsample(hold_story, min(len(hold_story), 500), 2473)

    def with_blanks(ids):
        out = []
        for i, x in enumerate(ids):
            out.append(x)
            if i + 1 < len(ids):
                out.append(BLANK)
        return out

    def run(tag, use_blank_two):
        n_emb = n_word + (1 if use_blank_two else 0)
        base = WordOutTransformer(n_word, n_emb, 256, 4, 2, 24, 0.1).to(device)
        base.init_word_rows(fps)
        if use_blank_two:
            with torch.no_grad():
                try:
                    base.tok.weight[BLANK] = F.normalize(stack.w("a").detach() * 0 + 0.01, dim=-1)
                except Exception:
                    base.tok.weight[BLANK].normal_(std=0.02)
                base.tok.weight[BLANK].normal_(std=0.02)
                base.tok.weight[BLANK] = F.normalize(base.tok.weight[BLANK], dim=-1)

        class M(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.b = b
                self.pad_id = b.pad_id
                self.max_len = b.max_len

            def logits_last_from_batch(self, ids, key_padding_mask=None):
                return self.b.logits_last_from_batch(ids, key_padding_mask)

            def logits_from_prefix(self, id_list):
                return self.b.logits_from_prefix(id_list)

            def parameters(self, recurse=True):
                return self.b.parameters(recurse)

            def state_dict(self, *a, **k):
                return self.b.state_dict(*a, **k)

            def load_state_dict(self, *a, **k):
                return self.b.load_state_dict(*a, **k)

            def train(self, mode=True):
                self.b.train(mode)
                return self

            def eval(self):
                self.b.eval()
                return self

        model = M(base)
        # pools: list of (pref, tgt) and optional (pref2, tgt2)
        fat_ex, st_ex = [], []
        for ln in fat:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            for i in range(1, len(ids)):
                pref = with_blanks(ids[:i]) if use_blank_two else ids[:i]
                fat_ex.append((pref, ids[i], ids[i + 1] if use_blank_two and i + 1 < len(ids) else None))
        for ln in story:
            ids = [stoi[w] for w in ln["words"] if w in stoi]
            for i in range(1, len(ids)):
                pref = with_blanks(ids[:i]) if use_blank_two else ids[:i]
                st_ex.append((pref, ids[i], ids[i + 1] if use_blank_two and i + 1 < len(ids) else None))

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rr = random.Random(147)
        best = {"story_all": -1.0, "obj": -1.0, "state": None}
        curve = []

        def pref_fn(ws, t):
            ids = [stoi[w] for w in ws[:t] if w in stoi]
            ids = with_blanks(ids) if use_blank_two else ids
            return ids[-model.max_len :]

        def ev_hold(hold):
            return word_eval(model, hold, stoi, words, n_word, pref_fn)

        def snap(step):
            model.eval()
            s, stv = ev_hold(ev_s), ev_hold(ev_t)
            obj = s["obj"]["hit1"]
            rel = s["roles"].get("rel", {}).get("hit1", 0.0)
            sall = stv["roles"].get("ALL", {}).get("hit1", 0.0)
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
        for step in range(1, steps + 1):
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, 1e-3, 200)
            losses = []
            for _ in range(BATCH):
                pool = fat_ex if rr.random() < FAT_FRAC else st_ex
                if not pool:
                    continue
                pref, t1, t2 = rr.choice(pool)
                packed = collate([pref], [t1], model.pad_id, device, model.max_len)
                if packed is None:
                    continue
                ids, mask, tgt = packed
                logits = model.logits_last_from_batch(ids, mask)
                losses.append(F.cross_entropy(logits, tgt))
                if use_blank_two and t2 is not None:
                    # teacher force one more: append t1 (and blank) then predict t2
                    pref2 = list(pref) + [t1]
                    if use_blank_two:
                        pref2 = list(pref) + [t1, BLANK]
                    packed2 = collate([pref2], [t2], model.pad_id, device, model.max_len)
                    if packed2 is not None:
                        ids2, mask2, tgt2 = packed2
                        losses.append(0.5 * F.cross_entropy(model.logits_last_from_batch(ids2, mask2), tgt2))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0 or step == steps:
                snap(step)

        model.load_state_dict(best["state"])
        model.eval()
        fs, ft = ev_hold(fin_s), ev_hold(fin_t)
        return {
            "seen_obj": fs["obj"]["hit1"],
            "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
            "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
            "curve": curve,
            "params": sum(p.numel() for p in model.parameters()),
        }

    ctrl = run("147_plain", False)
    dig = run("147_blank_two", True)
    v, lift, obj_d = verdict_vs(
        ctrl["story_all"], dig["story_all"], ctrl["seen_obj"], dig["seen_obj"], ctrl["seen_rel"], dig["seen_rel"]
    )
    return save_dig(
        147,
        "two_word_blank_space",
        v,
        {
            "hypothesis": "blank atom + two-word CE lifts first-word exact@1",
            "ctrl": ctrl,
            "dig": dig,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
        },
    )


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"141-147 weird pipeline start {datetime.now(timezone.utc).isoformat()}")
    log("Queue: reverse, mask, char, rand-bg, shuffle, copy+noise, blank+2word")
    try:
        up = wait_140()
        phrases = ensure_100k()
        ctx = setup_data(phrases, seed=272)
        results = {}
        for fn, num in (
            (dig_141, 141),
            (dig_142, 142),
            (dig_143, 143),
            (dig_144, 144),
            (dig_145, 145),
            (dig_146, 146),
            (dig_147, 147),
        ):
            log(f"\n##### START DIG {num} #####")
            try:
                results[str(num)] = fn(ctx)
            except Exception:
                log(f"DIG {num} FAILED\n{traceback.format_exc()}")
                results[str(num)] = {"verdict": "ERROR", "traceback": True}

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "upstream_140": up.get("verdict"),
            "digs": {k: {"verdict": v.get("verdict"), "lift": v.get("story_lift_pp")} for k, v in results.items()},
            "ref100": REF100,
        }
        PIPELINE_DEC.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"PIPELINE DONE {json.dumps(summary['digs'])}")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                "\n**Stages 141-147 weird digs:** "
                + ", ".join(f"{k}:{v.get('verdict')}" for k, v in results.items())
                + ". `stage141_147_weird_pipeline_decision.json`.\n"
            )
            if "Stages 141-147 weird" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
