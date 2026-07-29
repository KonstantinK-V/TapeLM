"""
Stage 111–112 after 109/110:

  111 fat_down_slot_dyn: same as 109 but fat_frac=0.45 (remove Stage98 artificial
      swing dampening; keep dynamic slot CE). Asks: does slot-dyn hold joint without fat?

  112 bpe_baseline: train corpus BPE (vocab 8k) + small GPT2 (~same width) on same
      100k windows; eval word-level next exact@1 on SEEN/STORY holds (greedy decode).
      Asks: is ~40%/20% a data/task ceiling vs ordinary BPE LM?

Run (waits for stage109_110_decision.json):
  python _stage111_112_follow.py
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors  # noqa: E402

from train import RELS, CKPT, _role  # noqa: E402
from _stage109_110_slot_baseline import (  # noqa: E402
    CORPUS_100K,
    DEC as DEC_109,
    REF100,
    RES,
    ensure_100k,
    train_run,
    _subsample,
    log as log109,
)

LOG = RES / "_stage111_112_log.txt"
DEC = RES / "stage111_112_decision.json"
BPE_TOK = RES / "stage112_bpe_tokenizer.json"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    try:
        log109(msg)
    except Exception:
        pass


def wait_109_110(timeout_s: int = 8 * 3600, poll_s: int = 40) -> dict:
    log(f"[wait] for {DEC_109} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_109.exists():
            d = json.loads(DEC_109.read_text(encoding="utf-8"))
            if d.get("stage109") and d.get("stage110"):
                log("[wait] 109/110 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("109/110 decision not ready")


def train_bpe_tokenizer(phrases: list[str], vocab_size: int = 8000) -> Tokenizer:
    if BPE_TOK.exists():
        log(f"[bpe] reuse tokenizer {BPE_TOK}")
        return Tokenizer.from_file(str(BPE_TOK))
    log(f"[bpe] train BPE vocab={vocab_size} on {len(phrases)} phrases")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=False,
    )
    tok.train_from_iterator(phrases, trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tok.token_to_id("[BOS]")), ("[EOS]", tok.token_to_id("[EOS]"))],
    )
    BPE_TOK.parent.mkdir(exist_ok=True)
    tok.save(str(BPE_TOK))
    return tok


def _encode_words(tok: Tokenizer, words: list[str], max_len: int, bos: int, eos: int, pad: int):
    """Encode word sequence as BPE ids (BOS … EOS), truncated."""
    text = " ".join(words)
    enc = tok.encode(text)
    ids = enc.ids[:max_len]
    if not ids:
        ids = [bos, eos]
    return ids


@torch.no_grad()
def bpe_pred_next_word(
    model, tok, prefix_words, device, max_new: int = 6, encode_max_len: int = 48
) -> str:
    bos = tok.token_to_id("[BOS]")
    eos = tok.token_to_id("[EOS]")
    pad = tok.token_to_id("[PAD]")
    # Cap by model n_positions so Stage152 short-window arms don't OOB.
    n_pos = int(getattr(getattr(model, "config", None), "n_positions", encode_max_len) or encode_max_len)
    cap = max(4, min(encode_max_len, n_pos))
    ids = _encode_words(tok, prefix_words, max_len=cap, bos=bos, eos=eos, pad=pad)
    # drop trailing EOS for generation continuation
    if ids and ids[-1] == eos:
        ids = ids[:-1]
    if len(ids) < 1:
        ids = [bos]
    # leave room for max_new generated pieces
    if len(ids) >= n_pos:
        ids = ids[-(n_pos - 1) :]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_tokens = []
    for _ in range(max_new):
        if x.size(1) >= n_pos:
            x = x[:, -(n_pos - 1) :]
        logits = model(x).logits[0, -1]
        nxt = int(logits.argmax())
        if nxt in (eos, pad, bos):
            break
        out_tokens.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        # word finished if piece has space or we already have a full whitespace word
        decoded = tok.decode(out_tokens).strip()
        if " " in decoded:
            return decoded.split()[0]
        # single piece that looks like a full word and next would continue — keep going
    decoded = tok.decode(out_tokens).strip()
    if not decoded:
        return ""
    return decoded.split()[0] if decoded.split() else decoded


@torch.no_grad()
def eval_bpe_word_holds(
    model, tok, hold_lines, device, max_n_lines: int = 400, encode_max_len: int = 48
):
    """Word-level next exact@1 by role + object slot (right after rel)."""
    lines = hold_lines if len(hold_lines) <= max_n_lines else _subsample(hold_lines, max_n_lines, 7)
    roles = defaultdict(lambda: {"n": 0, "h1": 0})
    obj = {"n": 0, "h1": 0}
    for ln in lines:
        ws = ln["words"]
        for t in range(1, len(ws)):
            prefix = ws[:t]
            gold = ws[t]
            pred = bpe_pred_next_word(
                model, tok, prefix, device, encode_max_len=encode_max_len
            )
            ok = int(pred == gold)
            # fake ex for _role
            ex = {"target_word": gold, "prefix_len": t}
            role = _role(ex, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h1"] += ok
            if t >= 1 and ws[t - 1] in RELS:
                obj["n"] += 1
                obj["h1"] += ok
    def pack(d):
        return {k: {"n": v["n"], "hit1": v["h1"] / max(v["n"], 1)} for k, v in d.items()}
    return {
        "roles": pack(roles),
        "obj": {"hit1": obj["h1"] / max(obj["n"], 1), "n": obj["n"]},
        "n_lines": len(lines),
    }


def train_bpe_baseline(
    phrases: list[str],
    *,
    stage: int = 112,
    tag: str = "bpe_baseline",
    ft_steps: int = 50000,
    eval_every: int = 2000,
    batch: int = 16,
    ref: dict | None = None,
) -> dict:
    from train import build_ts_repeat_mix, Config, line_subkind

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.c87_n_fat = 200
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = 40
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = 0.50
    cfg.c87_hold_frac = 0.15

    tok = train_bpe_tokenizer(phrases, vocab_size=8000)
    bos = tok.token_to_id("[BOS]")
    eos = tok.token_to_id("[EOS]")
    pad = tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()

    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    # train sequences = tokenized phrases (fat+story), next-token CE
    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    train_pool = fat_lines + story_lines
    # pre-tokenize
    seqs = []
    for ln in train_pool:
        ids = _encode_words(tok, ln["words"], max_len=48, bos=bos, eos=eos, pad=pad)
        if len(ids) >= 3:
            seqs.append(ids)
    log(f"[bpe] train seqs={len(seqs)} V={V} meta={ {k: meta[k] for k in meta if k != 'top_triple_freq'} }")

    conf = GPT2Config(
        vocab_size=V,
        n_positions=64,
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_inner=1024,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    model = GPT2LMHeadModel(conf).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[bpe] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = random.Random(272)

    ev_seen = _subsample(hold_seen, 300, 1001)  # slower eval (generate)
    ev_story = _subsample(hold_story, 200, 1003)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 600), 2001)
    fin_story = _subsample(hold_story, min(len(hold_story), 400), 2003)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), 120), 2002)

    best = {
        "step": 0, "obj": 0.0, "story_all": 0.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        seen = eval_bpe_word_holds(model, tok, ev_seen, device, max_n_lines=len(ev_seen))
        story = eval_bpe_word_holds(model, tok, ev_story, device, max_n_lines=len(ev_story))
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj_s, "rel": rel_s, "story_all": st_all})
        key = (obj_s, st_all, rel_s)
        bkey = (best["obj"], best["story_all"], 0.0)
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, story_all=st_all,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        log(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% | "
            f"STORY ALL={st_all*100:.1f}%"
        )
        model.train()

    log("=== BPE FT ===")
    snap(0)
    model.train()
    for step in range(1, ft_steps + 1):
        batch_seqs = [rng.choice(seqs) for _ in range(batch)]
        maxlen = max(len(s) for s in batch_seqs)
        x = torch.full((batch, maxlen), pad, dtype=torch.long, device=device)
        for i, s in enumerate(batch_seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        labels = x.clone()
        labels[labels == pad] = -100
        out = model(x, labels=labels)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == ft_steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = eval_bpe_word_holds(model, tok, fin_seen, device, max_n_lines=len(fin_seen))
    f_story = eval_bpe_word_holds(model, tok, fin_story, device, max_n_lines=len(fin_story))
    f_rare = eval_bpe_word_holds(model, tok, fin_rare, device, max_n_lines=len(fin_rare))
    obj_s = f_seen["obj"]["hit1"]
    rel_s = f_seen["roles"].get("rel", {}).get("hit1", 0.0)
    st_all = f_story["roles"].get("ALL", {}).get("hit1", 0.0)
    ref = ref or REF100
    obj_lift = obj_s - ref["obj"]
    st_lift = st_all - ref["story_all"]

    if abs(obj_lift) < 0.03 and abs(st_lift) < 0.03:
        verdict = "CEILING_LIKE"  # same soft ceiling as SOTE → data/task
    elif obj_s >= ref["obj"] + 0.05 or st_all >= ref["story_all"] + 0.05:
        verdict = "BPE_HIGHER"  # ordinary LM beats SOTE atom → SOTE head/init issue
    elif obj_s + 0.05 < ref["obj"]:
        verdict = "BPE_WEAKER"
    else:
        verdict = "MIXED"

    out_ckpt = CKPT / f"stage{stage}_{tag}.pt"
    out_txt = RES / f"stage{stage}_{tag}_report.txt"
    report = "\n".join([
        f"SOTE Stage {stage} — {tag} (corpus BPE + GPT2 mini)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"V_bpe={V} params={n_params} steps={ft_steps} batch={batch}",
        f"best_step={best['step']}",
        f"SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}%",
        f"STORY ALL={st_all*100:.1f}%",
        f"RARE obj={f_rare['obj']['hit1']*100:.1f}%",
        f"vs100 SOTE: obj {obj_lift*100:+.1f}pp story {st_lift*100:+.1f}pp",
        f"Verdict: {verdict}",
        "CEILING_LIKE => data/task soft ceiling; BPE_HIGHER => SOTE-specific headroom.",
    ]) + "\n"
    log("\n" + report)
    out_txt.write_text(report, encoding="utf-8")
    (RES / f"stage{stage}_{tag}_{verdict}.txt").write_text(report, encoding="utf-8")
    result = {
        "stage": stage, "tag": tag, "verdict": verdict,
        "params": n_params, "V_bpe": V,
        "seen": {"obj": obj_s, "rel": rel_s, "roles": f_seen["roles"]},
        "story": {"all": st_all, "roles": f_story["roles"]},
        "rare": {"obj": f_rare["obj"]["hit1"]},
        "obj_lift_pp": obj_lift, "story_lift_pp": st_lift,
        "curve": curve, "ckpt": str(out_ckpt), "best_step": best["step"],
        "tokenizer": str(BPE_TOK),
    }
    (RES / f"stage{stage}_{tag}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save({
        "stage": stage, "tag": tag, "verdict": verdict,
        "model": best["state"], "config": conf.to_dict(),
        "tokenizer": str(BPE_TOK),
    }, out_ckpt)
    log(f"Saved {out_ckpt}")
    return result


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage111/112 follow start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_109_110()
        phrases = ensure_100k()

        # ref: prefer 109 if held, else Stage100
        r109 = prior["stage109"]
        ref = REF100
        if r109.get("seen_held") or r109.get("seen", {}).get("rel", 0) >= 0.70:
            ref = {
                "obj": r109["seen"]["obj"],
                "rel": r109["seen"]["rel"],
                "story_all": r109["story"]["all"],
                "rare_obj": r109.get("rare", {}).get("obj", 0.12),
            }

        log("\n[111] fat_frac=0.45 + slot_dyn (drop Stage98 dampening)")
        r111 = train_run(
            stage=111,
            tag="fat45_slot_dyn",
            phrases=phrases,
            use_sote_init=True,
            slot_dynamic=True,
            ft_steps=50000,
            eval_every=1000,
            fat_frac=0.45,
            w_rel0=0.25,
            w_obj0=2.0,
            ref=ref,
        )

        log("\n[112] corpus BPE + GPT2 mini word-level eval")
        r112 = train_bpe_baseline(
            phrases,
            stage=112,
            tag="bpe_baseline",
            ft_steps=50000,
            eval_every=2500,
            batch=16,
            ref=REF100,  # compare to SOTE Stage100 absolute
        )

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prior_109_110": {
                "109": {"verdict": r109.get("verdict"), "seen": r109.get("seen"), "story": r109.get("story")},
                "110": {
                    "verdict": prior["stage110"].get("verdict"),
                    "seen": prior["stage110"].get("seen"),
                    "story": prior["stage110"].get("story"),
                },
            },
            "stage111": r111,
            "stage112": r112,
            "note": (
                "111=fat0.45+slot_dyn vs 109 fat0.75; "
                "112=BPE word-level vs Stage100 SOTE ceiling"
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stages 111–112:** fat0.45+slot_dyn obj="
                f"{r111['seen']['obj']*100:.1f}% ({r111['verdict']}); "
                f"BPE baseline obj={r112['seen']['obj']*100:.1f}% STORY="
                f"{r112['story']['all']*100:.1f}% ({r112['verdict']}). "
                f"`stage111_112_decision.json`.\n"
            )
            if "Stages 111–112" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")

        log("DONE 111/112")
        log(json.dumps({
            "111": {"verdict": r111["verdict"], "obj": r111["seen"]["obj"], "rel": r111["seen"]["rel"]},
            "112": {"verdict": r112["verdict"], "obj": r112["seen"]["obj"], "story": r112["story"]["all"]},
        }, indent=2))
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
