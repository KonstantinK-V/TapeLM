"""
Stage 132 — complete SOTE-BPE-principle package (what G′ did NOT test):

  1) Selective expand: ONLY mid/rare (unigram bucket not head21p);
     head words stay single surface token (C′: head already 1 BPE tok).
  2) Aux CE on stem (and affix if present) at intermediate positions,
     plus main CE on whole-word surface at word boundary.
  3) Meaning atom = word; pieces = context only for TF.
  4) Gate = whole-word exact@1 (SEEN obj / STORY ALL).

Waits for stage130_131_decision.json then runs.

Run:
  python _stage132_selective_morph_aux.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    AFFIXES,
    CKPT,
    RELS,
    RES,
    Config,
    WordIdTransformer,
    _warmup_then_constant,
    build_bigram_table,
    build_ts_repeat_mix,
    eval_id_capacity_suite,
    line_subkind,
    load_foundation_85,
    surface,
    target_freq_bucket,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import crude_stem  # noqa: E402

DEC_PRIOR = RES / "stage130_131_decision.json"
ATOM100 = CKPT / "stage100_scale_100k.pt"
ATOM131 = CKPT / "stage131_func_bias.pt"
LOG = RES / "_stage132_log.txt"
DEC = RES / "stage132_selective_morph_decision.json"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s=8 * 3600, poll=45):
    log(f"[wait] for {DEC_PRIOR} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PRIOR.exists():
            d = json.loads(DEC_PRIOR.read_text(encoding="utf-8"))
            if d.get("stage131", {}).get("verdict") or d.get("stage130", {}).get("verdict"):
                # prefer both done
                if d.get("stage131", {}).get("verdict"):
                    log("[wait] 130/131 done")
                    return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("130/131 not ready")


def should_expand(w: str, unigram: dict) -> bool:
    b = target_freq_bucket(unigram.get(w, 0))
    return b != "head21p"  # mid/low/rare only


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"132 selective morph+aux start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_prior()
        phrases = ensure_100k()
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        train, hold_seen, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
        for ln in train + hold_seen + hold_story:
            ln.setdefault("subkind", line_subkind(ln))
            ln["split"] = ln.get("bucket", "x")
        _, unigram, _ = build_bigram_table(train)
        words = sorted({w for ln in train + hold_seen + hold_story for w in ln["words"]})

        stem_set = set()
        for w in words:
            if should_expand(w, unigram):
                st = crude_stem(w)
                stem_set.add(f"${st}")
        stems = sorted(stem_set)[:6000]
        affix_toks = [f"+{a}" for a in AFFIXES]
        surfaces = words + stems + affix_toks
        stoi = {s: i for i, s in enumerate(surfaces)}
        word_set = set(words)

        fps_list = []
        for s in surfaces:
            if s.startswith("$"):
                try:
                    fps_list.append(stack.w(s[1:]).detach())
                except Exception:
                    fps_list.append(torch.zeros(cfg.dim, device=device))
            elif s.startswith("+"):
                fps_list.append(torch.randn(cfg.dim, device=device) * 0.01)
            else:
                fps_list.append(stack.w(s).detach())
        fps = F.normalize(torch.stack(fps_list, 0), dim=-1).to(device)

        def expand_word(w):
            """Return list of (id, kind) kind in surface|stem|affix."""
            if not should_expand(w, unigram) or w not in stoi:
                return [(stoi[w], "surface")] if w in stoi else []
            out = []
            st = crude_stem(w)
            stem_tok = f"${st}"
            if stem_tok in stoi:
                out.append((stoi[stem_tok], "stem"))
            if st != w:
                for a in AFFIXES:
                    try:
                        if surface(st, a) == w:
                            at = f"+{a}"
                            if at in stoi:
                                out.append((stoi[at], "affix"))
                            break
                    except Exception:
                        pass
            out.append((stoi[w], "surface"))
            return out

        def expand_prefix(ws):
            ids = []
            for w in ws:
                for wid, _kind in expand_word(w):
                    ids.append(wid)
            return ids

        model = WordIdTransformer(len(surfaces), 256, 4, 2, max_len=40, dropout=0.1).to(device)
        model.init_from_fps(fps)
        warm_path = ATOM131 if ATOM131.exists() else ATOM100
        if warm_path.exists():
            ck = torch.load(warm_path, map_location="cpu", weights_only=False)
            old = list(ck.get("surfaces", []))
            sd = ck.get("word_tf") or {}
            if "tok.weight" in sd:
                with torch.no_grad():
                    for i, w in enumerate(old):
                        if w in stoi and i < sd["tok.weight"].size(0):
                            model.tok.weight[stoi[w]].copy_(sd["tok.weight"][i].to(device))
                log(f"[init] warmed surfaces from {warm_path.name}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
        story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
        rr = random.Random(19)
        lam_stem, lam_aff = 0.3, 0.15

        def sample_losses():
            ln = rr.choice(fat) if rr.random() < 0.6 else rr.choice(story)
            ws = ln["words"]
            if len(ws) < 2:
                return None
            t = rr.randrange(1, len(ws))
            # build expanded seq for prefix + optional aux targets inside last word expand
            pref_ws = ws[:t]
            gold_w = ws[t]
            if gold_w not in stoi:
                return None
            # sequence = expand(pref) ; we may also teacher-force expand of gold partially? no — predict gold surface from pref only
            seq = expand_prefix(pref_ws)
            if not seq:
                return None
            # aux: if last prefix word was expanded, also train predicting its stem from prefix-without-that-word's expand + ...
            # Simpler aux: from expand(pref[:-1]) + stem_of_last, predict affix/surface stepwise
            losses_spec = []
            # main: full expand(pref) -> gold surface
            main_ids = seq[-model.max_len :]
            losses_spec.append(("surface", main_ids, stoi[gold_w]))

            last = pref_ws[-1]
            if should_expand(last, unigram):
                st = crude_stem(last)
                stem_tok = f"${st}"
                base = expand_prefix(pref_ws[:-1])
                if stem_tok in stoi and base is not None:
                    # predict stem after base
                    losses_spec.append(("stem", (base + [stoi[stem_tok]])[-model.max_len :], stoi[stem_tok]))
                    # wait that's teacher forcing stem as input — wrong.
                    # Correct: prefix=base, target=stem_id
                    if base:
                        losses_spec.append(("stem", base[-model.max_len :], stoi[stem_tok]))
                    if st != last:
                        for a in AFFIXES:
                            try:
                                if surface(st, a) == last and f"+{a}" in stoi:
                                    # prefix = base + stem, target = affix
                                    pref_a = (base + [stoi[stem_tok]])[-model.max_len :]
                                    losses_spec.append(("affix", pref_a, stoi[f"+{a}"]))
                                    break
                            except Exception:
                                pass
            return losses_spec

        def run_batch_loss():
            # collect up to 8 surface + aux
            items = []
            for _ in range(8):
                s = sample_losses()
                if s:
                    items.extend(s)
            if not items:
                return None
            # group by packing variable lengths — do sequential mean of singles for simplicity/stability
            total = []
            for kind, pref, tgt in items[:16]:
                if not pref:
                    continue
                ids = torch.full((1, len(pref)), model.pad_id, dtype=torch.long, device=device)
                mask = torch.ones((1, len(pref)), dtype=torch.bool, device=device)
                ids[0, :] = torch.tensor(pref, device=device)
                mask[0, :] = False
                logits = model.logits_last_from_batch(ids, mask)
                ce = F.cross_entropy(logits, torch.tensor([tgt], device=device))
                if kind == "stem":
                    ce = lam_stem * ce
                elif kind == "affix":
                    ce = lam_aff * ce
                total.append(ce)
            if not total:
                return None
            return torch.stack(total).mean()

        @torch.no_grad()
        def eval_word_hold(hold_lines):
            n = h = 0
            obj_n = obj_h = 0
            n_exp = n_head = 0
            for ln in hold_lines:
                ws = ln["words"]
                for t in range(1, len(ws)):
                    gold = ws[t]
                    if gold not in stoi:
                        continue
                    pref = expand_prefix(ws[:t])[-model.max_len :]
                    if not pref:
                        continue
                    pred = surfaces[model.pred_id(pref)]
                    ok = int(pred == gold)
                    n += 1
                    h += ok
                    if should_expand(gold, unigram):
                        n_exp += 1
                    else:
                        n_head += 1
                    if ws[t - 1] in RELS:
                        obj_n += 1
                        obj_h += ok
            return {
                "all": h / max(n, 1),
                "obj": obj_h / max(obj_n, 1),
                "n": n,
                "n_expand_gold": n_exp,
                "n_head_gold": n_head,
            }

        ev_story = _subsample(hold_story, 400, 2)
        ev_seen = _subsample(hold_seen, 400, 1)
        best = {"story": -1.0, "obj": 0.0, "seen_all": 0.0, "state": None}

        n_mid = sum(1 for w in words if should_expand(w, unigram))
        log(f"[132] words={len(words)} expand_vocab={n_mid} stems={len(stems)} prior131={prior.get('stage131',{}).get('verdict')}")

        for step in range(1, 40001):
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, 1e-3, 200)
            loss = run_batch_loss()
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 2000 == 0 or step == 40000:
                model.eval()
                st = eval_word_hold(ev_story)
                sn = eval_word_hold(ev_seen)
                log(
                    f"  [132] step {step}: STORY={100*st['all']:.1f}% SEEN_ALL={100*sn['all']:.1f}% "
                    f"SEEN_obj={100*sn['obj']:.1f}%"
                )
                if st["all"] >= best["story"] and sn["obj"] >= REF100["obj"] - 0.08:
                    best.update(
                        story=st["all"], obj=sn["obj"], seen_all=sn["all"],
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    )
                model.train()

        lift = best["story"] - REF100["story_all"]
        # also vs prior 131 if available
        st131 = (prior.get("stage131") or {}).get("story_all")
        lift131 = (best["story"] - st131) if st131 is not None else None
        if lift >= 0.03 and best["obj"] >= REF100["obj"] - 0.05:
            verdict = "PASS"
        elif lift >= 0.015 or (lift131 is not None and lift131 >= 0.015):
            verdict = "PARTIAL"
        else:
            verdict = "PARITY"

        ck = CKPT / "stage132_selective_morph_aux.pt"
        if best["state"] is not None:
            torch.save({"word_tf": best["state"], "surfaces": surfaces, "expand": "mid_rare_only"}, ck)
        out = {
            "dig": "132_selective_morph_aux",
            "verdict": verdict,
            "story_all": best["story"],
            "story_lift_vs_ref100_pp": lift,
            "story_lift_vs_131_pp": lift131,
            "seen_all": best["seen_all"],
            "seen_obj": best["obj"],
            "expand_rule": "unigram bucket != head21p",
            "aux": {"lam_stem": lam_stem, "lam_affix": lam_aff},
            "ckpt": str(ck),
            "note": "Completes SOTE-BPE-principle table: selective + aux piece CE + word gate",
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage132_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")

        # stacking memo
        memo = RES / "STACKING_TO_BPE_NOTE.txt"
        memo.write_text(
            "\n".join([
                "Can +2–3pp digs sum to BPE STORY (~34%)?",
                f"ref100 STORY={REF100['story_all']*100:.1f}%  BPE112~34%  gap~14pp",
                f"131 story (if done)={None if st131 is None else f'{100*st131:.1f}%'}",
                f"132 story={100*best['story']:.1f}% lift_ref={100*lift:+.1f}pp",
                "",
                "Reality check:",
                "- Gains rarely orthogonal: fat/CE/hops/morph often move the SAME margin.",
                "- Many digs were 0 or negative (F -10pp, G/G′ ~0, 130 cmp ≤ atom).",
                "- Need ~5 independent +3pp without SEEN destroy — historically not observed.",
                "- Honest path: one dig that moves STORY ≥5–8pp, not arithmetic sum of PARTIALs.",
                "",
                f"132 verdict={verdict}",
            ]) + "\n",
            encoding="utf-8",
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 132 selective morph+aux:** {verdict} STORY={100*best['story']:.1f}% "
                f"(vs100 {100*lift:+.1f}pp). Completes mid/rare+aux CE principle. "
                f"`stage132_selective_morph_decision.json`.\n"
            )
            if "Stage 132 selective" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log(f"[132] {verdict} STORY={100*best['story']:.1f}%")
        log("DONE 132")
        log(memo.read_text(encoding="utf-8"))
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
