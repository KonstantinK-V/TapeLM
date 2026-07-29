"""
Stage 167 — Wiki50M stretch (after 166), if main dig finished with time left.

User: if 166 finishes too fast, pile on more digs. Priority:
  1. word_0L_last_d256     — width underfit for 0L too
  2. word_2L_d128_h2       — does depth help on wiki?
  3. word_1L_d128_h4       — heads on wiki
  4. word_1L_d128_h2_fat0  — natural (no light fat) ablation
  5. bigram_table_baseline — zero-train majority-next ceiling on HOLD

Ambiguous 166 → long soak:
  - 300k on primary contenders (1L d128, and d256 or 0L as needed)
  - 500k if still gray / curve was climbing / wall was very fast

Policy:
  - always run bigram baseline (cheap)
  - if 166.wall_hours < 6.5 → full stretch (1–4)
  - elif wall_hours < 8.5 → arms 1–2 only
  - else → baseline only + note
  - ambiguous gaps/verdict/curves → +300k / +500k soaks (even if wall was long)

Reuses corpus/helpers from Stage166.

Run:
  python _stage167_wiki50m_stretch.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import CKPT, RES, Config, load_foundation_85  # noqa: E402
from _stage109_110_slot_baseline import REF100  # noqa: E402
import _stage166_wiki50m_0l_1l_bpe_probes as s166  # noqa: E402

DEC166 = RES / "stage166_wiki50m_0l_1l_bpe_probes_decision.json"
DEC = RES / "stage167_wiki50m_stretch_decision.json"
LOG = RES / "_stage167_wiki50m_stretch_log.txt"
CORPUS = s166.CORPUS


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_166(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage166 ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC166.exists():
            d = json.loads(DEC166.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 166 done verdict={d.get('verdict')} wall={d.get('wall_hours')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 166 ... {int(time.time()-t0)}s")
    raise TimeoutError("166 not ready")


def already_done(path: Path):
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("verdict"):
            log(f"[skip] {path.name}")
            return d
    return None


def write_dec(path: Path, out: dict):
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[write] {path.name}")


def load_phrases():
    return [
        ln.strip()
        for ln in CORPUS.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def bigram_baseline(phrases, seed=167, hold_frac=0.12, max_hold=800):
    """Majority next-word given previous word — zero-train ceiling."""
    import random

    rng = random.Random(seed)
    ph = phrases[:]
    rng.shuffle(ph)
    n_hold = max(2000, int(hold_frac * len(ph)))
    hold = ph[:n_hold]
    train = ph[n_hold:]
    big = defaultdict(Counter)
    uni = Counter()
    for p in train:
        ws = p.split()
        for i, w in enumerate(ws):
            uni[w] += 1
            if i >= 1:
                big[ws[i - 1]][w] += 1
    uni_maj = uni.most_common(1)[0][0] if uni else ""
    hold = hold if len(hold) <= max_hold else rng.sample(hold, max_hold)
    n = h = 0
    for p in hold:
        ws = p.split()
        for t in range(1, len(ws)):
            prev, gold = ws[t - 1], ws[t]
            pred = big[prev].most_common(1)[0][0] if big[prev] else uni_maj
            n += 1
            h += int(pred == gold)
    return {"story_all": h / max(n, 1), "n": n, "protocol": "majority_bigram_next"}


def _curve_still_climbing(arm: dict | None, min_lift: float = 0.01) -> bool:
    """True if late curve rose vs mid (80k may be premature)."""
    if not arm:
        return False
    curve = arm.get("curve") or []
    if len(curve) < 4:
        return False
    mid = curve[len(curve) // 2].get("story_all", 0.0)
    late = curve[-1].get("story_all", 0.0)
    return (late - mid) >= min_lift


def ambiguous_166(up: dict) -> tuple[bool, list[str], int]:
    """
    Detect gray-zone 166 results → recommend long soak steps (300k or 500k).

    Returns (is_ambiguous, reasons, soak_steps).
    """
    reasons: list[str] = []
    gaps = up.get("gaps") or {}
    verdict = up.get("verdict") or ""
    arms = up.get("arms") or {}

    gap_all = abs(float(gaps.get("word_1L_d128_minus_0L_all") or 0.0))
    width = abs(float(gaps.get("word_1L_d256_minus_d128_all") or 0.0))
    ord_g = abs(float(gaps.get("order_drop_1L_d128_minus_0L") or 0.0))
    sl_g = abs(float(gaps.get("same_last_lift_1L_d128_minus_0L") or 0.0))

    if verdict == "MIXED":
        reasons.append(f"verdict={verdict}")
    if 0.008 <= gap_all < 0.03:
        reasons.append(f"attention_gap_all={100 * gap_all:.1f}pp gray")
    if 0.008 <= width < 0.025:
        reasons.append(f"width_gap={100 * width:.1f}pp gray")
    if 0.015 <= ord_g < 0.05:
        reasons.append(f"order_gap={100 * ord_g:.1f}pp gray")
    if 0.015 <= sl_g < 0.05:
        reasons.append(f"same_last_gap={100 * sl_g:.1f}pp gray")

    climbing = []
    for name in ("word_1L_d128_h2", "word_1L_d256_h2", "word_0L_last_d128"):
        if _curve_still_climbing(arms.get(name)):
            climbing.append(name)
    if climbing:
        reasons.append("curve_climbing:" + ",".join(climbing))

    # Strong clear verdicts with flat curves → not ambiguous
    if not reasons:
        return False, [], 0
    if verdict == "UNDERFIT_AT_50M" and width >= 0.025 and not climbing and verdict != "MIXED":
        # clear width signal alone — skip long soak unless other gray reasons
        gray_only = [r for r in reasons if "gray" in r or r.startswith("verdict=")]
        if not gray_only and not climbing:
            return False, [], 0
    if (
        verdict == "STILL_REDUNDANT"
        and gap_all < 0.008
        and ord_g < 0.015
        and not climbing
        and "verdict=MIXED" not in reasons
    ):
        # only non-gray / non-climb reasons would remain — treat as clear
        if all(not (r.startswith("attention") or r.startswith("order") or r.startswith("same") or r.startswith("width") or r.startswith("curve") or r.startswith("verdict")) for r in reasons):
            return False, [], 0
    if verdict in ("ATTENTION_MATTERS_ON_WIKI", "ATTENTION_AND_UNDERFIT") and gap_all >= 0.03 and not climbing:
        # clear attention; still soak if width/order gray bands present
        if not any("gray" in r for r in reasons) and not climbing:
            return False, [], 0

    wall = float(up.get("wall_hours") or 99)
    soak = 300_000
    if len(reasons) >= 3 or climbing or wall < 3.5:
        soak = 500_000
    return True, reasons, soak


def pick_long_soak_targets(up: dict) -> list[tuple[str, dict]]:
    """Which (tag, kwargs) to retrain long when ambiguous."""
    gaps = up.get("gaps") or {}
    gap_all = float(gaps.get("word_1L_d128_minus_0L_all") or 0.0)
    width = float(gaps.get("word_1L_d256_minus_d128_all") or 0.0)
    targets: list[tuple[str, dict]] = [
        ("word_1L_d128_h2_long", {"n_layer": 1, "n_head": 2, "d": 128, "kind": "word"})
    ]
    if abs(width) >= 0.008:
        targets.append(
            ("word_1L_d256_h2_long", {"n_layer": 1, "n_head": 2, "d": 256, "kind": "word"})
        )
    if abs(gap_all) < 0.03:
        targets.append(("word_0L_last_d128_long", {"d": 128, "kind": "zero"}))
    return targets


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    # tee 166 logs into stretch log when training
    _orig = s166.log

    def _tee(msg):
        _orig(msg)
        line = msg if msg.endswith("\n") else msg + "\n"
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)

    s166.log = _tee

    log(f"167 wiki stretch start {datetime.now(timezone.utc).isoformat()}")
    if already_done(DEC):
        return 0
    try:
        up = wait_166()
        wall = float(up.get("wall_hours") or 99)
        amb, amb_reasons, soak_steps = ambiguous_166(up)
        log(
            f"[policy] 166 wall_hours={wall:.2f} verdict={up.get('verdict')} "
            f"ambiguous={amb} soak_steps={soak_steps} reasons={amb_reasons}"
        )

        if wall < 6.5:
            mode = "FULL"
            want = ["0L_d256", "2L_d128", "1L_h4", "1L_fat0", "bigram"]
        elif wall < 8.5:
            mode = "PARTIAL"
            want = ["0L_d256", "2L_d128", "bigram"]
        else:
            mode = "BASELINE_ONLY"
            want = ["bigram"]
        if amb:
            want = list(dict.fromkeys(want + ["long_soak"]))
        log(f"[policy] stretch mode={mode} arms={want}")

        phrases = load_phrases()
        if not phrases:
            raise RuntimeError(f"missing corpus {CORPUS}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = Config()
        cfg.max_word_len = s166.MAX_WORD_LEN
        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

        arms = {}
        t0 = time.time()

        if "bigram" in want:
            log("\n##### STRETCH: bigram table baseline #####")
            arms["bigram_table"] = bigram_baseline(phrases)
            log(f"[done] bigram_table STORY={100*arms['bigram_table']['story_all']:.1f}%")

        if "0L_d256" in want:
            log("\n##### STRETCH: 0L last d256 #####")
            arms["word_0L_last_d256"] = s166.train_zero_arm(
                "word_0L_last_d256", phrases, device, cfg, stack, d=256, steps=s166.STRETCH_STEPS
            )

        if "2L_d128" in want:
            log("\n##### STRETCH: word 2L d128 h2 #####")
            arms["word_2L_d128_h2"] = s166.train_word_arm(
                "word_2L_d128_h2",
                phrases,
                device,
                cfg,
                stack,
                n_layer=2,
                n_head=2,
                d=128,
                steps=s166.STRETCH_STEPS,
            )

        if "1L_h4" in want:
            log("\n##### STRETCH: word 1L d128 h4 #####")
            arms["word_1L_d128_h4"] = s166.train_word_arm(
                "word_1L_d128_h4",
                phrases,
                device,
                cfg,
                stack,
                n_layer=1,
                n_head=4,
                d=128,
                steps=s166.STRETCH_STEPS,
            )

        if "1L_fat0" in want:
            log("\n##### STRETCH: word 1L d128 h2 fat=0 #####")
            old_fat = s166.TRUNK["fat_frac"]
            s166.TRUNK["fat_frac"] = 0.0
            try:
                arms["word_1L_d128_h2_fat0"] = s166.train_word_arm(
                    "word_1L_d128_h2_fat0",
                    phrases,
                    device,
                    cfg,
                    stack,
                    n_layer=1,
                    n_head=2,
                    d=128,
                    steps=s166.STRETCH_STEPS,
                )
            finally:
                s166.TRUNK["fat_frac"] = old_fat

        # If 166 was very fast, re-soak primary longer (skip if ambiguous long_soak covers it)
        if mode == "FULL" and wall < 4.0 and "long_soak" not in want:
            log("\n##### STRETCH: word 1L d128 h2 @200k long soak #####")
            arms["word_1L_d128_h2_200k"] = s166.train_word_arm(
                "word_1L_d128_h2_200k",
                phrases,
                device,
                cfg,
                stack,
                n_layer=1,
                n_head=2,
                d=128,
                steps=200_000,
            )

        # Ambiguous 166 → 300k / 500k soaks on contenders
        if "long_soak" in want and soak_steps >= 300_000:
            log(f"\n##### LONG SOAK @ {soak_steps} (ambiguous 166) #####")
            for tag, spec in pick_long_soak_targets(up):
                log(f"\n----- {tag} steps={soak_steps} -----")
                if spec["kind"] == "zero":
                    arms[tag] = s166.train_zero_arm(
                        tag, phrases, device, cfg, stack, d=spec["d"], steps=soak_steps
                    )
                else:
                    arms[tag] = s166.train_word_arm(
                        tag,
                        phrases,
                        device,
                        cfg,
                        stack,
                        n_layer=spec["n_layer"],
                        n_head=spec["n_head"],
                        d=spec["d"],
                        steps=soak_steps,
                    )
                log(f"[done] {tag} STORY={100 * arms[tag]['story_all']:.1f}% steps={soak_steps}")

            # If 300k still gray on 1L vs 0L, escalate one 500k on best word arm
            if soak_steps == 300_000:
                long_1l = arms.get("word_1L_d128_h2_long")
                long_0l = arms.get("word_0L_last_d128_long")
                if long_1l and long_0l:
                    g = abs(long_1l["story_all"] - long_0l["story_all"])
                    if 0.008 <= g < 0.03 or _curve_still_climbing(long_1l):
                        log("\n##### ESCALATE 500k on word_1L_d128 (still gray after 300k) #####")
                        arms["word_1L_d128_h2_500k"] = s166.train_word_arm(
                            "word_1L_d128_h2_500k",
                            phrases,
                            device,
                            cfg,
                            stack,
                            n_layer=1,
                            n_head=2,
                            d=128,
                            steps=500_000,
                        )

        # gaps vs 166 refs
        ref = up.get("arms") or {}
        w128 = (ref.get("word_1L_d128_h2") or {}).get("story_all")
        w256 = (ref.get("word_1L_d256_h2") or {}).get("story_all")
        z128 = (ref.get("word_0L_last_d128") or {}).get("story_all")

        deltas = {}
        if w128 is not None:
            for k, a in arms.items():
                if isinstance(a, dict) and "story_all" in a and k != "bigram_table":
                    deltas[k] = a["story_all"] - w128
            if "bigram_table" in arms:
                deltas["bigram_minus_1L_d128"] = arms["bigram_table"]["story_all"] - w128

        notes = []
        if amb:
            notes.append(f"ambiguous_long_soak={soak_steps}")
            notes.extend(amb_reasons)
        if "word_2L_d128_h2" in arms and w128 is not None:
            g = arms["word_2L_d128_h2"]["story_all"] - w128
            notes.append(f"depth_2L_vs_1L={100*g:+.1f}pp")
        if "word_0L_last_d256" in arms and z128 is not None:
            g = arms["word_0L_last_d256"]["story_all"] - z128
            notes.append(f"0L_d256_vs_d128={100*g:+.1f}pp")
        if "bigram_table" in arms and w128 is not None:
            notes.append(
                f"1L_minus_bigram={100*(w128 - arms['bigram_table']['story_all']):+.1f}pp"
            )
        long_1l = arms.get("word_1L_d128_h2_long") or arms.get("word_1L_d128_h2_500k")
        long_0l = arms.get("word_0L_last_d128_long")
        if long_1l and long_0l:
            notes.append(
                f"long_1L_minus_0L={100*(long_1l['story_all'] - long_0l['story_all']):+.1f}pp"
            )
        long_256 = arms.get("word_1L_d256_h2_long")
        if long_1l and long_256:
            notes.append(
                f"long_d256_minus_d128={100*(long_256['story_all'] - long_1l['story_all']):+.1f}pp"
            )

        verdict = f"STRETCH_{mode}"
        if amb:
            verdict = f"STRETCH_{mode}_LONGSOAK_{soak_steps // 1000}k"
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "wiki50m_stretch_after_166",
            "stretch_mode": mode,
            "ambiguous_166": amb,
            "ambiguous_reasons": amb_reasons,
            "long_soak_steps": soak_steps if amb else 0,
            "upstream_166_wall_hours": wall,
            "upstream_166_verdict": up.get("verdict"),
            "wanted": want,
            "arms": arms,
            "deltas_vs_166_1L_d128": deltas,
            "notes": notes,
            "verdict": verdict,
            "stretch_wall_hours": (time.time() - t0) / 3600,
            "ref100": REF100,
            "ref_166_stories": {"d128": w128, "d256": w256, "0L_d128": z128},
        }
        write_dec(DEC, out)
        log(f"[167] {verdict} notes={notes}")
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
