"""
Stage 262 — Does the semantic channel survive a trunk swap?

The claim the whole architecture rests on is that memory and reasoning are separable: keys live
in frozen character fp, understanding lives in the trunk, and the two meet only through W_sem.
If that is real, the trunk is replaceable. 258 showed the channel works with the curve trunk and
beats matched GPT-2 on unseen paraphrases (0.646 vs 0.276). This runs the SAME exam with a third
trunk - any HuggingFace causal LM - and changes nothing else.

Nothing about the tape moves. Keys are canonical frozen P1 fp, written from characters, so the
external model's tokenizer never touches them. The only coupling is h_t -> W_sem -> key space,
and W_sem is a single Linear whose input dim is read off the model. Retrieval only: no decode,
so there is no vocabulary to reconcile at all.

Read it as:
  external >= curve   the interface transfers; the trunk is rentable and 209's scale wall stops
                      being the blocking problem for the product story
  external >> curve   understanding was the bottleneck, exactly as predicted
  external <= curve   the channel is tuned to curve states specifically - a real negative, and
                      the one that would sink the "any reasoner" claim

  python _stage262_trunk_swap.py --model Qwen/Qwen2.5-0.5B [--smoke]
  python _stage262_trunk_swap.py --model sshleifer/tiny-gpt2 --smoke      # wiring check
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
import _stage258_semantic_query as s258
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE

RES = Path("results")
DECISION = RES / "stage262_decision.json"
MINI = RES / "stage262_mini.md"
LOG = RES / "_stage262_log.txt"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 262


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class ExternalTrunk:
    """Any HF causal LM, frozen, read for its last hidden state. Its tokenizer is used ONLY to
    feed itself - the tape is keyed on characters through P1 and never sees it."""

    def __init__(self, name: str, device, dtype=torch.float16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype if device.type == "cuda" else torch.float32
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def state(self, text: str) -> torch.Tensor | None:
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=256)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        if ids["input_ids"].numel() == 0:
            return None
        out = self.model(**ids, output_hidden_states=True)
        return out.hidden_states[-1][0, -1].detach().float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF causal LM id, frozen")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(s258.SEED)          # 258's seed: identical subjects, identical exam
    torch.manual_seed(s258.SEED)
    t0 = time.time()
    steps = args.steps or (150 if args.smoke else 600)
    n_subj = args.subjects or (12 if args.smoke else 64)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_lines = 400 if args.smoke else 6000

    log(f"Stage262 trunk swap start {datetime.now(timezone.utc).isoformat()} model={args.model}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    curve = SelfModelXL(n_char, V).to(device)
    curve.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)["model"])
    curve.eval()
    for p in curve.parameters():
        p.requires_grad_(False)
    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank = FpBank(model_can, stoi, device)

    try:
        ext = ExternalTrunk(args.model, device)
        log(f"  external trunk loaded: {args.model} hidden={ext.dim}")
    except Exception as e:  # noqa: BLE001
        log(f"  could not load {args.model}: {type(e).__name__}: {e}")
        return 1

    # ---- 258's exam, rebuilt with 258's own helpers so the comparison is the same exam ----
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_000_000 if args.smoke else 6_000_000)
    pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:max_lines]

    subjects = s258.make_subjects(pool, rng, n_subj, min(len(pool), 400))
    n_fit = n_subj // 2
    fit_subj, ev_subj = subjects[:n_fit], subjects[n_fit:]
    keys, values, owner = s258.build_tape(bank, subjects, device)
    n_slot = len(values)

    used = {su["S"] for su in subjects} | set(values)
    for ln in lines:
        if len(values) >= n_slot + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = max(0, m.start() - 120), min(len(ln), m.end() + 120)
            c = bank.ctx_fp(ln[lo:hi], exclude=e)
            if c is None:
                continue
            an = [w for w in ANCHOR_RE.findall(ln[lo : m.start()]) if w != e]
            if not an:
                continue
            keys.append(F.normalize(bank.fp([an[-1]])[0] + c, dim=-1))
            values.append(e)
            owner.append((None, None))
            used.add(e)
            if len(values) >= n_slot + n_dist:
                break
    K = torch.stack(keys, 0).to(device).float()
    owner_idx: dict[str, list] = {}
    for j, (sid, r) in enumerate(owner):
        if sid is not None:
            owner_idx.setdefault(sid, []).append((r, j))
    gold_of = {(sid, r): j for j, (sid, r) in enumerate(owner) if sid is not None}

    fit_qs = (
        s258.queries_for(fit_subj, s258.FIT_RELS, "para")
        + s258.queries_for(fit_subj, s258.FIT_RELS, "para_b")
        + s258.queries_for(fit_subj, s258.FIT_RELS, "anchored")
    )
    for q in fit_qs:
        q["gold_idx"] = gold_of[(q["sid"], q["rel"])]
    ev_seen = s258.queries_for(ev_subj, s258.FIT_RELS, "para")
    ev_unseen = s258.queries_for(ev_subj, s258.FIT_RELS, "para_hold")
    ev_anch = s258.queries_for(ev_subj, list(s258.RELS), "anchored")
    texts = sorted({q["text"] for q in fit_qs + ev_seen + ev_unseen + ev_anch})
    log(f"  tape={len(values)} ({n_slot} subject facts + {len(values)-n_slot} noise) "
        f"fit={len(fit_qs)} seen={len(ev_seen)} unseen={len(ev_unseen)} chance={s258.CHANCE:.3f}")

    st_curve = {}
    for t in texts:
        v = s258.curve_state(curve, char_table, tok, pad_id, device, t)
        if v is not None:
            st_curve[t] = v.float()
    st_ext = {}
    for t in texts:
        v = ext.state(t)
        if v is not None:
            st_ext[t] = v
    log(f"  cached states: curve {len(st_curve)} (d={next(iter(st_curve.values())).numel()}) "
        f"| external {len(st_ext)} (d={ext.dim})")

    def run(states, tag):
        W_q = L.init_query_adapter(device)
        semq = s258.SemQuery(int(next(iter(states.values())).numel()), device)
        loss = s258.train_channel(W_q, semq, bank, K, fit_qs, states, None, None,
                                  steps, args.lr, args.tau, rng, tag)
        return {
            "loss": loss,
            "seen_rel": s258.evaluate(W_q, semq, bank, K, owner_idx, ev_seen, states, True),
            "unseen_para": s258.evaluate(W_q, semq, bank, K, owner_idx, ev_unseen, states, True),
            "anchored": s258.evaluate(W_q, semq, bank, K, owner_idx, ev_anch, states, True),
        }

    W_q0 = L.init_query_adapter(device)
    fp_seen = s258.evaluate(W_q0, None, bank, K, owner_idx, ev_seen, st_curve, False)
    fp_unseen = s258.evaluate(W_q0, None, bank, K, owner_idx, ev_unseen, st_curve, False)
    log(f"fp-only: seen={fp_seen['sel_acc']:.3f} unseen={fp_unseen['sel_acc']:.3f}")

    res_curve = run(st_curve, "curve")
    log(f"curve : seen={res_curve['seen_rel']['sel_acc']:.3f} "
        f"unseen={res_curve['unseen_para']['sel_acc']:.3f} "
        f"anchored={res_curve['anchored']['sel_acc']:.3f}")
    res_ext = run(st_ext, "external")
    log(f"extern: seen={res_ext['seen_rel']['sel_acc']:.3f} "
        f"unseen={res_ext['unseen_para']['sel_acc']:.3f} "
        f"anchored={res_ext['anchored']['sel_acc']:.3f}")

    c_un = res_curve["unseen_para"]["sel_acc"]
    e_un = res_ext["unseen_para"]["sel_acc"]
    g_exam_valid = fp_unseen["sel_acc"] <= s258.CHANCE + 0.10
    g_ext_works = e_un >= s258.CHANCE + 0.20
    g_transfers = e_un >= c_un - 0.05
    g_ext_better = e_un >= c_un + 0.10

    if not g_exam_valid:
        overall = "TRUNK_SWAP_INVALID"
    elif g_ext_works and g_ext_better:
        overall = "TRUNK_SWAP_BETTER"      # understanding was the bottleneck
    elif g_ext_works and g_transfers:
        overall = "TRUNK_SWAP_OK"          # the interface is trunk-agnostic
    elif g_ext_works:
        overall = "TRUNK_SWAP_PARTIAL"     # works, but worse than curve
    else:
        overall = "TRUNK_SWAP_NO"          # channel is tuned to curve states specifically

    out = {
        "stage": 262, "overall": overall, "external_model": args.model,
        "external_hidden": ext.dim, "curve_trunk": trunk_ckpt.name,
        "steps": steps, "n_subjects": len(subjects), "tape_slots": len(values),
        "subject_slots": n_slot, "chance": s258.CHANCE,
        "fit_rels": s258.FIT_RELS, "exam_holdout": "para_hold (258 unseen_para)",
        "gates": {
            "G_exam_valid": g_exam_valid, "G_external_works": g_ext_works,
            "G_interface_transfers": g_transfers, "G_external_better": g_ext_better,
        },
        "summary": {"fp_only": {"seen_rel": fp_seen, "unseen_para": fp_unseen},
                    "curve_trunk": res_curve, "external_trunk": res_ext},
        "note": "258's exam verbatim - same seed, same subjects, same relations, same helpers - "
        "with the trunk swapped for an external frozen causal LM. Nothing about the tape moves: "
        "keys are canonical P1 fp written from characters, so the external tokenizer never "
        "touches them, and the only coupling is h_t -> W_sem -> key space. Retrieval only, so "
        "there is no vocabulary to reconcile. TRUNK_SWAP_NO would mean the channel is tuned to "
        "curve states specifically, which is the result that sinks the 'any reasoner' claim.",
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 262 trunk swap\n\n**{overall}** external={args.model} (d={ext.dim}) "
        f"slots={len(values)} chance={s258.CHANCE:.3f}\n\n"
        f"- unseen paraphrase: fp-only **{fp_unseen['sel_acc']:.3f}** | "
        f"curve **{c_un:.3f}** | external **{e_un:.3f}**\n"
        f"- seen paraphrase: curve {res_curve['seen_rel']['sel_acc']:.3f} | "
        f"external {res_ext['seen_rel']['sel_acc']:.3f}\n"
        f"- anchored: curve {res_curve['anchored']['sel_acc']:.3f} | "
        f"external {res_ext['anchored']['sel_acc']:.3f}\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
