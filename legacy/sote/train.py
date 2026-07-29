"""
Stage 100: scale atom to ~100k TinyStories windows (6h budget).

Same recipe as Stage98 FROZEN atom: batch8, warmup200->1e-3, emb x1, fat_frac=0.75.
More data + more steps; subsampled eval. Ask: does STORY/RARE lift vs mini soft ceiling?

Results:
  data/external_tinystories_100k_85.txt
  results/stage100_scale_100k_report.txt
  results/stage100_scale_100k_metrics.json
  checkpoints/stage100_scale_100k.pt

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
SCRATCH27C = CKPT / "thought_scratch_27c_FROZEN.pt"
HOP25 = CKPT / "hop_bank_25.pt"
PARENT28P = CKPT / "stage28_plus_dirty_ft.pt"
PARENT29B = CKPT / "stage29b_disambig.pt"
PARENT29C = CKPT / "stage29c_ablate.pt"
PARENT29D = CKPT / "stage29d_combo.pt"
PARENT31 = CKPT / "stage31_chain_link.pt"
PARENT32 = CKPT / "stage32_dense_multihop_FROZEN.pt"
PARENT33 = CKPT / "stage33_stress_FROZEN.pt"
PARENT34 = CKPT / "stage34_hop3_FROZEN.pt"
PARENT35A = CKPT / "stage35a_h1_cue.pt"
PARENT35B = CKPT / "stage35b_left_instance_FROZEN.pt"
PARENT36A = CKPT / "stage36a_stress_FROZEN.pt"
PARENT36B = CKPT / "stage36b_h2_residual_FROZEN.pt"
PARENT37 = CKPT / "stage37_text_bridge_FROZEN.pt"
PARENT38 = CKPT / "stage38_decode_FROZEN.pt"
PARENT39 = CKPT / "stage39_decode_stress_FROZEN.pt"
PARENT40 = CKPT / "stage40_next_fp.pt"
PARENT41 = CKPT / "stage41_phrase_shortlist.pt"
PARENT42 = CKPT / "stage42_char_readout.pt"
WORD_PATH = CKPT / "word_memory_9_2.pt"
MORPH_PATH = CKPT / "morph_letter_9f_plusplus_FROZEN.pt"
OUT_CORPUS_DIV = CKPT / "fp_language_corpus_diverse.pt"
OUT_CORPUS_BEYOND = CKPT / "fp_language_corpus_beyond_FROZEN.pt"
PARENT47 = CKPT / "stage47_long_seq_FROZEN.pt"
PARENT48 = CKPT / "stage48_order_ctrl_FROZEN.pt"
PARENT49 = CKPT / "stage49_diverse.pt"
PARENT51 = CKPT / "stage51_hop_atom.pt"
PARENT52 = CKPT / "stage52_beyond_FROZEN.pt"
PARENT53 = CKPT / "stage53_external_FROZEN.pt"
PARENT55 = CKPT / "stage55_ft_curve.pt"
PARENT56 = CKPT / "stage56_order_harden.pt"
PARENT57F = CKPT / "stage57_rehearsal_FROZEN.pt"
PARENT59F = CKPT / "stage59_scale_1k_FROZEN.pt"
PARENT57 = CKPT / "stage57_rehearsal.pt"
TINY_EXT = ROOT / "data" / "tiny_external.txt"
MID_EXT = ROOT / "data" / "external_mid.txt"
EXT_1K = ROOT / "data" / "external_1k.txt"
EXT_2K = ROOT / "data" / "external_2k.txt"
EXT_VDENSE = ROOT / "data" / "external_verb_dense.txt"
PARENT85 = CKPT / "stage85_foundation_FROZEN.pt" if (CKPT / "stage85_foundation_FROZEN.pt").exists() else CKPT / "stage85_foundation.pt"
OUT_CKPT = CKPT / "stage100_scale_100k.pt"
OUT_TXT = RES / "stage100_scale_100k_report.txt"
OUT_JSON = RES / "stage100_scale_100k_metrics.json"
PARENT38 = CKPT / "stage38_decode_FROZEN.pt"
PARENT39 = CKPT / "stage39_decode_stress_FROZEN.pt"
FACT_BANK_EXP_F = ROOT / "data" / "external_facts_live_exp_FROZEN.txt"
CONTRACT = RES / "fp_language_contract.md"

EVAL_SEEDS = (272, 273, 274)
HOLD_EVAL_SEED = 10027
DENSE_EVAL_SEED = 10029
DIRTY_KINDS = ("drop", "swap", "trunc", "sub")
PRIMARY_K = 7
MODES = ("clean_bind",)
HUB2_MIN_PAIRS = 5
REF_36_JOINT3 = 0.885
REF_37_TEXT_QA = 0.880
REF_38_ANSWER_TEXT = 0.948
REF_39_NATIVE_TEXT = 0.910

LETTER_CHARS = [chr(c) for c in range(ord("a"), ord("z") + 1)]
DIGIT_CHARS = [chr(c) for c in range(ord("0"), ord("9") + 1)]
CHARS = LETTER_CHARS + DIGIT_CHARS + [" "]
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
    dim: int = 256
    seed: int = 272
    episode_k: int = 3
    episode_k_dense: int = 7
    n_hold_episodes: int = 80
    n_dense_episodes: int = 64
    create_below: float = 0.55
    stabilize_lo: float = 0.55
    lock_above: float = 0.92
    ema: float = 0.85
    retrieve_margin: float = 0.05
    max_phrase_len: int = 5
    max_word_len: int = 24
    disambig_topk: int = 5
    disambig_gap: float = 0.12
    edge_w: float = 0.50
    where_w: float = 0.30
    bind_w: float = 0.20
    # Stage38 decode gates
    dense_joint_min: float = 0.70
    dense_h1_min: float = 0.90
    dense_h2_given_h1_min: float = 0.85
    dense_h3_given_h2_min: float = 0.85
    dirty_bind_min: float = 0.55
    dirty_bind_within_clean_pp: float = 0.25
    dirty_hard_min: float = 0.50
    ctrl_joint_min: float = 0.85
    pass_seeds_min: int = 2
    hubby_sample: bool = True
    dirty_both_min: float = 0.40
    h1_cue_mode: str = "what_tail"
    use_left_instance: bool = True
    hub2_h1_min: float = 0.85
    k9_joint_min: float = 0.65
    native_joint_min: float = 0.55
    lexicon_min: float = 0.99
    encode_recon_min: float = 0.95
    text_qa_joint_floor: float = 0.55
    text_qa_within_ctrl_pp: float = 0.20
    projector_steps: int = 2500
    projector_lr: float = 1e-3
    word_decode_min: float = 0.95
    phrase_decode_min: float = 0.95
    answer_text_min: float = 0.70  # balanced bank
    answer_text_stress_min: float = 0.55  # native/wider soft floor
    answer_text_within_bal_pp: float = 0.20  # or within 20pp of balanced
    holdout_decode_floor: float = 0.0  # diagnostic; NN holdout often low
    n_dense_episodes_stress: int = 48  # slightly leaner for 3 banks × 3 seeds
    # Stage40 E next-fp soft gates
    next_fp_steps: int = 2500
    next_fp_lr: float = 1e-3
    next_fp_gap_min: float = 0.10  # hold - shuffle
    next_fp_abs_min: float = 0.25  # soft absolute floor
    next_fp_batch: int = 64
    # Stage41 shortlist
    shortlist_gap_min: float = 0.12  # soft: 10–15pp band
    shortlist_abs_min: float = 0.35  # above mean 1/|cands|
    shortlist_ft_steps: int = 1500  # light InfoNCE among siblings
    shortlist_min_cands: int = 2
    # Stage42 char readout
    char_steps: int = 3000
    char_lr: float = 1e-3
    char_batch: int = 64
    char_acc_gap_min: float = 0.20  # vs 1/|CHARS|
    char_exact_floor: float = 0.05  # soft; free-run exact often low
    # Stage45
    long_prefix_max: int = 8
    mask_steps: int = 2500
    mask_batch: int = 64
    # Stage46 seq order
    seq_steps: int = 3000
    seq_batch: int = 32
    seq_layers: int = 2
    seq_l2_gap_min: float = 0.10  # main gate
    seq_l1_abs_min: float = 0.25  # do not collapse L=1
    seq_order_drop_min: float = 0.05  # clean - shuffled_prefix on L>=2
    # Stage48 order hardening
    order_steps: int = 3500
    order_margin: float = 0.15
    order_drop_gate: float = 0.10  # stronger than 46/47
    order_n_perm: int = 2  # hard-neg perms per example
    # Stage52 beyond-path
    beyond_steps: int = 5000
    beyond_sample_frac: float = 0.80  # fraction of train draws from beyond kinds
    beyond_l2_gap_min: float = 0.10
    beyond_l2_abs_min: float = 0.25
    path_l2_floor: float = 0.28  # do not crush path channel
    # Stage53 external
    ext_ft_steps: int = 1500
    ext_hold_frac: float = 0.40
    ext_l2_gap_min: float = 0.10
    ext_l2_abs_min: float = 0.25
    ext_d_min: float = 0.95
    # Stage54/55 mid scale + curve dig
    mid_ft_steps: int = 3000  # budget ceiling only; early-stop restores best
    mid_hold_frac: float = 0.35
    mid_ft_lr: float = 1e-3
    mid_ft_eval_every: int = 150
    mid_ft_lrs: tuple = (1e-3, 5e-4)  # order-harden arms
    # Stage56 order harden on mid
    mid_order_n_perm: int = 2
    mid_order_margin: float = 0.15
    mid_order_weight: float = 1.0  # milder so L2 can climb with rehearsal
    mid_order_drop_gate: float = 0.10  # >=10pp; 15pp stretch
    mid_order_l2_floor: float = 0.28  # hard PASS floor (stage55 band)
    mid_order_l2_soft: float = 0.25  # PARTIAL restore floor if hard empty
    # Stage57 rehearsal
    mid_rehearse_frac: float = 0.55  # fraction of FT draws from path/tiny pool
    mid_tiny_repeat: int = 6  # oversample tiny vs path in pool
    mid_ft_lrs_57: tuple = (5e-4, 1e-3)
    mid_rehearse_from_55: bool = True  # start from L2-strong 55, not order-soft 56
    # Stage58 L1 dig
    mid_l1_frac: float = 0.65  # fraction of steps: L1-only (w0→w1)
    mid_l1_abs_min: float = 0.15  # soft PASS
    mid_l1_stretch: float = 0.20
    mid_l1_steps: int = 3000
    mid_l1_lrs: tuple = (5e-4, 1e-3)
    mid_l1_protect_w: float = 0.5  # weight on full-line protect steps' order term
    # Stage59 ~1k scale
    scale_ft_steps: int = 3500
    scale_hold_frac: float = 0.30
    scale_ft_lrs: tuple = (5e-4, 1e-3)
    scale_rehearse_frac: float = 0.50
    scale_l2_abs_min: float = 0.25
    scale_l2_gap_min: float = 0.10
    # Stage61 external hop probe
    ext_hop_joint_min: float = 0.55  # soft probe (internal ctrl is 85%)
    ext_hop_joint_stretch: float = 0.70
    ext_hop_n_eps: int = 96  # more eps at larger bank
    ext_hop_episode_k: int = 7
    ext_fact_cap: int = 120  # stage61 tiny
    ext_fact_cap_exp: int = 0  # 0 = all chain-participating LIVE atoms
    # Stage66 verb_ing isolated dig
    verb_ing_ft_steps: int = 1200
    verb_ing_ft_lr: float = 5e-4
    verb_ing_eval_every: int = 100
    verb_ing_protect_pp: float = 0.05  # max drop on noun_rel/polarity L2 vs ZS
    # Stage67 ing-as-atom
    ing_ft_steps: int = 1500
    ing_ft_lr: float = 5e-4
    ing_eval_every: int = 100
    # Stage70 rehearsal
    ing_rehearse_frac: float = 0.45  # draw noun/polarity this often
    ing_rehearse_steps: int = 2000
    # Stage71 side hint
    side_hint_steps: int = 1500
    side_hint_lr: float = 1e-3
    # Stage72 shortlist
    shortlist_min_cands_obj: int = 2
    # Stage74 dense verb
    dense_ft_steps: int = 2500
    dense_ft_lr: float = 5e-4
    dense_eval_every: int = 100
    dense_hold_frac: float = 0.25
    dense_triple_hold_frac: float = 0.20
    dense_obj_abs_min: float = 0.25
    dense_obj_gap_min: float = 0.10
    dense_rehearse_1k_frac: float = 0.25
    # Stage75 soft object
    soft_obj_steps: int = 2000
    soft_obj_lr: float = 5e-4
    soft_obj_eval_every: int = 100
    soft_obj_gap_min: float = 0.12
    soft_obj_abs_min: float = 0.35  # among shortlist
    soft_obj_abs_soft: float = 0.25
    # Stage82 RightHead (noun_rel object slot)
    right_head_steps: int = 2000
    right_head_lr: float = 1e-3
    right_head_eval_every: int = 100
    right_partial_min: float = 0.15
    right_live_min: float = 0.25
    right_protect_pp: float = 0.05
    # Stage83 noun_rel repeat scale
    nr_n_fat: int = 120
    nr_fat_copies: int = 80  # 83b: was 30
    nr_n_rare: int = 60
    nr_rare_copies: int = 3
    nr_seen_hold_frac: float = 0.20
    nr_ft_steps: int = 8000  # 83b: was 4000
    nr_ft_lr: float = 1e-3
    nr_eval_every: int = 400
    nr_seen_right_partial: float = 0.25
    nr_seen_right_live: float = 0.50
    nr_rel_protect_pp: float = 0.05
    # Stage85 foundation
    f85_word_steps: int = 5000
    f85_morph_steps: int = 3000
    f85_phrase_steps: int = 2500
    f85_lr: float = 1e-3
    f85_eval_every: int = 500
    f85_batch: int = 64
    # Stage86 capacity
    c86_max_lines: int = 5000
    c86_hold_frac: float = 0.25
    c86_ft_steps: int = 6000
    c86_ft_lr: float = 1e-3
    c86_eval_every: int = 500
    c86_partial_all: float = 0.12
    c86_pass_all: float = 0.20
    # Stage87 TinyStories + fat repeats
    c87_n_fat: int = 150
    c87_n_rare: int = 60
    c87_fat_copies: int = 60
    c87_rare_copies: int = 2
    c87_seen_hold_frac: float = 0.20
    c87_story_keep_frac: float = 0.40
    c87_hold_frac: float = 0.25
    c87_ft_steps: int = 8000
    c87_ft_lr: float = 1e-3
    c87_eval_every: int = 500
    c87_obj_partial: float = 0.25
    c87_obj_live: float = 0.50
    # Stage88 word-id Transformer
    c88_ft_steps: int = 8000
    c88_ft_lr: float = 1e-3
    c88_eval_every: int = 500
    c88_d_model: int = 256
    c88_n_layers: int = 2
    c88_n_heads: int = 4
    c88_max_len: int = 16
    c88_dropout: float = 0.1
    c88_obj_partial: float = 0.25
    c88_obj_live: float = 0.50
    # Stage89 batched word-id Transformer
    c89_ft_steps: int = 8000
    c89_ft_lr: float = 3e-4
    c89_eval_every: int = 500
    c89_batch: int = 32
    c89_d_model: int = 256
    c89_n_layers: int = 2
    c89_n_heads: int = 4
    c89_max_len: int = 16
    c89_dropout: float = 0.1
    c89_obj_partial: float = 0.25
    c89_obj_live: float = 0.50
    # Stage90 batch + Stage88 lr
    c90_ft_steps: int = 8000
    c90_ft_lr: float = 1e-3
    c90_eval_every: int = 500
    c90_batch: int = 32
    c90_d_model: int = 256
    c90_n_layers: int = 2
    c90_n_heads: int = 4
    c90_max_len: int = 16
    c90_dropout: float = 0.1
    c90_obj_partial: float = 0.25
    c90_obj_live: float = 0.50
    # Stage91 denser lr up
    c91_ft_steps: int = 8000
    c91_ft_lr: float = 2e-3
    c91_eval_every: int = 500
    c91_batch: int = 32
    c91_d_model: int = 256
    c91_n_layers: int = 2
    c91_n_heads: int = 4
    c91_max_len: int = 16
    c91_dropout: float = 0.1
    c91_obj_partial: float = 0.25
    c91_obj_live: float = 0.50
    # Stage92 SOTE schedule @1e-3
    c92_ft_steps: int = 8000
    c92_ft_lr: float = 1e-3
    c92_eval_every: int = 500
    c92_batch: int = 32
    c92_warmup: int = 500
    c92_emb_lr_mult: float = 0.2
    c92_min_lr_ratio: float = 0.1
    c92_d_model: int = 256
    c92_n_layers: int = 2
    c92_n_heads: int = 4
    c92_max_len: int = 16
    c92_dropout: float = 0.1
    c92_obj_partial: float = 0.25
    c92_obj_live: float = 0.50

    # Stage93 warmup-only @1e-3
    c93_ft_steps: int = 8000
    c93_ft_lr: float = 1e-3
    c93_eval_every: int = 500
    c93_batch: int = 32
    c93_warmup: int = 200
    c93_d_model: int = 256
    c93_n_layers: int = 2
    c93_n_heads: int = 4
    c93_max_len: int = 16
    c93_dropout: float = 0.1
    c93_obj_partial: float = 0.25
    c93_obj_live: float = 0.50

    # Stage94 emb slow isolate
    c94_ft_steps: int = 8000
    c94_ft_lr: float = 1e-3
    c94_eval_every: int = 500
    c94_batch: int = 32
    c94_warmup: int = 200
    c94_emb_lr_mult: float = 0.2
    c94_d_model: int = 256
    c94_n_layers: int = 2
    c94_n_heads: int = 4
    c94_max_len: int = 16
    c94_dropout: float = 0.1
    c94_obj_partial: float = 0.25
    c94_obj_live: float = 0.50

    # Stage95 batch sweep
    c95_ft_steps: int = 8000
    c95_ft_lr: float = 1e-3
    c95_eval_every: int = 500
    c95_warmup: int = 200
    c95_batches: tuple = (8, 16, 64)
    c95_d_model: int = 256
    c95_n_layers: int = 2
    c95_n_heads: int = 4
    c95_max_len: int = 16
    c95_dropout: float = 0.1
    c95_obj_partial: float = 0.25
    c95_obj_live: float = 0.50

    # Stage96 emb fast
    c96_ft_steps: int = 8000
    c96_ft_lr: float = 1e-3
    c96_eval_every: int = 500
    c96_batch: int = 8
    c96_warmup: int = 200
    c96_emb_lr_mult: float = 2.0
    c96_d_model: int = 256
    c96_n_layers: int = 2
    c96_n_heads: int = 4
    c96_max_len: int = 16
    c96_dropout: float = 0.1
    c96_obj_partial: float = 0.25
    c96_obj_live: float = 0.50

    # Stage97 atom+hops
    c97_ft_steps: int = 8000
    c97_ft_lr: float = 1e-3
    c97_eval_every: int = 500
    c97_batch: int = 8
    c97_warmup: int = 200
    c97_d_model: int = 256
    c97_n_layers: int = 2
    c97_n_heads: int = 4
    c97_max_len: int = 16
    c97_dropout: float = 0.1
    c97_obj_partial: float = 0.25
    c97_obj_live: float = 0.50
    c97_hop_n_eps: int = 96
    c97_hop_episode_k: int = 7
    c97_hop2_partial: float = 0.40
    c97_hop2_live: float = 0.70

    # Stage98 swing
    c98_ft_steps: int = 8000
    c98_ft_lr: float = 1e-3
    c98_eval_every: int = 250
    c98_batch: int = 8
    c98_warmup: int = 200
    c98_fat_frac: float = 0.75
    c98_patience: int = 8
    c98_d_model: int = 256
    c98_n_layers: int = 2
    c98_n_heads: int = 4
    c98_max_len: int = 16
    c98_dropout: float = 0.1
    c98_obj_partial: float = 0.25
    c98_obj_live: float = 0.50
    c98_swing_target: float = 0.30

    # Stage99 dirty-hop FT
    c99_ft_steps: int = 4000
    c99_ft_lr: float = 1e-3
    c99_eval_every: int = 500
    c99_hop_n_eps: int = 96
    c99_hop_episode_k: int = 7
    c99_dirty_mix: float = 0.6
    c99_obj_partial: float = 0.25

    # Stage100 scale 100k
    c100_max_lines: int = 100000
    c100_ft_steps: int = 50000
    c100_ft_lr: float = 1e-3
    c100_eval_every: int = 1000
    c100_batch: int = 8
    c100_warmup: int = 200
    c100_fat_frac: float = 0.75
    c100_eval_seen_n: int = 600
    c100_eval_story_n: int = 400
    c100_eval_rare_n: int = 120
    c100_d_model: int = 256
    c100_n_layers: int = 2
    c100_n_heads: int = 4
    c100_max_len: int = 16
    c100_dropout: float = 0.1
    c100_obj_partial: float = 0.25
    c100_story_lift_pp: float = 0.08
    c100_n_fat: int = 200
    c100_fat_copies: int = 40
    c100_story_keep_frac: float = 0.50
    c100_hold_frac: float = 0.15

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
    left_fp: torch.Tensor | None = None  # instance/left channel


class SoftPhraseMemory:
    """
    Phrase slots + episode chain-links / handles + edge fingerprints.

    forward_links[A] = [B, ...] where A.right == B.left
    edge_fp[(A,B)] = normalize(fp_A * fp_B)  — instance edge for hop2 fan-out
    handles[phrase] = episode-local int id
    """

    def __init__(self, dim: int, cfg: Config, device):
        self.dim = dim
        self.cfg = cfg
        self.device = device
        self.slots: dict[str, MemSlot] = {}
        self.fact_meta: dict[str, dict] = {}
        self.handles: dict[str, int] = {}
        self.forward_links: dict[str, list[str]] = {}
        self.edge_fp: dict[tuple[str, str], torch.Tensor] = {}
        self._next_handle = 0

    def top2(self, fp):
        if not self.slots:
            return None, 0.0, None, 0.0
        names = list(self.slots.keys())
        sims = fp @ torch.stack([self.slots[n].fp for n in names], 0).T
        if len(names) == 1:
            return names[0], float(sims[0]), None, 0.0
        vals, idx = torch.topk(sims, k=2)
        return names[int(idx[0])], float(vals[0]), names[int(idx[1])], float(vals[1])

    def topk(self, fp, k: int = 4):
        fp = F.normalize(fp.detach(), dim=-1)
        if not self.slots:
            return []
        names = list(self.slots.keys())
        sims = fp @ torch.stack([self.slots[n].fp for n in names], 0).T
        kk = min(k, len(names))
        vals, idx = torch.topk(sims, k=kk)
        return [(names[int(idx[i])], float(vals[i])) for i in range(kk)]

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

    def observe(self, key, fp, fact=None):
        fp = F.normalize(fp.detach(), dim=-1)
        if fact is not None:
            self.fact_meta[key] = fact
        if key not in self.handles:
            self.handles[key] = self._next_handle
            self._next_handle += 1
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

    def observe_strict(self, key, fp, fact=None, left_fp=None):
        """Episode write: never merge a distinct phrase key into another slot."""
        fp = F.normalize(fp.detach(), dim=-1)
        if fact is not None:
            self.fact_meta[key] = fact
        if key not in self.handles:
            self.handles[key] = self._next_handle
            self._next_handle += 1
        if left_fp is not None:
            left_fp = F.normalize(left_fp.detach(), dim=-1)
        if key in self.slots:
            slot = self.slots[key]
            slot.fp = F.normalize(self.cfg.ema * slot.fp + (1 - self.cfg.ema) * fp, dim=-1)
            slot.n += 1
            if left_fp is not None:
                slot.left_fp = left_fp.clone()
            return "stabilize"
        self.slots[key] = MemSlot(key, fp.clone(), n=1, locked=False, left_fp=left_fp.clone() if left_fp is not None else None)
        return "create"

    def build_chain_links(self, facts: list[dict]):
        """Write forward A->B edges and instance edge fingerprints."""
        self.forward_links = {}
        self.edge_fp = {}
        by_left: dict[str, list[str]] = {}
        for f in facts:
            by_left.setdefault(f["left"], []).append(f["phrase"])
        for f in facts:
            a = f["phrase"]
            if a not in self.slots:
                continue
            outs = [p for p in by_left.get(f["right"], []) if p != a and p in self.slots]
            if outs:
                self.forward_links[a] = outs
                fp_a = self.slots[a].fp
                for b in outs:
                    fp_b = self.slots[b].fp
                    self.edge_fp[(a, b)] = F.normalize(fp_a * fp_b, dim=-1)

    def follow_forward(self, key: str, bridge: str | None = None) -> list[str]:
        outs = list(self.forward_links.get(key, []))
        if bridge is None:
            return outs
        kept = []
        for p in outs:
            f = self.fact_meta.get(p) or parse_rel(p)
            if f and f["left"] == bridge:
                kept.append(p)
        return kept


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
    dim = encoder.emb.embedding_dim
    if not text:
        return torch.zeros(0, dim, device=device)
    ids = [CHAR2ID[c] for c in text if c in CHAR2ID]
    if not ids:
        return torch.zeros(0, dim, device=device)
    return encoder(torch.tensor(ids, device=device))


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
    assert SCRATCH27C.exists() and HOP25.exists()
    assert PARENT32.exists() or PARENT31.exists() or PARENT29D.exists() or PARENT28P.exists()
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


def load_28plus_binders(device, cfg):
    ck = torch.load(PARENT28P, map_location="cpu", weights_only=False)
    out = {}
    by = ck.get("binders_by_seed") or {}
    if by:
        for s, sd in by.items():
            h1 = Hop1PartialBinder(cfg.dim).to(device)
            h1.load_state_dict(sd["hop1"], strict=True)
            h2 = Hop2LeftBinder(cfg.dim).to(device)
            h2.load_state_dict(sd["hop2"], strict=True)
            for m in (h1, h2):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
            out[int(s)] = {"hop1": h1, "hop2": h2}
    else:
        h1 = Hop1PartialBinder(cfg.dim).to(device)
        h1.load_state_dict(ck["hop1_partial_binder"], strict=True)
        h2 = Hop2LeftBinder(cfg.dim).to(device)
        h2.load_state_dict(ck["hop2_left_binder"], strict=True)
        for m in (h1, h2):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        out[int(ck.get("best_seed", cfg.seed))] = {"hop1": h1, "hop2": h2}
    return out, ck


def build_dense_chains(bank: dict):
    """All multi-hop pairs where A.right == B.left."""
    by_left: dict[str, list] = {}
    for f in bank.values():
        by_left.setdefault(f["left"], []).append(f)
    pairs = []
    for a in bank.values():
        for b in by_left.get(a["right"], []):
            if b["phrase"] != a["phrase"]:
                pairs.append((a["phrase"], b["phrase"]))
    # stable unique
    seen, uniq = set(), []
    for ab in pairs:
        if ab not in seen:
            seen.add(ab)
            uniq.append(ab)
    return uniq


def build_dense_triples(bank: dict):
    """All A->B->C chains: A.right==B.left and B.right==C.left."""
    by_left: dict[str, list] = {}
    for f in bank.values():
        by_left.setdefault(f["left"], []).append(f)
    triples = []
    for a in bank.values():
        for b in by_left.get(a["right"], []):
            if b["phrase"] == a["phrase"]:
                continue
            for c in by_left.get(b["right"], []):
                if c["phrase"] in (a["phrase"], b["phrase"]):
                    continue
                triples.append((a["phrase"], b["phrase"], c["phrase"]))
    seen, uniq = set(), []
    for t in triples:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


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


def hop1_cue(fact, which: str, cue_mode: str):
    """
    legacy_partial: Stage34 bare obj/rel token
    what_tail: '{rel} {right}' — surface instance of WHAT without left
    """
    if cue_mode == "what_tail":
        return f"{fact['rel']} {fact['right']}", "what_tail"
    return hop1_partial(fact, which)


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


def partial_consistent(fact, partial: str, ptype: str, clean_partial: str | None = None):
    """Soft match of fact to partial cue (supports mild dirty trunc/prefix)."""
    target = clean_partial or partial
    if ptype == "what_tail":
        # exact-ish match on rel+right; dirty may truncate tokens
        parts = target.split()
        if len(parts) >= 2:
            rel, right = parts[0], parts[-1]
            pref = partial.split()
            if len(pref) >= 2:
                return (
                    (fact["rel"] == rel or fact["rel"] == pref[0] or pref[0] in fact["rel"])
                    and (
                        fact["right"] == right
                        or fact["right"] == pref[-1]
                        or pref[-1] in fact["right"]
                        or fact["right"] in pref[-1]
                    )
                )
            # single dirty token: fall back to right/rel containment
            tok = pref[0] if pref else partial
            return fact["rel"] == tok or fact["right"] == tok or tok in fact["right"]
        return False
    if ptype == "obj_only":
        r = fact["right"]
        return r == partial or r == target or partial in r or r in partial or target in r
    if ptype == "rel_only":
        return fact["rel"] == partial or fact["rel"] == target
    return True


@torch.no_grad()
def hop1_disambiguate(stack, mem, scratch, bound, partial, ptype, facts, cfg, clean_partial=None):
    """
    Collision-aware hop1:
      top-k by binder key -> keep partial-consistent episode facts
      re-rank by binder-sim + full WHAT CueBinder alignment.
    """
    ranked = mem.topk(bound, k=cfg.disambig_topk)
    if not ranked:
        key, sim, gap, conf = mem.retrieve(bound)
        return key, sim, gap, conf, "retrieve_empty"

    # Always consider consistency filter first among episode facts
    fact_by_phrase = {f["phrase"]: f for f in facts}
    cands = []
    for name, sim in ranked:
        f = fact_by_phrase.get(name) or parse_rel(name)
        if f is None:
            continue
        if partial_consistent(f, partial, ptype, clean_partial):
            cands.append((name, sim, f))
    # If top-k filter empty, scan all episode facts consistent with partial
    if not cands:
        for f in facts:
            if partial_consistent(f, partial, ptype, clean_partial) and f["phrase"] in mem.slots:
                sim = float((bound * mem.slots[f["phrase"]].fp).sum())
                cands.append((f["phrase"], sim, f))
    if not cands:
        key, sim, gap, conf = mem.retrieve(bound)
        return key, sim, gap, conf, "no_consistent"

    best = None
    for name, sim, f in cands:
        full = full_what(f)
        full_b = bind_full(stack, "what", full)
        align = float((full_b * mem.slots[name].fp).sum())
        # redesigned score: binder hit + full-cue confirmation
        score = 0.55 * sim + 0.45 * align
        scratch.write(f"h1_collide:{name}", mem.slots[name].fp, meta=f"sim={sim:.3f}|align={align:.3f}")
        if best is None or score > best["score"]:
            best = {"key": name, "sim": sim, "align": align, "score": score, "full": full, "bound": full_b}

    scratch.write(
        f"h1_disambig:{best['key']}",
        best["bound"],
        cell="complete_cue",
        meta=best["full"],
    )
    # synthetic gap vs second-best consistent
    sims = sorted([c[1] for c in cands], reverse=True)
    gap = sims[0] - sims[1] if len(sims) > 1 else sims[0]
    return best["key"], best["sim"], gap, True, "disambig"




def answer_right(key):
    f = parse_rel(key) if key else None
    return f["right"] if f else None


def full_where(fact):
    return f"{fact['left']} {fact['rel']} {fact['right']}"


def sample_episode(pairs, bank, k, rng, pair_weights=None):
    if pair_weights and len(pair_weights) == len(pairs):
        # weighted choice without numpy
        total = sum(pair_weights)
        x = rng.random() * total
        acc = 0.0
        idx = 0
        for i, w in enumerate(pair_weights):
            acc += w
            if x <= acc:
                idx = i
                break
        a_p, b_p = pairs[idx]
    else:
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


def bridge_hub_size(bank: dict, bridge: str) -> int:
    """Bank fan-in: how many facts have left == bridge."""
    return sum(1 for f in bank.values() if f["left"] == bridge)


def hub_bucket(n: int) -> str:
    if n <= 1:
        return "hub1"
    if n == 2:
        return "hub2"
    return "hub3p"


def classify_pairs_by_hub(pairs, bank):
    buckets = {"hub1": [], "hub2": [], "hub3p": []}
    for ab in pairs:
        a_p, _ = ab
        buckets[hub_bucket(bridge_hub_size(bank, bank[a_p]["right"]))].append(ab)
    return buckets


def classify_triples_by_hub(triples, bank):
    """Hub strata by first bridge (A.right) — same axis as 33b for readable map."""
    buckets = {"hub1": [], "hub2": [], "hub3p": []}
    for abc in triples:
        a_p = abc[0]
        buckets[hub_bucket(bridge_hub_size(bank, bank[a_p]["right"]))].append(abc)
    return buckets


def analyze_native_hub_structure(bank: dict, pairs):
    """Why hub2 may be empty without subsetting."""
    from collections import Counter

    left_deg = Counter(f["left"] for f in bank.values())
    rights = {f["right"] for f in bank.values()}
    hub2_lefts = sorted(w for w, d in left_deg.items() if d == 2)
    hub2_as_bridge = [w for w in hub2_lefts if w in rights]
    buckets = classify_pairs_by_hub(pairs, bank)
    bridgeable = []
    for left, deg in sorted(left_deg.items()):
        if left in rights and deg >= 1:
            inbound = sum(1 for f in bank.values() if f["right"] == left)
            if inbound:
                bridgeable.append({"left": left, "deg": deg, "inbound": inbound})
    return {
        "n_facts": len(bank),
        "left_deg_hist": dict(sorted(Counter(left_deg.values()).items())),
        "pair_hub_counts": {k: len(v) for k, v in buckets.items()},
        "hub2_lefts_n": len(hub2_lefts),
        "hub2_lefts": hub2_lefts,
        "hub2_lefts_also_rights": hub2_as_bridge,
        "bridgeable_deg_hist": dict(
            sorted(Counter(b["deg"] for b in bridgeable).items())
        ),
        "n_bridgeable": len(bridgeable),
    }


def resample_hub_balanced_bank(bank: dict, rng: random.Random):
    """
    One-shot subset: assign bridgeable lefts to target fan-in 1 / 2 / all.
    Creates mid-hub pairs that native sampling cannot (hub2 lefts never bridges).

    Inbound facts may only come from non-controlled lefts so we do not inflate
    the controlled fan-in we just set.
    """
    facts = list(bank.values())
    left_groups: dict[str, list] = {}
    for f in facts:
        left_groups.setdefault(f["left"], []).append(f)
    rights = {f["right"] for f in facts}
    usable = []
    for left, outs in left_groups.items():
        if left not in rights or not outs:
            continue
        inbound_free = [
            f for f in facts if f["right"] == left and f["left"] != left
        ]
        # defer controlled filter until assign known; require some inbound
        if inbound_free:
            usable.append(left)
    rng.shuffle(usable)

    # capacity pools
    pool1 = [L for L in usable if len(left_groups[L]) >= 1]
    pool2 = [L for L in usable if len(left_groups[L]) >= 2]
    pool3 = [L for L in usable if len(left_groups[L]) >= 3]
    if len(pool2) < 1 or len(pool1) < 1:
        return dict(bank), {
            "resample_ok": False,
            "reason": "no_capacity_for_hub1_or_hub2",
            "n_usable": len(usable),
            "n_pool2": len(pool2),
            "pair_hub_counts": {},
        }

    # pick disjoint bridges: prefer 3+/2/1 split without reuse
    rng.shuffle(pool3)
    rng.shuffle(pool2)
    rng.shuffle(pool1)
    assign: dict[str, int] = {}
    # hub3p first from deg>=3
    n3 = max(1, min(len(pool3), max(1, len(pool3) // 3)))
    for L in pool3[:n3]:
        assign[L] = 0  # keep all
    used = set(assign)
    # hub2 from deg>=2 not used
    cand2 = [L for L in pool2 if L not in used]
    n2 = max(1, min(len(cand2), max(1, len(cand2) // 2))) if cand2 else 0
    for L in cand2[:n2]:
        assign[L] = 2
    used = set(assign)
    # hub1 from remaining
    cand1 = [L for L in pool1 if L not in used]
    n1 = max(1, min(len(cand1), len(cand1))) if cand1 else 0
    # keep a few hub1, not all leftover as hub1 — rest stay uncontrolled padding sources
    n1_keep = max(1, min(len(cand1), max(n2, n3)))
    for L in cand1[:n1_keep]:
        assign[L] = 1
    used = set(assign)

    if 2 not in assign.values():
        return dict(bank), {
            "resample_ok": False,
            "reason": "could_not_assign_hub2",
            "n_usable": len(usable),
            "pair_hub_counts": {},
        }

    keep: set[str] = set()
    controlled = set(assign.keys())
    # outbound at exact fan-in
    for left, keep_n in assign.items():
        outs = list(left_groups[left])
        rng.shuffle(outs)
        selected = outs if keep_n == 0 else outs[: min(keep_n, len(outs))]
        for f in selected:
            keep.add(f["phrase"])
    # inbound only from non-controlled lefts (won't inflate hubs)
    for left in assign:
        ins = [f for f in facts if f["right"] == left and f["left"] not in controlled]
        if not ins:
            # last resort: allow any inbound not already violating keep_n
            ins = [f for f in facts if f["right"] == left and f["phrase"] in keep]
        for f in ins:
            keep.add(f["phrase"])
    # padding: non-controlled lefts only
    for f in facts:
        if f["left"] not in controlled:
            keep.add(f["phrase"])

    new_bank = {p: bank[p] for p in keep if p in bank}
    # verify controlled degrees
    deg = {}
    for left in assign:
        deg[left] = sum(1 for f in new_bank.values() if f["left"] == left)
    pairs = [(a, b) for a, b in build_dense_chains(new_bank) if a in new_bank and b in new_bank]
    buckets = classify_pairs_by_hub(pairs, new_bank)
    counts = {k: len(v) for k, v in buckets.items()}
    return new_bank, {
        "resample_ok": True,
        "n_facts_raw": len(bank),
        "n_facts_resampled": len(new_bank),
        "n_pairs_resampled": len(pairs),
        "assign_n": {
            "hub1": sum(1 for v in assign.values() if v == 1),
            "hub2": sum(1 for v in assign.values() if v == 2),
            "hub3p": sum(1 for v in assign.values() if v == 0),
        },
        "controlled_deg": deg,
        "pair_hub_counts": counts,
        "hub2_sufficient": counts.get("hub2", 0) >= HUB2_MIN_PAIRS,
    }


def sample_episode_stratified(pairs, bank, k, rng, hub_buckets, ep_i: int):
    """Round-robin hub strata so hub1/2/>=3 all appear in the report."""
    order = ("hub1", "hub2", "hub3p")
    # prefer non-empty buckets in rotation
    nonempty = [b for b in order if hub_buckets.get(b)]
    if not nonempty:
        return sample_episode(pairs, bank, k, rng)
    bname = nonempty[ep_i % len(nonempty)]
    pool_pairs = hub_buckets[bname]
    a_p, b_p = pool_pairs[rng.randrange(len(pool_pairs))]
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


def sample_episode_triple(triples, bank, k, rng):
    a_p, b_p, c_p = triples[rng.randrange(len(triples))]
    a, b, c = bank[a_p], bank[b_p], bank[c_p]
    facts = [a, b, c]
    seen = {a_p, b_p, c_p}
    pool = list(bank.values())
    while len(facts) < k:
        d = pool[rng.randrange(len(pool))]
        if d["phrase"] not in seen:
            facts.append(d)
            seen.add(d["phrase"])
    rng.shuffle(facts)
    return facts, a, b, c


def sample_episode_triple_stratified(triples, bank, k, rng, hub_buckets, ep_i: int):
    order = ("hub1", "hub2", "hub3p")
    nonempty = [b for b in order if hub_buckets.get(b)]
    if not nonempty:
        return sample_episode_triple(triples, bank, k, rng)
    bname = nonempty[ep_i % len(nonempty)]
    pool = hub_buckets[bname]
    a_p, b_p, c_p = pool[rng.randrange(len(pool))]
    a, b, c = bank[a_p], bank[b_p], bank[c_p]
    facts = [a, b, c]
    seen = {a_p, b_p, c_p}
    pool_f = list(bank.values())
    while len(facts) < k:
        d = pool_f[rng.randrange(len(pool_f))]
        if d["phrase"] not in seen:
            facts.append(d)
            seen.add(d["phrase"])
    rng.shuffle(facts)
    return facts, a, b, c


def dirty_token_hard(tok: str, rng: random.Random):
    """Harder noise: 2 sequential ops and/or aggressive trunc; separate from dirty_bind."""
    tok0 = "".join(c for c in tok.lower() if c in CHAR2ID and c != " ")
    if not tok0:
        return "x", "hard_empty"
    cur = tok0
    kinds = []
    # always aggressive trunc toward short prefix
    keep = max(1, min(len(cur) - 1, max(1, len(cur) // 3)))
    if len(cur) > 1:
        cur = cur[:keep]
        kinds.append("hard_trunc")
    # second op
    kind = DIRTY_KINDS[rng.randrange(len(DIRTY_KINDS))]
    cur2, k2 = dirty_token(cur, kind, rng)
    kinds.append(f"hard_{k2}")
    # optional third sub if still long
    if len(cur2) > 2 and rng.random() < 0.5:
        cur2, k3 = dirty_token(cur2, "sub", rng)
        kinds.append(f"hard_{k3}")
    return cur2, "+".join(kinds)


def bind_full(stack, kind, text):
    cue = safe_phrase_fp(stack, text)
    kid = torch.tensor([0 if kind == "what" else 1], device=stack.device)
    return stack.binder(cue.unsqueeze(0), kid).squeeze(0)


@torch.no_grad()
def hop2_disambiguate(stack, mem, scratch, bound, bridge: str, facts, cfg):
    ranked = mem.topk(bound, k=cfg.disambig_topk)
    if not ranked:
        key, sim, gap, conf = mem.retrieve(bound)
        return key, sim, gap, conf, "retrieve_empty"
    fact_by_phrase = {f["phrase"]: f for f in facts}
    cands = []
    for name, sim in ranked:
        f = fact_by_phrase.get(name) or parse_rel(name)
        if f is None:
            continue
        if f["left"] == bridge:
            cands.append((name, sim, f))
    if not cands:
        for f in facts:
            if f["left"] == bridge and f["phrase"] in mem.slots:
                sim = float((bound * mem.slots[f["phrase"]].fp).sum())
                cands.append((f["phrase"], sim, f))
    if not cands:
        key, sim, gap, conf = mem.retrieve(bound)
        return key, sim, gap, conf, "no_consistent"
    best = None
    for name, sim, f in cands:
        full = full_where(f)
        full_b = bind_full(stack, "where", full)
        align = float((full_b * mem.slots[name].fp).sum())
        score = 0.55 * sim + 0.45 * align
        if best is None or score > best["score"]:
            best = {"key": name, "sim": sim, "score": score, "full": full, "bound": full_b}
    scratch.write(f"h2_disambig:{best['key']}", best["bound"], cell="complete_cue", meta=best["full"])
    sims = sorted([c[1] for c in cands], reverse=True)
    gap = sims[0] - sims[1] if len(sims) > 1 else sims[0]
    return best["key"], best["sim"], gap, True, "disambig"


def maybe_h1_disambig(stack, mem, scratch, bound, partial, ptype, facts, cfg, clean_partial, key1, gap1, conf1):
    if (not conf1) or (gap1 < cfg.disambig_gap) or key1 is None:
        return hop1_disambiguate(
            stack, mem, scratch, bound, partial, ptype, facts, cfg, clean_partial=clean_partial
        )
    cons = [
        f for f in facts if partial_consistent(f, partial, ptype, clean_partial) and f["phrase"] != key1
    ]
    if cons:
        return hop1_disambiguate(
            stack, mem, scratch, bound, partial, ptype, facts, cfg, clean_partial=clean_partial
        )
    sim = float((bound * mem.slots[key1].fp).sum()) if key1 in mem.slots else 0.0
    return key1, sim, gap1, conf1, "keep"


def maybe_h2_disambig(stack, mem, scratch, bound, bridge, facts, cfg, key2, gap2, conf2):
    if (not conf2) or (gap2 < cfg.disambig_gap) or key2 is None:
        return hop2_disambiguate(stack, mem, scratch, bound, bridge, facts, cfg)
    cons = [f for f in facts if f["left"] == bridge and f["phrase"] != key2]
    if cons:
        return hop2_disambiguate(stack, mem, scratch, bound, bridge, facts, cfg)
    return key2, None, gap2, conf2, "keep"


@torch.no_grad()
def hop1_rich_pick(
    stack, mem, scratch, bound, partial, ptype, facts, cfg, clean_partial, hop1_binder, use_chain: bool,
    prefer_deep: bool = False,
):
    cands = [f for f in facts if partial_consistent(f, partial, ptype, clean_partial) and f["phrase"] in mem.slots]
    if not cands:
        key, sim, gap, conf = mem.retrieve(bound)
        return key, sim, gap, conf, "no_consistent"
    if use_chain:
        chained = [f for f in cands if f["phrase"] in mem.forward_links]
        if prefer_deep and chained:
            deep = []
            for f in chained:
                outs = mem.forward_links.get(f["phrase"], [])
                if any(t in mem.forward_links for t in outs):
                    deep.append(f)
            if deep:
                cands = deep
            elif chained:
                cands = chained
        elif chained:
            cands = chained

    # same_tail collision? (rel,right) duplicated among remaining cands
    from collections import Counter

    tail_of = lambda f: (f["rel"], f["right"])
    tail_counts = Counter(tail_of(f) for f in cands)
    colliding = len(cands) > 1 and any(n > 1 for n in tail_counts.values())
    use_inst = bool(getattr(cfg, "use_left_instance", False) and colliding)

    if use_inst:
        # prefer chain-source instances (not someone else's forward target)
        targets = set()
        for outs in mem.forward_links.values():
            targets.update(outs)
        sources = [f for f in cands if f["phrase"] not in targets]
        if sources:
            cands = sources

    best = None
    for f in cands:
        name = f["phrase"]
        slot = mem.slots[name]
        sim = float((bound * slot.fp).sum())
        full = full_what(f)
        full_b = bind_full(stack, "what", full)
        align = float((full_b * slot.fp).sum())
        if use_inst:
            left_fp = slot.left_fp if slot.left_fp is not None else stack.w(f["left"])
            inst = F.normalize(slot.fp * left_fp, dim=-1)
            inst_sim = float((bound * inst).sum())
            # left-instance dominates bare binder attractor under same_tail
            score = 0.70 * inst_sim + 0.20 * sim + 0.10 * align
            tag = "left_instance"
        else:
            score = 0.50 * sim + 0.35 * align
            tag = "rich"
        if use_chain and name in mem.forward_links:
            score += 0.05
            if prefer_deep and any(t in mem.forward_links for t in mem.forward_links.get(name, [])):
                score += 0.05
        if best is None or score > best["score"]:
            best = {"key": name, "sim": sim, "score": score, "full": full, "bound": full_b, "tag": tag}
    scratch.write(f"h1_pick:{best['key']}", best["bound"], cell="complete_cue", meta=best["full"])
    sims = sorted([float((bound * mem.slots[f["phrase"]].fp).sum()) for f in cands], reverse=True)
    gap = sims[0] - sims[1] if len(sims) > 1 else sims[0]
    return best["key"], best["sim"], gap, True, best.get("tag", "rich")



@torch.no_grad()
def hop2_via_chain(stack, mem, scratch, bound, key1, bridge, facts, cfg):
    targets = mem.follow_forward(key1, bridge=bridge) if key1 else []
    targets = [t for t in targets if t in mem.slots]
    if not targets:
        return hop2_disambiguate(stack, mem, scratch, bound, bridge, facts, cfg)
    if len(targets) == 1:
        name = targets[0]
        sim = float((bound * mem.slots[name].fp).sum())
        scratch.write(f"h2_link:{name}", mem.slots[name].fp, cell="complete_cue", meta="chain_link")
        return name, sim, 1.0, True, "chain_link"
    best = None
    for name in targets:
        f = mem.fact_meta.get(name) or parse_rel(name)
        sim = float((bound * mem.slots[name].fp).sum())
        full = full_where(f)
        full_b = bind_full(stack, "where", full)
        align = float((full_b * mem.slots[name].fp).sum())
        score = 0.55 * sim + 0.45 * align
        if best is None or score > best["score"]:
            best = {"key": name, "sim": sim, "score": score, "bound": full_b, "full": full}
    scratch.write(f"h2_link:{best['key']}", best["bound"], cell="complete_cue", meta=best["full"])
    return best["key"], best["sim"], 0.5, True, "chain_link_multi"


def hop3_via_chain(stack, mem, scratch, bound, key2, bridge, facts, cfg):
    """Same chain-link step as hop2, from key2 with bridge=right(key2)."""
    return hop2_via_chain(stack, mem, scratch, bound, key2, bridge, facts, cfg)


@torch.no_grad()
def run_episode_2hop(
    stack, cfg, facts, fact_a, fact_b, mode, which, hop1_binder=None, hop2_binder=None,
    dirty_rng=None, use_strict=True,
):
    """Ctrl / regression 2-hop (stage32/33 protocol)."""
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    scratch = ThoughtScratch(stack.device)
    for f in facts:
        if use_strict:
            mem.observe_strict(
                f["phrase"],
                stack.phrase_fp(f["phrase"].split()),
                fact=f,
                left_fp=stack.w(f["left"]),
            )
        else:
            mem.observe(f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f)
    mem.build_chain_links(facts)

    dirty = mode.startswith("dirty")
    partial_clean, ptype = hop1_partial(fact_a, which)
    partial = partial_clean
    if dirty:
        assert dirty_rng is not None
        if mode == "dirty_hard":
            partial, _ = dirty_token_hard(partial_clean, dirty_rng)
        else:
            dkind = DIRTY_KINDS[dirty_rng.randrange(len(DIRTY_KINDS))]
            partial, _ = dirty_token(partial_clean, dkind, dirty_rng)

    assert hop1_binder is not None
    cue = partial_clean if mode == "clean_bind" else partial
    cue_fp = stack.phrase_fp(cue.split()) if mode == "clean_bind" else safe_phrase_fp(stack, cue)
    pid = torch.tensor([PTYPE2ID[ptype]], device=stack.device)
    kid = torch.tensor([0], device=stack.device)
    bound = hop1_binder(cue_fp.unsqueeze(0), pid, kid).squeeze(0)
    key1, _, _, _, _ = hop1_rich_pick(
        stack, mem, scratch, bound, cue if mode != "clean_bind" else partial_clean,
        ptype, facts, cfg, partial_clean, hop1_binder, True, prefer_deep=False,
    )
    ok1 = key1 == fact_a["phrase"] and answer_right(key1) == fact_a["right"]
    bridge = answer_right(key1) if key1 else None
    ok2 = False
    if bridge is not None and hop2_binder is not None:
        bound2 = hop2_binder(safe_phrase_fp(stack, bridge).unsqueeze(0)).squeeze(0)
        key2, _, _, _, _ = hop2_via_chain(stack, mem, scratch, bound2, key1, bridge, facts, cfg)
        ok2 = key2 == fact_b["phrase"] and answer_right(key2) == fact_b["right"]
    return {"ok1": ok1, "ok2": ok2, "joint": bool(ok1 and ok2), "depth": scratch.depth()}


@torch.no_grad()
def run_episode_3hop(
    stack, cfg, facts, fact_a, fact_b, fact_c, mode, which,
    hop1_binder=None, hop2_binder=None, dirty_rng=None, use_strict=True,
    cue_mode: str = "what_tail",
):
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    scratch = ThoughtScratch(stack.device)
    for f in facts:
        if use_strict:
            mem.observe_strict(
                f["phrase"],
                stack.phrase_fp(f["phrase"].split()),
                fact=f,
                left_fp=stack.w(f["left"]),
            )
        else:
            mem.observe(f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f)
    mem.build_chain_links(facts)

    dirty = mode.startswith("dirty")
    partial_clean, ptype = hop1_cue(fact_a, which, cue_mode)
    partial = partial_clean
    dkind = None
    if dirty:
        assert dirty_rng is not None
        # dirty one token of the cue (last token preferred for what_tail)
        toks = partial_clean.split()
        ti = len(toks) - 1 if toks else 0
        if mode == "dirty_hard":
            dirty_tok, dkind = dirty_token_hard(toks[ti], dirty_rng)
        else:
            dkind = DIRTY_KINDS[dirty_rng.randrange(len(DIRTY_KINDS))]
            dirty_tok, dkind = dirty_token(toks[ti], dkind, dirty_rng)
        toks = list(toks)
        toks[ti] = dirty_tok
        partial = " ".join(toks)

    assert hop1_binder is not None and hop2_binder is not None
    if mode == "clean_bind":
        used_partial = partial_clean
        cue_fp = stack.phrase_fp(partial_clean.split())
    else:
        used_partial = partial
        cue_fp = safe_phrase_fp(stack, partial)

    if ptype == "what_tail":
        # CueBinder WHAT on the compound surface — no new FT head
        bound = bind_full(stack, "what", used_partial)
    else:
        pid = torch.tensor([PTYPE2ID[ptype]], device=stack.device)
        kid = torch.tensor([0], device=stack.device)
        bound = hop1_binder(cue_fp.unsqueeze(0), pid, kid).squeeze(0)

    scratch.write(f"h1_bind:{used_partial}", bound, cell="partial_cue", meta=f"{ptype}:{used_partial}")
    key1, sim1, gap1, conf1, _ = hop1_rich_pick(
        stack, mem, scratch, bound, used_partial if mode != "clean_bind" else partial_clean,
        ptype, facts, cfg, partial_clean, hop1_binder, True, prefer_deep=True,
    )
    ok1 = key1 == fact_a["phrase"] and answer_right(key1) == fact_a["right"]
    bridge1 = answer_right(key1) if key1 else None
    if bridge1 is not None:
        scratch.write(f"bridge1:{bridge1}", stack.w(bridge1), cell="bridge_entity", meta=bridge1)
    if key1 and key1 in mem.slots:
        scratch.write("attended_1", mem.slots[key1].fp, cell="attended_1")

    ok2, key2, bridge2 = False, None, None
    if bridge1 is not None:
        bound2 = hop2_binder(safe_phrase_fp(stack, bridge1).unsqueeze(0)).squeeze(0)
        key2, sim2, gap2, conf2, _ = hop2_via_chain(
            stack, mem, scratch, bound2, key1, bridge1, facts, cfg
        )
        ok2 = key2 == fact_b["phrase"] and answer_right(key2) == fact_b["right"]
        bridge2 = answer_right(key2) if key2 else None
        if key2 and key2 in mem.slots:
            scratch.write("attended_2", mem.slots[key2].fp, cell="attended_2")

    ok3, key3 = False, None
    if bridge2 is not None:
        scratch.write(f"bridge2:{bridge2}", stack.w(bridge2), cell="bridge_entity_2", meta=bridge2)
        bound3 = hop2_binder(safe_phrase_fp(stack, bridge2).unsqueeze(0)).squeeze(0)
        key3, sim3, gap3, conf3, _ = hop3_via_chain(
            stack, mem, scratch, bound3, key2, bridge2, facts, cfg
        )
        ok3 = key3 == fact_c["phrase"] and answer_right(key3) == fact_c["right"]
        if key3 and key3 in mem.slots:
            scratch.write("attended_3", mem.slots[key3].fp, cell="attended_3")

    return {
        "ok1": ok1,
        "ok2": ok2,
        "ok3": ok3,
        "joint2": bool(ok1 and ok2),
        "joint3": bool(ok1 and ok2 and ok3),
        "bridge1": bridge1,
        "bridge2": bridge2,
        "dirty_kind": dkind,
        "depth": scratch.depth(),
        "a": fact_a["phrase"],
        "b": fact_b["phrase"],
        "c": fact_c["phrase"],
        "mode": mode,
        "cue_mode": cue_mode,
        "partial": used_partial,
        "ptype": ptype,
    }


@torch.no_grad()
def eval_suite_2hop(stack, pairs, bank, cfg, rng, n_eps, modes, hop1_binder, hop2_binder, episode_k):
    stats = {m: {"ok1": 0, "ok2": 0, "joint": 0} for m in modes}
    for i in range(n_eps):
        facts, a, b = sample_episode(pairs, bank, episode_k, rng)
        which = "first" if (i % 2 == 0) else "second"
        ep_seed = HOLD_EVAL_SEED * 10007 + i * 17
        for m in modes:
            r = run_episode_2hop(
                stack, cfg, facts, a, b, m, which, hop1_binder, hop2_binder,
                dirty_rng=random.Random(ep_seed), use_strict=True,
            )
            stats[m]["ok1"] += int(r["ok1"])
            stats[m]["ok2"] += int(r["ok2"])
            stats[m]["joint"] += int(r["joint"])
    n = max(n_eps, 1)
    return {m: {k: stats[m][k] / n for k in ("ok1", "ok2", "joint")} for m in modes}


@torch.no_grad()
def eval_suite_3hop(
    stack, triples, bank, cfg, rng, n_eps, modes, hop1_binder, hop2_binder, episode_k,
    stratify_hub=True, cue_mode: str = "what_tail",
):
    hub_buckets = classify_triples_by_hub(triples, bank) if stratify_hub else None
    stats = {
        m: {
            "ok1": 0, "ok2": 0, "ok3": 0, "joint2": 0, "joint3": 0,
            "h1_ok": 0, "h2_given_h1": 0, "h2_ok": 0, "h3_given_h2": 0,
            "both12": 0, "h3_given_12": 0, "depth": 0,
        }
        for m in modes
    }
    strata = {
        b: {"n": 0, "ok1": 0, "ok2": 0, "ok3": 0, "joint3": 0, "h1_ok": 0, "h2_given_h1": 0,
            "h2_ok": 0, "h3_given_h2": 0, "both12": 0, "h3_given_12": 0}
        for b in ("hub1", "hub2", "hub3p")
    }
    for i in range(n_eps):
        if stratify_hub and hub_buckets is not None:
            facts, a, b, c = sample_episode_triple_stratified(triples, bank, episode_k, rng, hub_buckets, i)
        else:
            facts, a, b, c = sample_episode_triple(triples, bank, episode_k, rng)
        which = "first" if (i % 2 == 0) else "second"
        ep_seed = DENSE_EVAL_SEED * 10007 + i * 19 + (0 if which == "first" else 1)
        hub_n = bridge_hub_size(bank, a["right"])
        bucket = hub_bucket(hub_n)
        strata[bucket]["n"] += 1
        for m in modes:
            r = run_episode_3hop(
                stack, cfg, facts, a, b, c, m, which, hop1_binder, hop2_binder,
                dirty_rng=random.Random(ep_seed), use_strict=True, cue_mode=cue_mode,
            )
            s = stats[m]
            s["ok1"] += int(r["ok1"])
            s["ok2"] += int(r["ok2"])
            s["ok3"] += int(r["ok3"])
            s["joint2"] += int(r["joint2"])
            s["joint3"] += int(r["joint3"])
            s["depth"] += r["depth"]
            if r["ok1"]:
                s["h1_ok"] += 1
                s["h2_given_h1"] += int(r["ok2"])
            if r["ok2"]:
                s["h2_ok"] += 1
                s["h3_given_h2"] += int(r["ok3"])
            if r["ok1"] and r["ok2"]:
                s["both12"] += 1
                s["h3_given_12"] += int(r["ok3"])
            if m == "clean_bind":
                st = strata[bucket]
                st["ok1"] += int(r["ok1"])
                st["ok2"] += int(r["ok2"])
                st["ok3"] += int(r["ok3"])
                st["joint3"] += int(r["joint3"])
                if r["ok1"]:
                    st["h1_ok"] += 1
                    st["h2_given_h1"] += int(r["ok2"])
                if r["ok2"]:
                    st["h2_ok"] += 1
                    st["h3_given_h2"] += int(r["ok3"])
                if r["ok1"] and r["ok2"]:
                    st["both12"] += 1
                    st["h3_given_12"] += int(r["ok3"])
    n = max(n_eps, 1)

    def pack_mode(s):
        return {
            "hop1": s["ok1"] / n,
            "hop2": s["ok2"] / n,
            "hop3": s["ok3"] / n,
            "joint2": s["joint2"] / n,
            "joint3": s["joint3"] / n,
            "h2_given_h1": (s["h2_given_h1"] / s["h1_ok"]) if s["h1_ok"] else 0.0,
            "h3_given_h2": (s["h3_given_h2"] / s["h2_ok"]) if s["h2_ok"] else 0.0,
            "h3_given_12": (s["h3_given_12"] / s["both12"]) if s["both12"] else 0.0,
            "n_h1_ok": s["h1_ok"],
            "n_h2_ok": s["h2_ok"],
            "n_both12": s["both12"],
            "mean_depth": s["depth"] / n,
        }

    out_strata = {}
    for bname, st in strata.items():
        nn = max(st["n"], 1)
        out_strata[bname] = {
            "n": st["n"],
            "hop1": st["ok1"] / nn,
            "hop2": st["ok2"] / nn,
            "hop3": st["ok3"] / nn,
            "joint3": st["joint3"] / nn,
            "h2_given_h1": (st["h2_given_h1"] / st["h1_ok"]) if st["h1_ok"] else 0.0,
            "h3_given_h2": (st["h3_given_h2"] / st["h2_ok"]) if st["h2_ok"] else 0.0,
            "h3_given_12": (st["h3_given_12"] / st["both12"]) if st["both12"] else 0.0,
        }
    return {
        "n": n,
        "n_triples": len(triples),
        "episode_k": episode_k,
        "cue_mode": cue_mode,
        "modes": {m: pack_mode(stats[m]) for m in modes},
        "strata": out_strata,
        "hub_triple_counts": {k: len(v) for k, v in (hub_buckets or {}).items()} if hub_buckets else {},
    }


def seed_ok_stress(primary_k7, primary_k9, native_k7, ctrl, cfg):
    d7 = primary_k7["modes"]["clean_bind"]
    d9 = primary_k9["modes"]["clean_bind"]
    nat = native_k7["modes"]["clean_bind"]
    db = primary_k7["modes"]["dirty_bind"]
    dhard = primary_k7["modes"]["dirty_hard"]
    c = ctrl["clean_bind"]
    return (
        d7["joint3"] >= cfg.dense_joint_min
        and d7["hop1"] >= cfg.dense_h1_min
        and d7["h2_given_h1"] >= cfg.dense_h2_given_h1_min
        and d7["h3_given_h2"] >= cfg.dense_h3_given_h2_min
        and d9["joint3"] >= cfg.k9_joint_min
        and nat["joint3"] >= cfg.native_joint_min
        and (
            db["joint3"] >= cfg.dirty_bind_min
            or db["joint3"] >= d7["joint3"] - cfg.dirty_bind_within_clean_pp
        )
        and dhard["joint3"] >= cfg.dirty_hard_min
        and c["joint"] >= cfg.ctrl_joint_min
    )


def format_stress_report(cfg, device, seed_rows, parent_name, bal_meta, nat_meta):
    lines = [
        "SOTE Stage 36a report (stress plateau on 35b freeze)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"device: {device}",
        f"parent: {parent_name}",
        "Contract: observe_strict + chain-link + what_tail + left_instance. No FT.",
        f"balanced: {bal_meta}",
        f"native: {nat_meta}",
        "",
        f"gates: k7 j3>={cfg.dense_joint_min:.0%} H1>={cfg.dense_h1_min:.0%} "
        f"k9 j3>={cfg.k9_joint_min:.0%} native j3>={cfg.native_joint_min:.0%} "
        f"dirty_hard>={cfg.dirty_hard_min:.0%} ctrl2>={cfg.ctrl_joint_min:.0%}",
        "",
    ]
    best = max(seed_rows, key=lambda r: (r["ok"], r["k7"]["modes"]["clean_bind"]["joint3"]))
    lines += ["=== k-sweep balanced clean (best seed) ===", f"{'k':>4} {'H1':>7} {'H2|H1':>8} {'H3|H2':>8} {'JOINT3':>8}", "-" * 40]
    for k in K_STRESS:
        blob = best["k7"] if k == 7 else best["k9"]
        c = blob["modes"]["clean_bind"]
        lines.append(
            f"{k:>4} {c['hop1']*100:6.1f}% {c['h2_given_h1']*100:7.1f}% "
            f"{c['h3_given_h2']*100:7.1f}% {c['joint3']*100:7.1f}%"
        )
    lines += [
        "",
        "=== Native bank @ k=7 clean (best seed) ===",
        f"  hub_triples={best['native'].get('hub_triple_counts', {})}",
    ]
    cn = best["native"]["modes"]["clean_bind"]
    lines.append(
        f"  H1={cn['hop1']*100:.1f}% H2|H1={cn['h2_given_h1']*100:.1f}% "
        f"joint3={cn['joint3']*100:.1f}%"
    )
    for b in ("hub1", "hub2", "hub3p"):
        h = best["native"]["strata"][b]
        lines.append(f"  {b}: n={h['n']} H1={h['hop1']*100:.1f}% j3={h['joint3']*100:.1f}%")

    lines += ["", "=== Dirty @ k=7 balanced (best seed) ==="]
    for m in MODES:
        h = best["k7"]["modes"][m]
        lines.append(f"  {m}: H1={h['hop1']*100:.1f}% joint3={h['joint3']*100:.1f}%")

    lines += ["", "=== Hub strata balanced k=7 (best seed) ==="]
    for b in ("hub1", "hub2", "hub3p"):
        h = best["k7"]["strata"][b]
        lines.append(
            f"  {b}: n={h['n']} H1={h['hop1']*100:.1f}% H2|H1={h['h2_given_h1']*100:.1f}% "
            f"j3={h['joint3']*100:.1f}%"
        )

    lines += ["", "=== Multi-seed ==="]
    n_pass = sum(1 for r in seed_rows if r["ok"])
    for r in seed_rows:
        tag = "PASS" if r["ok"] else "MISS"
        c7 = r["k7"]["modes"]["clean_bind"]
        c9 = r["k9"]["modes"]["clean_bind"]
        cn = r["native"]["modes"]["clean_bind"]
        lines.append(
            f"  seed {r['seed']}: {tag} k7_j3={c7['joint3']*100:.1f}% k9_j3={c9['joint3']*100:.1f}% "
            f"nat_j3={cn['joint3']*100:.1f}% hub2_H2|H1={r['k7']['strata']['hub2']['h2_given_h1']*100:.1f}% "
            f"ctrl2={r['ctrl']['clean_bind']['joint']*100:.1f}%"
        )
    mean7 = sum(r["k7"]["modes"]["clean_bind"]["joint3"] for r in seed_rows) / len(seed_rows)
    mean9 = sum(r["k9"]["modes"]["clean_bind"]["joint3"] for r in seed_rows) / len(seed_rows)
    mean_nat = sum(r["native"]["modes"]["clean_bind"]["joint3"] for r in seed_rows) / len(seed_rows)
    mean_h2gh1_hub2 = sum(r["k7"]["strata"]["hub2"]["h2_given_h1"] for r in seed_rows) / len(seed_rows)
    overall = n_pass >= cfg.pass_seeds_min
    lines.append(
        f"  aggregate: {n_pass}/{len(seed_rows)} pass; mean k7_j3={mean7*100:.1f}% "
        f"k9_j3={mean9*100:.1f}% nat_j3={mean_nat*100:.1f}% "
        f"(vs 35b j3~{REF_35B_JOINT3*100:.1f}%: {(mean7-REF_35B_JOINT3)*100:+.1f}pp); "
        f"hub2 H2|H1={mean_h2gh1_hub2*100:.1f}%"
    )
    lines += ["", "=== Verdict 36a ==="]
    if overall:
        lines.append(
            f"  PASS: 35b holds under k9/dirty/native stress "
            f"(k7_j3~{mean7*100:.0f}%, k9~{mean9*100:.0f}%, nat~{mean_nat*100:.0f}%)."
        )
        if mean_h2gh1_hub2 < 0.97:
            lines.append(
                f"  note: residual H2 on hub2 (H2|H1~{mean_h2gh1_hub2*100:.0f}%) → Stage 36b."
            )
    else:
        lines.append("  FAIL: stress broke 35b plateau; inspect k9/native/dirty.")
    return "\n".join(lines) + "\n", overall, best, {
        "mean7": mean7, "mean9": mean9, "mean_nat": mean_nat, "hub2_h2gh1": mean_h2gh1_hub2
    }


def diagnose_h2_residual(stack, cfg, binders, triples, bank, n=63):
    """Map hub2 H2 misses under 35b contract (H1 usually ok)."""
    from collections import Counter

    hub_b = classify_triples_by_hub(triples, bank)
    hub2_only = {"hub2": hub_b["hub2"]} if hub_b["hub2"] else None
    if not hub2_only:
        return {"ok": False, "reason": "no_hub2_triples", "fail": {}}

    hop1, hop2 = binders[272]["hop1"], binders[272]["hop2"]
    rng = random.Random(DENSE_EVAL_SEED)
    fail = Counter()
    examples = []
    for i in range(n):
        facts, a, b, c = sample_episode_triple_stratified(triples, bank, PRIMARY_K, rng, hub2_only, i)
        which = "first" if i % 2 == 0 else "second"
        r = run_episode_3hop(
            stack, cfg, facts, a, b, c, "clean_bind", which, hop1, hop2,
            dirty_rng=random.Random(i), use_strict=True, cue_mode="what_tail",
        )
        if not r["ok1"]:
            fail["h1_miss"] += 1
            continue
        if r["ok2"]:
            fail["ok12"] += 1
            continue
        # H1 ok, H2 miss
        mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
        for f in facts:
            mem.observe_strict(
                f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f, left_fp=stack.w(f["left"])
            )
        mem.build_chain_links(facts)
        outs = mem.follow_forward(a["phrase"], bridge=a["right"]) if a["phrase"] in mem.forward_links else []
        outs = [t for t in outs if t in mem.slots]
        gold_b = b["phrase"]
        if not outs:
            kind = "empty_link"
        elif gold_b not in outs:
            kind = "gold_not_in_link"
        elif len(outs) == 1 and outs[0] != gold_b:
            kind = "single_wrong"
        elif len(outs) > 1:
            kind = "multi_link_pick_wrong"
        else:
            kind = "other"
        fail[kind] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "kind": kind,
                    "a": a["phrase"],
                    "b": gold_b,
                    "outs": outs,
                    "bridge": a["right"],
                    "n_outs": len(outs),
                }
            )
    return {
        "ok": True,
        "n": n,
        "fail": dict(fail),
        "h2_given_h1": fail["ok12"] / max(fail["ok12"] + sum(v for k, v in fail.items() if k != "ok12" and k != "h1_miss"), 1),
        "examples": examples,
        "hub2_n_triples": len(hub_b["hub2"]),
    }


def hop2_via_chain_instance(stack, mem, scratch, bound, key1, bridge, facts, cfg):
    """
    36b: when multiple forward targets share bridge, prefer chain-source-ish /
    left-ink among targets (same idea as H1 left_instance, on hop2 cands).
    """
    targets = mem.follow_forward(key1, bridge=bridge) if key1 else []
    targets = [t for t in targets if t in mem.slots]
    if not targets:
        return hop2_disambiguate(stack, mem, scratch, bound, bridge, facts, cfg)
    if len(targets) == 1:
        name = targets[0]
        sim = float((bound * mem.slots[name].fp).sum())
        scratch.write(f"h2_link:{name}", mem.slots[name].fp, cell="complete_cue", meta="chain_link")
        return name, sim, 1.0, True, "chain_link"

    # multi: left-instance among link targets
    link_targets = set()
    for outs in mem.forward_links.values():
        link_targets.update(outs)
    best = None
    for name in targets:
        f = mem.fact_meta.get(name) or parse_rel(name)
        slot = mem.slots[name]
        sim = float((bound * slot.fp).sum())
        left_fp = slot.left_fp if slot.left_fp is not None else stack.w(f["left"])
        inst = F.normalize(slot.fp * left_fp, dim=-1)
        inst_sim = float((bound * inst).sum())
        full = full_where(f)
        full_b = bind_full(stack, "where", full)
        align = float((full_b * slot.fp).sum())
        score = 0.55 * inst_sim + 0.25 * sim + 0.20 * align
        # prefer targets that themselves continue (deep) slightly
        if name in mem.forward_links:
            score += 0.05
        if best is None or score > best["score"]:
            best = {"key": name, "sim": sim, "score": score, "bound": full_b, "full": full}
    scratch.write(f"h2_link_inst:{best['key']}", best["bound"], cell="complete_cue", meta=best["full"])
    return best["key"], best["sim"], 0.5, True, "chain_link_instance"


def run_episode_3hop_h2inst(
    stack, cfg, facts, fact_a, fact_b, fact_c, mode, which,
    hop1_binder=None, hop2_binder=None, dirty_rng=None, use_strict=True, cue_mode="what_tail",
):
    """Like run_episode_3hop but hop2 uses left-instance multi-link rank."""
    # reuse write path by calling base then... better inline critical hop2 swap
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    scratch = ThoughtScratch(stack.device)
    for f in facts:
        if use_strict:
            mem.observe_strict(
                f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f, left_fp=stack.w(f["left"])
            )
        else:
            mem.observe(f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f)
    mem.build_chain_links(facts)

    dirty = mode.startswith("dirty")
    partial_clean, ptype = hop1_cue(fact_a, which, cue_mode)
    partial = partial_clean
    if dirty:
        assert dirty_rng is not None
        toks = partial_clean.split()
        ti = len(toks) - 1 if toks else 0
        if mode == "dirty_hard":
            dirty_tok, _ = dirty_token_hard(toks[ti], dirty_rng)
        else:
            dkind = DIRTY_KINDS[dirty_rng.randrange(len(DIRTY_KINDS))]
            dirty_tok, _ = dirty_token(toks[ti], dkind, dirty_rng)
        toks = list(toks)
        toks[ti] = dirty_tok
        partial = " ".join(toks)

    assert hop1_binder is not None and hop2_binder is not None
    used_partial = partial_clean if mode == "clean_bind" else partial
    if ptype == "what_tail":
        bound = bind_full(stack, "what", used_partial)
    else:
        cue_fp = stack.phrase_fp(used_partial.split()) if mode == "clean_bind" else safe_phrase_fp(stack, used_partial)
        pid = torch.tensor([PTYPE2ID[ptype]], device=stack.device)
        kid = torch.tensor([0], device=stack.device)
        bound = hop1_binder(cue_fp.unsqueeze(0), pid, kid).squeeze(0)

    key1, _, _, _, _ = hop1_rich_pick(
        stack, mem, scratch, bound, used_partial if mode != "clean_bind" else partial_clean,
        ptype, facts, cfg, partial_clean, hop1_binder, True, prefer_deep=True,
    )
    ok1 = key1 == fact_a["phrase"] and answer_right(key1) == fact_a["right"]
    bridge1 = answer_right(key1) if key1 else None

    ok2, key2, bridge2 = False, None, None
    if bridge1 is not None:
        bound2 = hop2_binder(safe_phrase_fp(stack, bridge1).unsqueeze(0)).squeeze(0)
        key2, _, _, _, _ = hop2_via_chain_instance(
            stack, mem, scratch, bound2, key1, bridge1, facts, cfg
        )
        ok2 = key2 == fact_b["phrase"] and answer_right(key2) == fact_b["right"]
        bridge2 = answer_right(key2) if key2 else None

    ok3 = False
    if bridge2 is not None:
        bound3 = hop2_binder(safe_phrase_fp(stack, bridge2).unsqueeze(0)).squeeze(0)
        key3, _, _, _, _ = hop3_via_chain(stack, mem, scratch, bound3, key2, bridge2, facts, cfg)
        ok3 = key3 == fact_c["phrase"] and answer_right(key3) == fact_c["right"]

    return {
        "ok1": ok1, "ok2": ok2, "ok3": ok3,
        "joint2": bool(ok1 and ok2), "joint3": bool(ok1 and ok2 and ok3),
        "depth": scratch.depth(),
    }


@torch.no_grad()
def eval_suite_3hop_h2inst(stack, triples, bank, cfg, rng, n_eps, modes, hop1_binder, hop2_binder, episode_k):
    hub_buckets = classify_triples_by_hub(triples, bank)
    stats = {m: {"ok1": 0, "ok2": 0, "ok3": 0, "joint3": 0, "h1_ok": 0, "h2_given_h1": 0, "h2_ok": 0, "h3_given_h2": 0} for m in modes}
    strata = {b: {"n": 0, "ok1": 0, "ok2": 0, "joint3": 0, "h1_ok": 0, "h2_given_h1": 0} for b in ("hub1", "hub2", "hub3p")}
    for i in range(n_eps):
        facts, a, b, c = sample_episode_triple_stratified(triples, bank, episode_k, rng, hub_buckets, i)
        which = "first" if (i % 2 == 0) else "second"
        ep_seed = DENSE_EVAL_SEED * 10007 + i * 19
        bucket = hub_bucket(bridge_hub_size(bank, a["right"]))
        strata[bucket]["n"] += 1
        for m in modes:
            r = run_episode_3hop_h2inst(
                stack, cfg, facts, a, b, c, m, which, hop1_binder, hop2_binder,
                dirty_rng=random.Random(ep_seed), cue_mode="what_tail",
            )
            s = stats[m]
            s["ok1"] += int(r["ok1"]); s["ok2"] += int(r["ok2"]); s["ok3"] += int(r["ok3"])
            s["joint3"] += int(r["joint3"])
            if r["ok1"]:
                s["h1_ok"] += 1; s["h2_given_h1"] += int(r["ok2"])
            if r["ok2"]:
                s["h2_ok"] += 1; s["h3_given_h2"] += int(r["ok3"])
            if m == "clean_bind":
                st = strata[bucket]
                st["ok1"] += int(r["ok1"]); st["ok2"] += int(r["ok2"]); st["joint3"] += int(r["joint3"])
                if r["ok1"]:
                    st["h1_ok"] += 1; st["h2_given_h1"] += int(r["ok2"])
    n = max(n_eps, 1)
    out_modes = {}
    for m in modes:
        s = stats[m]
        out_modes[m] = {
            "hop1": s["ok1"] / n, "hop2": s["ok2"] / n, "hop3": s["ok3"] / n, "joint3": s["joint3"] / n,
            "h2_given_h1": (s["h2_given_h1"] / s["h1_ok"]) if s["h1_ok"] else 0.0,
            "h3_given_h2": (s["h3_given_h2"] / s["h2_ok"]) if s["h2_ok"] else 0.0,
        }
    out_strata = {}
    for b, st in strata.items():
        nn = max(st["n"], 1)
        out_strata[b] = {
            "n": st["n"], "hop1": st["ok1"] / nn, "hop2": st["ok2"] / nn, "joint3": st["joint3"] / nn,
            "h2_given_h1": (st["h2_given_h1"] / st["h1_ok"]) if st["h1_ok"] else 0.0,
        }
    return {"modes": out_modes, "strata": out_strata}


def normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


class TextProjector(nn.Module):
    """Thin bag-char MLP → dim; align to frozen phrase_fp (Stage37 C)."""

    def __init__(self, n_chars: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_chars, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, char_ids: torch.Tensor, lengths: torch.Tensor):
        # char_ids: [B, L]
        x = self.emb(char_ids)
        mask = (torch.arange(char_ids.size(1), device=char_ids.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        pooled = (x * mask.unsqueeze(-1)).sum(1) / lengths.clamp(min=1).float().unsqueeze(1)
        return F.normalize(self.net(pooled), dim=-1)


def text_to_ids(text: str, max_len: int = 48):
    s = normalize_text(text)
    ids = [CHAR2ID[c] for c in s if c in CHAR2ID]
    if not ids:
        ids = [CHAR2ID.get(" ", 0)]
    ids = ids[:max_len]
    return ids, len(ids)


def nearest_slot(fp, mem: SoftPhraseMemory):
    ranked = mem.topk(fp, k=1)
    if not ranked:
        return None, 0.0
    return ranked[0][0], ranked[0][1]


def eval_lexicon_A(phrases):
    """A: normalize → exact key hit in lexicon."""
    lex = {normalize_text(p): p for p in phrases}
    ok = 0
    # hygiene variants
    variants_ok = 0
    variants_n = 0
    for p in phrases:
        if normalize_text(p) in lex:
            ok += 1
        for v in (p.upper(), f"  {p}  ", p.replace(" ", "  ")):
            variants_n += 1
            hit = lex.get(normalize_text(v))
            if hit == p or (hit and normalize_text(hit) == normalize_text(p)):
                variants_ok += 1
    n = max(len(phrases), 1)
    return {
        "exact": ok / n,
        "norm_variants": variants_ok / max(variants_n, 1),
        "n": n,
    }


@torch.no_grad()
def eval_encode_B(stack, phrases, cfg):
    """B: text → frozen compose → NN into memory written by same compose."""
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    for p in phrases:
        words = normalize_text(p).split()
        fp = stack.phrase_fp(words)
        fact = parse_rel(p)
        left_fp = stack.w(fact["left"]) if fact else None
        mem.observe_strict(p, fp, fact=fact, left_fp=left_fp)
    ok = 0
    sims = []
    for p in phrases:
        q = stack.phrase_fp(normalize_text(p).split())
        key, sim = nearest_slot(q, mem)
        sims.append(sim)
        if key == p:
            ok += 1
    n = max(len(phrases), 1)
    return {"recon": ok / n, "mean_sim": sum(sims) / max(len(sims), 1), "n": n}


def train_projector_C(stack, phrases, cfg, device, steps: int):
    proj = TextProjector(len(CHARS), cfg.dim).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=cfg.projector_lr)
    # cache targets
    with torch.no_grad():
        targets = []
        for p in phrases:
            targets.append(stack.phrase_fp(normalize_text(p).split()).detach())
    proj.train()
    rng = random.Random(DENSE_EVAL_SEED + 7)
    for step in range(steps):
        batch = [rng.randrange(len(phrases)) for _ in range(min(32, len(phrases)))]
        ids_list, lens, tgts = [], [], []
        max_l = 1
        for i in batch:
            ids, L = text_to_ids(phrases[i])
            ids_list.append(ids)
            lens.append(L)
            max_l = max(max_l, L)
            tgts.append(targets[i])
        mat = torch.zeros(len(batch), max_l, dtype=torch.long, device=device)
        for bi, ids in enumerate(ids_list):
            mat[bi, : len(ids)] = torch.tensor(ids, device=device)
        lengths = torch.tensor(lens, device=device)
        pred = proj(mat, lengths)
        tgt = torch.stack(tgts, 0).to(device)
        loss = 1.0 - (pred * tgt).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    proj.eval()
    return proj


@torch.no_grad()
def eval_projector_C(stack, proj, phrases, cfg, device):
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    for p in phrases:
        fp = stack.phrase_fp(normalize_text(p).split())
        fact = parse_rel(p)
        left_fp = stack.w(fact["left"]) if fact else None
        mem.observe_strict(p, fp, fact=fact, left_fp=left_fp)
    ok = 0
    sims = []
    for p in phrases:
        ids, L = text_to_ids(p)
        mat = torch.zeros(1, L, dtype=torch.long, device=device)
        mat[0, :L] = torch.tensor(ids, device=device)
        q = proj(mat, torch.tensor([L], device=device)).squeeze(0)
        key, sim = nearest_slot(q, mem)
        sims.append(sim)
        if key == p:
            ok += 1
    n = max(len(phrases), 1)
    return {"recon": ok / n, "mean_sim": sum(sims) / max(len(sims), 1), "n": n}


@torch.no_grad()
def eval_text_qa_B(stack, triples, bank, cfg, hop1, hop2, n_eps=64):
    """
    External text path: write facts via encode-only phrase_fp from strings;
    hop1 cue = what_tail string through CueBinder; hop2 uses left-instance multi-link.
    """
    hub_b = classify_triples_by_hub(triples, bank)
    rng = random.Random(DENSE_EVAL_SEED)
    stats = {"ok1": 0, "ok2": 0, "ok3": 0, "joint3": 0, "h1_ok": 0, "h2_given_h1": 0}
    for i in range(n_eps):
        facts, a, b, c = sample_episode_triple_stratified(triples, bank, PRIMARY_K, rng, hub_b, i)
        # treat phrases as external strings
        ext_facts = []
        for f in facts:
            ph = normalize_text(f["phrase"])
            parsed = parse_rel(ph) or f
            ext_facts.append(parsed)
        a = parse_rel(normalize_text(a["phrase"])) or a
        b = parse_rel(normalize_text(b["phrase"])) or b
        c = parse_rel(normalize_text(c["phrase"])) or c
        r = run_episode_3hop_h2inst(
            stack, cfg, ext_facts, a, b, c, "clean_bind",
            "first" if i % 2 == 0 else "second",
            hop1, hop2, dirty_rng=random.Random(i), cue_mode="what_tail",
        )
        stats["ok1"] += int(r["ok1"])
        stats["ok2"] += int(r["ok2"])
        stats["ok3"] += int(r["ok3"])
        stats["joint3"] += int(r["joint3"])
        if r["ok1"]:
            stats["h1_ok"] += 1
            stats["h2_given_h1"] += int(r["ok2"])
    n = max(n_eps, 1)
    return {
        "hop1": stats["ok1"] / n,
        "hop2": stats["ok2"] / n,
        "hop3": stats["ok3"] / n,
        "joint3": stats["joint3"] / n,
        "h2_given_h1": (stats["h2_given_h1"] / stats["h1_ok"]) if stats["h1_ok"] else 0.0,
    }


@dataclass
class Codebook:
    labels: list[str]
    fps: torch.Tensor  # [N, dim]

    @torch.no_grad()
    def decode(self, fp: torch.Tensor, k: int = 1):
        fp = F.normalize(fp.detach(), dim=-1)
        sims = self.fps @ fp
        if k == 1:
            i = int(sims.argmax())
            return self.labels[i], float(sims[i])
        vals, idx = torch.topk(sims, k=min(k, len(self.labels)))
        return [(self.labels[int(idx[j])], float(vals[j])) for j in range(vals.numel())]


@torch.no_grad()
def build_word_codebook(stack, words) -> Codebook:
    labels = sorted(set(words))
    fps = torch.stack([stack.w(w) for w in labels], 0)
    return Codebook(labels, F.normalize(fps, dim=-1))


@torch.no_grad()
def build_phrase_codebook(stack, phrases) -> Codebook:
    labels = sorted(set(normalize_text(p) for p in phrases))
    fps = torch.stack([stack.phrase_fp(p.split()) for p in labels], 0)
    return Codebook(labels, F.normalize(fps, dim=-1))


@torch.no_grad()
def eval_word_decode(stack, words, codebook: Codebook):
    ok = 0
    for w in words:
        pred, _ = codebook.decode(stack.w(w))
        ok += int(pred == w)
    n = max(len(words), 1)
    return {"exact": ok / n, "n": n}


@torch.no_grad()
def eval_phrase_decode(stack, phrases, codebook: Codebook):
    ok = 0
    token_ok = 0
    token_n = 0
    for p in phrases:
        p = normalize_text(p)
        pred, _ = codebook.decode(stack.phrase_fp(p.split()))
        ok += int(pred == p)
        gw, pw = p.split(), pred.split()
        for i in range(max(len(gw), len(pw))):
            token_n += 1
            if i < len(gw) and i < len(pw) and gw[i] == pw[i]:
                token_ok += 1
    n = max(len(phrases), 1)
    return {"exact": ok / n, "token": token_ok / max(token_n, 1), "n": n}


@torch.no_grad()
def eval_phrase_holdout_decode(stack, phrases, hold_frac=0.25, seed=DENSE_EVAL_SEED):
    """Codebook without held-out phrases — NN cannot invent OOV labels."""
    rng = random.Random(seed)
    labels = sorted(set(normalize_text(p) for p in phrases))
    rng.shuffle(labels)
    n_hold = max(1, int(len(labels) * hold_frac))
    hold = labels[:n_hold]
    train = labels[n_hold:]
    cb = build_phrase_codebook(stack, train)
    ok = 0
    # best possible for pure NN holdout is 0 unless collision with train surface
    for p in hold:
        pred, _ = cb.decode(stack.phrase_fp(p.split()))
        ok += int(pred == p)
    return {
        "exact": ok / max(len(hold), 1),
        "n_hold": len(hold),
        "n_train": len(train),
        "note": "NN codebook cannot emit held-out strings; low exact is expected",
    }


@torch.no_grad()
def eval_answer_as_text(stack, triples, bank, cfg, hop1, hop2, phrase_cb: Codebook, n_eps=64):
    """
    Hop3 → slot key → decode slot.fp to text; also decode gold path.
    Scores: joint3 (slot ids) and answer_text_exact (decoded C text == gold C).
    """
    hub_b = classify_triples_by_hub(triples, bank)
    rng = random.Random(DENSE_EVAL_SEED)
    stats = {
        "ok1": 0, "ok2": 0, "ok3": 0, "joint3": 0,
        "text_c_exact": 0, "text_c_given_ok3": 0, "n_ok3": 0,
        "decode_slot_matches_key": 0, "n_key3": 0,
    }
    for i in range(n_eps):
        facts, a, b, c = sample_episode_triple_stratified(triples, bank, PRIMARY_K, rng, hub_b, i)
        ext_facts = []
        for f in facts:
            ph = normalize_text(f["phrase"])
            ext_facts.append(parse_rel(ph) or f)
        a = parse_rel(normalize_text(a["phrase"])) or a
        b = parse_rel(normalize_text(b["phrase"])) or b
        c = parse_rel(normalize_text(c["phrase"])) or c
        # run episode manually to get key3 + mem
        mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
        scratch = ThoughtScratch(stack.device)
        for f in ext_facts:
            mem.observe_strict(
                f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f, left_fp=stack.w(f["left"])
            )
        mem.build_chain_links(ext_facts)
        which = "first" if i % 2 == 0 else "second"
        partial_clean, ptype = hop1_cue(a, which, "what_tail")
        bound = bind_full(stack, "what", partial_clean)
        key1, _, _, _, _ = hop1_rich_pick(
            stack, mem, scratch, bound, partial_clean, ptype, ext_facts, cfg, partial_clean,
            hop1, True, prefer_deep=True,
        )
        ok1 = key1 == a["phrase"] and answer_right(key1) == a["right"]
        bridge1 = answer_right(key1) if key1 else None
        ok2, key2, bridge2 = False, None, None
        if bridge1 is not None:
            bound2 = hop2(safe_phrase_fp(stack, bridge1).unsqueeze(0)).squeeze(0)
            key2, _, _, _, _ = hop2_via_chain_instance(
                stack, mem, scratch, bound2, key1, bridge1, ext_facts, cfg
            )
            ok2 = key2 == b["phrase"] and answer_right(key2) == b["right"]
            bridge2 = answer_right(key2) if key2 else None
        ok3, key3 = False, None
        if bridge2 is not None:
            bound3 = hop2(safe_phrase_fp(stack, bridge2).unsqueeze(0)).squeeze(0)
            key3, _, _, _, _ = hop3_via_chain(stack, mem, scratch, bound3, key2, bridge2, ext_facts, cfg)
            ok3 = key3 == c["phrase"] and answer_right(key3) == c["right"]
        stats["ok1"] += int(ok1)
        stats["ok2"] += int(ok2)
        stats["ok3"] += int(ok3)
        stats["joint3"] += int(ok1 and ok2 and ok3)
        if key3 and key3 in mem.slots:
            stats["n_key3"] += 1
            dec, _ = phrase_cb.decode(mem.slots[key3].fp)
            stats["decode_slot_matches_key"] += int(normalize_text(dec) == normalize_text(key3))
            gold_c = normalize_text(c["phrase"])
            stats["text_c_exact"] += int(normalize_text(dec) == gold_c)
            if ok3:
                stats["n_ok3"] += 1
                stats["text_c_given_ok3"] += int(normalize_text(dec) == gold_c)
    n = max(n_eps, 1)
    return {
        "hop1": stats["ok1"] / n,
        "hop2": stats["ok2"] / n,
        "hop3": stats["ok3"] / n,
        "joint3": stats["joint3"] / n,
        "answer_text_exact": stats["text_c_exact"] / n,
        "answer_text_given_ok3": (stats["text_c_given_ok3"] / stats["n_ok3"]) if stats["n_ok3"] else 0.0,
        "slot_fp_decodes_to_key": (
            stats["decode_slot_matches_key"] / stats["n_key3"]
        ) if stats["n_key3"] else 0.0,
    }


CORPUS_DIV = CKPT / "fp_language_corpus_diverse.pt"
CORPUS_BEYOND = CKPT / "fp_language_corpus_beyond.pt"
PATH_KINDS = frozenset({"base", "path2", "path3"})
BEYOND_KINDS = frozenset({
    "adj_rel", "adj_verb_rel", "polarity_rel", "verb_rel", "list_rel",
    "morph_paradigm", "frame_grid", "verb_noun_chain", "adj_list_rel",
})
ADJECTIVES = ("big", "red", "old", "new", "good")
POLARITY = ("no", "not")


class TinySeqLM(nn.Module):
    def __init__(self, dim: int, max_len: int = 8, n_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pos = nn.Embedding(max_len, dim)
        self.rnn = nn.GRU(dim, dim, num_layers=n_layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, fps: torch.Tensor, lengths=None) -> torch.Tensor:
        bsz, tmax, _ = fps.shape
        device = fps.device
        pos = self.pos(torch.arange(tmax, device=device).unsqueeze(0).expand(bsz, -1))
        x = F.normalize(fps + pos, dim=-1)
        y, _ = self.rnn(x)
        return F.normalize(self.out(y), dim=-1)

    def pred_from_prefix(self, fps: torch.Tensor) -> torch.Tensor:
        if fps.dim() == 2:
            fps = fps.unsqueeze(0)
        return self.forward(fps)[0, fps.shape[1] - 1]


def read_tiny_external(path: Path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = normalize_text(s)
        words = [w for w in s.split() if w]
        if len(words) >= 2:
            lines.append({"phrase": " ".join(words), "words": words})
    return lines


def split_external(lines, hold_frac, seed=DENSE_EVAL_SEED):
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    n_hold = max(1, int(len(lines) * hold_frac))
    hold_i = set(idx[:n_hold])
    train, hold = [], []
    for i, line in enumerate(lines):
        rec = dict(line)
        rec["split"] = "hold" if i in hold_i else "train"
        (hold if i in hold_i else train).append(rec)
    return train, hold


@torch.no_grad()
def grow_codebook(stack, base_surfaces, base_fps, ext_lines):
    """Union frozen lexicon with external surfaces (encode via stack)."""
    surfaces = list(base_surfaces)
    stoi = {s: i for i, s in enumerate(surfaces)}
    fps = [base_fps[i].detach().cpu() for i in range(len(surfaces))]
    new_words = []
    for line in ext_lines:
        for w in line["words"]:
            if w not in stoi:
                stoi[w] = len(surfaces)
                surfaces.append(w)
                fps.append(F.normalize(stack.w(w).detach().cpu(), dim=-1))
                new_words.append(w)
    mat = torch.stack(fps, 0)
    return surfaces, mat, sorted(set(new_words))


def lines_to_pairs(lines, stoi):
    pairs = []
    for li, line in enumerate(lines):
        ids = [stoi[w] for w in line["words"]]
        for t in range(1, len(ids)):
            pairs.append({
                "line_i": li,
                "prefix_word_ids": ids[:t],
                "target_word_id": ids[t],
                "target_word": line["words"][t],
                "prefix_len": t,
                "split": line["split"],
                "phrase": line["phrase"],
            })
    return pairs


def ft_atom(model, pairs_train, word_fps, cfg, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.next_fp_lr)
    if not pairs_train:
        return model
    model.train()
    for step in range(cfg.ext_ft_steps):
        ex = pairs_train[random.randrange(len(pairs_train))]
        ids = ex["prefix_word_ids"][-model.max_len :]
        # also sample full-line teacher force sometimes
        if random.random() < 0.5 and len(ex["prefix_word_ids"]) >= 1:
            # extend: use random line from same pool via prefix only
            pass
        fps = word_fps[ids]
        pred = model.pred_from_prefix(fps)
        gold = word_fps[ex["target_word_id"]]
        loss = 1.0 - (pred * gold).sum()
        if len(ids) >= 2:
            perm = list(range(len(ids)))
            random.shuffle(perm)
            if perm != list(range(len(ids))):
                ps = model.pred_from_prefix(fps[perm])
                loss = loss + F.relu(cfg.order_margin - ((pred * gold).sum() - (ps * gold).sum()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def ft_atom_lines(model, train_lines, stoi, word_fps, cfg, device):
    """Teacher-force whole lines + order margin (stronger than pairs-only)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.next_fp_lr)
    model.train()
    for step in range(cfg.ext_ft_steps):
        line = train_lines[random.randrange(len(train_lines))]
        ids = [stoi[w] for w in line["words"]][: model.max_len]
        if len(ids) < 2:
            continue
        fps = word_fps[ids]
        pred = model(fps.unsqueeze(0)).squeeze(0)
        loss = 0.0
        n = 0
        for t in range(len(ids) - 1):
            gold = word_fps[ids[t + 1]]
            pc = pred[t]
            loss = loss + (1.0 - (pc * gold).sum())
            n += 1
            if t + 1 >= 2:
                perm = list(range(t + 1))
                random.shuffle(perm)
                if perm != list(range(t + 1)):
                    ps = model.pred_from_prefix(fps[perm])
                    loss = loss + F.relu(cfg.order_margin - ((pc * gold).sum() - (ps * gold).sum()))
        loss = loss / max(n, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def is_path_like(line) -> bool:
    """Short on/to frames only (not every mid line with a preposition)."""
    ws = line.get("words") or []
    if not (3 <= len(ws) <= 5):
        return False
    return ("on" in ws) or ("to" in ws)


def build_rehearsal_pool(mid_train, tiny_lines, tiny_repeat: int = 5):
    """Short path-like mid train + oversampled tiny (53-style) lines."""
    path = [dict(l, kind="path") for l in mid_train if is_path_like(l)]
    tiny = []
    for l in tiny_lines:
        rec = dict(l)
        rec.setdefault("split", "rehearse")
        rec["kind"] = "tiny"
        tiny.append(rec)
    pool = path + tiny * max(int(tiny_repeat), 1)
    return pool, {
        "path_n": len(path),
        "tiny_n": len(tiny),
        "tiny_repeat": int(tiny_repeat),
        "pool_n": len(pool),
    }


def ft_atom_lines_curve(
    model,
    train_lines,
    stoi,
    word_fps,
    codebook,
    ho_pairs,
    cfg,
    device,
    lr: float,
    zs_l2: float = 0.0,
    steps: int | None = None,
    eval_every: int | None = None,
    seed: int = DENSE_EVAL_SEED,
    harden_order: bool = True,
    rehearse_pool=None,
    rehearse_frac: float | None = None,
):
    """FT with HOLD curve; joint restore: L2 floor then max order_drop."""
    steps = int(steps if steps is not None else cfg.mid_ft_steps)
    eval_every = int(eval_every if eval_every is not None else cfg.mid_ft_eval_every)
    l2_hold = bucket(ho_pairs, 2)
    l2_floor = float(cfg.mid_order_l2_floor)
    l2_soft = float(getattr(cfg, "mid_order_l2_soft", cfg.ext_l2_abs_min))
    drop_gate = float(cfg.mid_order_drop_gate)
    n_perm = int(cfg.mid_order_n_perm if harden_order else 1)
    margin = float(cfg.mid_order_margin if harden_order else cfg.order_margin)
    o_w = float(cfg.mid_order_weight if harden_order else 1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    best = {
        "step": 0,
        "acc": -1.0,
        "gap": -1.0,
        "order_drop": -1.0,
        "tier": -1,  # 3=hard+order, 2=soft+order, 1=hard L2 only, 0=else
        "L1": None,
        "L2p": None,
        "L3p": None,
        "order": None,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }

    def _tier(acc: float, drop: float) -> int:
        if acc >= l2_floor - 1e-12 and drop >= drop_gate - 1e-12:
            return 3
        if acc >= l2_soft - 1e-12 and drop >= drop_gate - 1e-12:
            return 2
        if acc >= l2_floor - 1e-12:
            return 1
        return 0

    def _score_key(tier, drop, acc, gap):
        return (tier, drop, acc, gap)

    def _snap(step: int):
        model.eval()
        m = {
            "L1": eval_next(model, bucket(ho_pairs, 1, 1), word_fps, codebook, device),
            "L2p": eval_next(model, l2_hold, word_fps, codebook, device),
            "L3p": eval_next(model, bucket(ho_pairs, 3), word_fps, codebook, device),
        }
        clean = eval_next(model, l2_hold, word_fps, codebook, device, shuffle_prefix=False)
        shuf = eval_next(model, l2_hold, word_fps, codebook, device, shuffle_prefix=True)
        drop = clean["acc"] - shuf["acc"]
        acc = m["L2p"]["acc"]
        gap = m["L2p"]["gap"]
        tier = _tier(acc, drop)
        pt = {
            "step": step,
            "L1_acc": m["L1"]["acc"],
            "L2_acc": acc,
            "L2_gap": gap,
            "L3_acc": m["L3p"]["acc"],
            "order_clean": clean["acc"],
            "order_shuf": shuf["acc"],
            "order_drop": drop,
            "tier": tier,
            "eligible": tier >= 3,
            "soft_ok": tier >= 2,
        }
        curve.append(pt)
        cand = _score_key(tier, drop, acc, gap)
        cur = _score_key(best["tier"], best["order_drop"], best["acc"], best["gap"])
        if cand > cur:
            best["step"] = step
            best["acc"] = acc
            best["gap"] = gap
            best["order_drop"] = drop
            best["tier"] = tier
            best["L1"] = m["L1"]
            best["L2p"] = m["L2p"]
            best["L3p"] = m["L3p"]
            best["order"] = {"clean": clean["acc"], "shuffled": shuf["acc"], "drop": drop}
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        return pt

    random.seed(seed)
    torch.manual_seed(seed)
    r_frac = float(rehearse_frac if rehearse_frac is not None else getattr(cfg, "mid_rehearse_frac", 0.0))
    r_pool = list(rehearse_pool) if rehearse_pool else []
    _snap(0)
    model.train()
    for step in range(1, steps + 1):
        if r_pool and random.random() < r_frac:
            line = r_pool[random.randrange(len(r_pool))]
        else:
            line = train_lines[random.randrange(len(train_lines))]
        ids = [stoi[w] for w in line["words"] if w in stoi][: model.max_len]
        if len(ids) < 2:
            if step % eval_every == 0 or step == steps:
                _snap(step)
            continue
        fps = word_fps[ids]
        pred = model(fps.unsqueeze(0)).squeeze(0)
        loss = 0.0
        n = 0
        for t in range(len(ids) - 1):
            gold = word_fps[ids[t + 1]]
            pc = pred[t]
            loss = loss + (1.0 - (pc * gold).sum())
            n += 1
            if t + 1 >= 2:
                for _ in range(n_perm):
                    perm = list(range(t + 1))
                    random.shuffle(perm)
                    if perm == list(range(t + 1)):
                        continue
                    ps = model.pred_from_prefix(fps[perm])
                    loss = loss + o_w * F.relu(margin - ((pc * gold).sum() - (ps * gold).sum()))
        loss = loss / max(n, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            pt = _snap(step)
            print(
                f"    lr={lr:g} step={step} L2={pt['L2_acc']*100:.1f}%/"
                f"{pt['L2_gap']*100:+.1f}pp order_drop={pt['order_drop']*100:+.1f}pp "
                f"L1={pt['L1_acc']*100:.1f}% "
                f"(best@{best['step']} L2={best['acc']*100:.1f}% drop={best['order_drop']*100:+.1f}pp "
                f"tier={best['tier']})",
                flush=True,
            )

    model.load_state_dict(best["state"])
    model.eval()
    end = curve[-1] if curve else {"L2_acc": 0.0, "L2_gap": 0.0, "order_drop": 0.0, "step": steps}
    return {
        "model": model,
        "curve": curve,
        "best_step": best["step"],
        "best_L2": {"acc": best["acc"], "gap": best["gap"], "n": best["L2p"]["n"] if best["L2p"] else 0},
        "best_L1": best["L1"],
        "best_L3": best["L3p"],
        "best_order": best["order"] or {"clean": 0.0, "shuffled": 0.0, "drop": best["order_drop"]},
        "best_eligible": best["tier"] >= 3,
        "best_tier": best["tier"],
        "end_L2": {"acc": end["L2_acc"], "gap": end.get("L2_gap", 0.0), "step": end["step"]},
        "end_order_drop": end.get("order_drop", 0.0),
        "lr": lr,
        "steps": steps,
        "overshoot_pp": (best["acc"] - end["L2_acc"]) * 100.0,
    }


@torch.no_grad()
def eval_next(model, pairs, word_fps, codebook, device, seed=DENSE_EVAL_SEED, shuffle_prefix=False):
    if not pairs:
        return {"acc": 0.0, "gap": 0.0, "shuffle_tgt": 0.0, "n": 0, "cos": 0.0}
    rng = random.Random(seed + (9 if shuffle_prefix else 0))
    ok = cos_sum = 0.0
    preds, golds = [], []
    for ex in pairs:
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        if shuffle_prefix and len(ids) >= 2:
            ids = ids.copy()
            rng.shuffle(ids)
        pred = model.pred_from_prefix(word_fps[ids])
        dec, _ = codebook.decode(pred)
        ok += int(dec == ex["target_word"])
        cos_sum += float((pred * word_fps[int(ex["target_word_id"])]).sum())
        preds.append(pred)
        golds.append(ex["target_word"])
    sh = list(golds)
    rng.shuffle(sh)
    sh_ok = sum(int(codebook.decode(preds[i])[0] == sh[i]) for i in range(len(preds)))
    n = max(len(pairs), 1)
    acc = ok / n
    return {"acc": acc, "shuffle_tgt": sh_ok / n, "gap": acc - sh_ok / n, "cos": cos_sum / n, "n": n}


def bucket(pairs, lo, hi=None):
    if hi is None:
        return [p for p in pairs if p["prefix_len"] >= lo]
    return [p for p in pairs if lo <= p["prefix_len"] <= hi]


def line_kind(line) -> str:
    """Partition external lines for kinds dig."""
    ws = line.get("words") or []
    n = len(ws)
    has_rel = ("on" in ws) or ("to" in ws)
    if has_rel and 3 <= n <= 5:
        return "path_short"  # 57 rehearsal shape
    if has_rel and n >= 6:
        return "path_long"
    if has_rel:
        return "path_other"
    return "no_rel"  # tail / verb-ish / lists without on|to


def line_subkind(line) -> str:
    """Finer slice inside path-heavy 1k."""
    ws = line.get("words") or []
    if any(w in ("no", "not") for w in ws):
        return "polarity"
    if any(w.endswith("ing") for w in ws):
        return "verb_ing"
    if ("on" in ws) or ("to" in ws):
        return "noun_rel"
    return "other"


def pair_kind(ex, hold_lines) -> str:
    li = ex.get("line_i")
    if li is None or li < 0 or li >= len(hold_lines):
        return "unknown"
    return line_kind(hold_lines[li])


LIVE_SUBKINDS = frozenset({"noun_rel", "polarity"})
PATH_MODS = frozenset({"no", "not", "big", "red", "old", "new", "good"})


def explode_atomic_phrases(phrase: str) -> list[str]:
    """Split multi-rel lines into atomic '{mod?} left on|to right' facts."""
    ws = [w for w in normalize_text(phrase).split() if w]
    atoms = []
    for i, w in enumerate(ws):
        if w in RELS and i > 0 and i + 1 < len(ws):
            toks = [ws[i - 1], w, ws[i + 1]]
            if i >= 2 and ws[i - 2] in PATH_MODS:
                toks = [ws[i - 2]] + toks
            atoms.append(" ".join(toks))
    return atoms


def collect_live_lines(paths) -> list[dict]:
    out = []
    seen = set()
    for path in paths:
        if not Path(path).exists():
            continue
        for line in read_tiny_external(Path(path)):
            sk = line_subkind(line)
            if sk not in LIVE_SUBKINDS:
                continue
            ph = line["phrase"]
            if ph in seen:
                continue
            seen.add(ph)
            line = dict(line)
            line["subkind"] = sk
            line["kind"] = line_kind(line)
            out.append(line)
    return out


def build_external_fact_bank(live_lines, cap: int = 120):
    """Atomic facts from LIVE lines; prefer chain participants.

    cap<=0 means keep all chain-participating atoms (no hard trim).
    """
    phrases = []
    seen = set()
    for line in live_lines:
        for atom in explode_atomic_phrases(line["phrase"]):
            if atom not in seen and parse_rel(atom):
                seen.add(atom)
                phrases.append(atom)
    facts = make_rel_bank(phrases)
    bank = {f["phrase"]: f for f in facts}
    pairs = build_dense_chains(bank)
    # keep facts that appear in >=1 chain pair (or keep all if few pairs)
    if pairs:
        used = set()
        for a, b in pairs:
            used.add(a)
            used.add(b)
        phrases2 = [p for p in phrases if p in used]
        if len(phrases2) >= 20:
            phrases = phrases2
            facts = make_rel_bank(phrases)
            bank = {f["phrase"]: f for f in facts}
            pairs = build_dense_chains(bank)
    # optional cap
    if cap and cap > 0 and len(phrases) > cap:
        keep = []
        keep_set = set()
        for a, b in pairs:
            for p in (a, b):
                if p not in keep_set:
                    keep_set.add(p)
                    keep.append(p)
                if len(keep) >= cap:
                    break
            if len(keep) >= cap:
                break
        for p in phrases:
            if len(keep) >= cap:
                break
            if p not in keep_set:
                keep.append(p)
                keep_set.add(p)
        phrases = keep
        facts = make_rel_bank(phrases)
        bank = {f["phrase"]: f for f in facts}
        pairs = build_dense_chains(bank)
    triples = build_dense_triples(bank)
    return phrases, facts, bank, pairs, triples


def load_fact_bank_txt(path: Path):
    phrases = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = normalize_text(s)
        if parse_rel(s):
            phrases.append(s)
    facts = make_rel_bank(phrases)
    bank = {f["phrase"]: f for f in facts}
    return phrases, facts, bank, build_dense_chains(bank), build_dense_triples(bank)


def is_polarity_phrase(phrase: str) -> bool:
    ws = phrase.split()
    return bool(ws) and ws[0] in ("no", "not")


def slice_bank(bank: dict, polarity: bool):
    if polarity:
        sub = {k: v for k, v in bank.items() if is_polarity_phrase(k)}
    else:
        sub = {k: v for k, v in bank.items() if not is_polarity_phrase(k)}
    pairs = build_dense_chains(sub)
    triples = build_dense_triples(sub)
    return sub, pairs, triples


@torch.no_grad()
def eval_junction_rel_shortlist(model, word_fps, stoi, codebook, pairs, bank, device, max_n=200, seed=DENSE_EVAL_SEED):
    """Atom ranks gold next-fact rel among hub shortlist (diagnostic, like Stage51)."""
    rng = random.Random(seed)
    pool = list(pairs)
    rng.shuffle(pool)
    pool = pool[:max_n]
    ok = 0
    n = 0
    shortlist_sizes = []
    for ap, bp in pool:
        a, b = bank[ap], bank[bp]
        cands = [f for f in bank.values() if f["left"] == a["right"]]
        if len(cands) < 2:
            continue
        # unique by rel; keep gold
        by_rel = {}
        for f in cands:
            by_rel.setdefault(f["rel"], f)
        if b["rel"] not in by_rel:
            by_rel[b["rel"]] = b
        rels = list(by_rel.keys())
        if len(rels) < 2:
            continue
        ids = [stoi[w] for w in a["phrase"].split() if w in stoi]
        if len(ids) < 1:
            continue
        pred = model.pred_from_prefix(word_fps[ids])
        scores = []
        for r in rels:
            if r not in stoi:
                scores.append(-1e9)
                continue
            scores.append(float((pred * word_fps[stoi[r]]).sum()))
        pick = rels[int(max(range(len(rels)), key=lambda i: scores[i]))]
        ok += int(pick == b["rel"])
        n += 1
        shortlist_sizes.append(len(rels))
    return {
        "acc": ok / max(n, 1),
        "n": n,
        "mean_shortlist": sum(shortlist_sizes) / max(len(shortlist_sizes), 1),
        "chance": 1.0 / max(sum(shortlist_sizes) / max(len(shortlist_sizes), 1), 1),
    }


def polarity_stem(phrase: str) -> str | None:
    ws = phrase.split()
    if not ws:
        return None
    if ws[0] in ("no", "not") and len(ws) >= 2:
        return " ".join(ws[1:])
    return None


def polarity_mate_phrase(phrase: str, max_words: int = 4) -> str | None:
    """Pos↔neg surface mate; None if unparseable or too long for PhraseComposer."""
    stem = polarity_stem(phrase)
    if stem is not None:
        return stem if parse_rel(stem) and len(stem.split()) <= max_words else None
    # prefer 'no' mate for positives; keep <= max_words (phrase max_len-1 with END)
    for pref in ("no", "not"):
        mate = f"{pref} {phrase}"
        if len(mate.split()) <= max_words and parse_rel(mate):
            return mate
    return None


def build_true_neg_pairs(bank: dict) -> list[dict]:
    """Pairs sharing (left,rel,right) with one polarity and one positive."""
    out = []
    seen = set()
    for ph, f in bank.items():
        mate = polarity_mate_phrase(ph)
        if mate is None or mate not in bank:
            # allow synthetic mate not already in bank
            if mate is None or parse_rel(mate) is None:
                continue
            mf = parse_rel(mate)
        else:
            mf = bank[mate]
        if is_polarity_phrase(ph) == is_polarity_phrase(mate):
            continue
        if (f["left"], f["rel"], f["right"]) != (mf["left"], mf["rel"], mf["right"]):
            continue
        pos = ph if not is_polarity_phrase(ph) else mate
        neg = mate if not is_polarity_phrase(ph) else ph
        key = (pos, neg)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "pos": pos,
            "neg": neg,
            "left": f["left"],
            "rel": f["rel"],
            "right": f["right"],
            "tail": f"{f['rel']} {f['right']}",
            "in_bank_both": pos in bank and neg in bank,
        })
    return out


@torch.no_grad()
def _write_episode(stack, cfg, phrases: list[str]):
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    scratch = ThoughtScratch(stack.device)
    facts = []
    max_w = int(getattr(stack.phrase, "max_len", 5)) - 1
    for ph in phrases:
        f = parse_rel(ph)
        if f is None or len(ph.split()) > max_w:
            continue
        facts.append(f)
        mem.observe_strict(
            f["phrase"],
            safe_phrase_fp(stack, f["phrase"]),
            fact=f,
            left_fp=stack.w(f["left"]),
        )
    mem.build_chain_links(facts)
    return mem, scratch, facts


@torch.no_grad()
def _hop1_pick(stack, mem, scratch, facts, cfg, hop1_binder, cue: str, ptype: str = "what_tail"):
    if ptype == "what_tail":
        bound = bind_full(stack, "what", cue)
    else:
        cue_fp = stack.phrase_fp(cue.split())
        pid = torch.tensor([PTYPE2ID.get(ptype, 0)], device=stack.device)
        kid = torch.tensor([0], device=stack.device)
        bound = hop1_binder(cue_fp.unsqueeze(0), pid, kid).squeeze(0)
    key, sim, gap, conf, tag = hop1_rich_pick(
        stack, mem, scratch, bound, cue, ptype, facts, cfg, cue, hop1_binder, True, prefer_deep=False,
    )
    # also raw top2 among pos/neg-ish for diagnostics
    top = mem.topk(bound, k=4)
    return {
        "key": key,
        "sim": sim,
        "gap": gap,
        "conf": conf,
        "tag": tag,
        "top": top,
        "bound_margin": float(top[0][1] - top[1][1]) if len(top) >= 2 else (float(top[0][1]) if top else 0.0),
    }


def _classify_pick(key: str | None, pos: str, neg: str) -> str:
    if key is None:
        return "abstain"
    if key == pos:
        return "pos"
    if key == neg:
        return "neg"
    return "other"


@torch.no_grad()
def eval_true_neg_suite(stack, bank, cfg, hop1_binder, rng: random.Random, n_pairs: int = 96, episode_k: int = 7):
    pairs = build_true_neg_pairs(bank)
    # prefer natural co-presence, then synthetic
    natural = [p for p in pairs if p["in_bank_both"]]
    synth = [p for p in pairs if not p["in_bank_both"]]
    rng.shuffle(natural)
    rng.shuffle(synth)
    ordered = natural + synth
    ordered = ordered[:n_pairs]

    all_phrases = list(bank.keys())
    tallies = {
        "conflict_tail": {"pos": 0, "neg": 0, "other": 0, "abstain": 0, "n": 0, "mean_margin": 0.0},
        "pos_only": {"ok": 0, "n": 0},
        "neg_only": {"ok": 0, "n": 0},
        "full_pos_cue": {"ok": 0, "n": 0},
        "full_neg_cue": {"ok": 0, "n": 0},
        "natural_n": len(natural),
        "synth_used": 0,
        "examples": [],
    }
    margin_sum = 0.0

    for i, pair in enumerate(ordered):
        pos, neg, tail = pair["pos"], pair["neg"], pair["tail"]
        if not pair["in_bank_both"]:
            tallies["synth_used"] += 1

        # distractors: other tails, exclude mates
        pool = [p for p in all_phrases if p not in (pos, neg)]
        rng.shuffle(pool)
        fillers = pool[: max(episode_k - 2, 0)]

        # --- conflict: both mates + fillers; cue = shared what_tail ---
        mem, scratch, facts = _write_episode(stack, cfg, [pos, neg] + fillers)
        # ensure synthetic mates are in facts even if not in bank dict keys for fillers path
        if pos not in {f["phrase"] for f in facts}:
            continue
        if neg not in {f["phrase"] for f in facts}:
            # inject parse of synth neg
            nf = parse_rel(neg)
            if nf is None or len(neg.split()) > int(getattr(stack.phrase, "max_len", 5)) - 1:
                continue
            facts.append(nf)
            mem.observe_strict(neg, safe_phrase_fp(stack, neg), fact=nf, left_fp=stack.w(nf["left"]))
            mem.build_chain_links(facts)

        r = _hop1_pick(stack, mem, scratch, facts, cfg, hop1_binder, tail, "what_tail")
        lab = _classify_pick(r["key"], pos, neg)
        tallies["conflict_tail"][lab] += 1
        tallies["conflict_tail"]["n"] += 1
        margin_sum += r["bound_margin"]

        if len(tallies["examples"]) < 8:
            tallies["examples"].append({
                "pos": pos, "neg": neg, "tail": tail, "pick": r["key"], "label": lab,
                "margin": round(r["bound_margin"], 4), "top": r["top"][:3], "natural": pair["in_bank_both"],
            })

        # --- pos-only baseline ---
        mem, scratch, facts = _write_episode(stack, cfg, [pos] + fillers)
        r = _hop1_pick(stack, mem, scratch, facts, cfg, hop1_binder, tail, "what_tail")
        tallies["pos_only"]["n"] += 1
        tallies["pos_only"]["ok"] += int(r["key"] == pos)

        # --- neg-only baseline ---
        mem, scratch, facts = _write_episode(stack, cfg, [neg] + fillers)
        # inject if synth
        if neg not in mem.slots:
            nf = parse_rel(neg)
            if nf is not None and len(neg.split()) <= int(getattr(stack.phrase, "max_len", 5)) - 1:
                facts.append(nf)
                mem.observe_strict(neg, safe_phrase_fp(stack, neg), fact=nf, left_fp=stack.w(nf["left"]))
                mem.build_chain_links(facts)
        r = _hop1_pick(stack, mem, scratch, facts, cfg, hop1_binder, tail, "what_tail")
        tallies["neg_only"]["n"] += 1
        tallies["neg_only"]["ok"] += int(r["key"] == neg)

        # --- full-surface cues (pos phrase / neg phrase as WHAT) ---
        mem, scratch, facts = _write_episode(stack, cfg, [pos, neg] + fillers)
        if neg not in mem.slots:
            nf = parse_rel(neg)
            if nf is not None and len(neg.split()) <= int(getattr(stack.phrase, "max_len", 5)) - 1:
                facts.append(nf)
                mem.observe_strict(neg, safe_phrase_fp(stack, neg), fact=nf, left_fp=stack.w(nf["left"]))
                mem.build_chain_links(facts)
        r = _hop1_pick(stack, mem, scratch, facts, cfg, hop1_binder, pos, "what_tail")
        # full phrase may not match partial_consistent what_tail (expects rel right)
        # use bind_full what on full phrase + hop1_rich with ptype that matches — fallback: score slots directly
        bound = bind_full(stack, "what", pos)
        top = mem.topk(bound, k=2)
        pick = top[0][0] if top else None
        tallies["full_pos_cue"]["n"] += 1
        tallies["full_pos_cue"]["ok"] += int(pick == pos)

        bound = bind_full(stack, "what", neg)
        top = mem.topk(bound, k=2)
        pick = top[0][0] if top else None
        tallies["full_neg_cue"]["n"] += 1
        tallies["full_neg_cue"]["ok"] += int(pick == neg)

    n = max(tallies["conflict_tail"]["n"], 1)
    tallies["conflict_tail"]["mean_margin"] = margin_sum / n
    for k in ("pos_only", "neg_only", "full_pos_cue", "full_neg_cue"):
        nn = max(tallies[k]["n"], 1)
        tallies[k]["acc"] = tallies[k]["ok"] / nn
    ct = tallies["conflict_tail"]
    ct["pos_rate"] = ct["pos"] / n
    ct["neg_rate"] = ct["neg"] / n
    ct["other_rate"] = ct["other"] / n
    ct["abstain_rate"] = ct["abstain"] / n
    # among resolved pos|neg, share of pos (bias)
    resolved = ct["pos"] + ct["neg"]
    ct["pos_among_pn"] = ct["pos"] / max(resolved, 1)
    tallies["n_pairs"] = n
    tallies["n_available"] = len(pairs)
    return tallies


def pair_subkind(ex, lines) -> str:
    li = ex.get("line_i")
    if li is None or li < 0 or li >= len(lines):
        return "unknown"
    return lines[li].get("subkind") or line_subkind(lines[li])


def filter_pairs_subkind(pairs, lines, sk: str):
    return [p for p in pairs if pair_subkind(p, lines) == sk]


def eval_l2_bundle(model, pairs, word_fps, codebook, device):
    l2 = bucket(pairs, 2)
    m = eval_next(model, l2, word_fps, codebook, device)
    clean = eval_next(model, l2, word_fps, codebook, device, shuffle_prefix=False)
    shuf = eval_next(model, l2, word_fps, codebook, device, shuffle_prefix=True)
    return {
        "acc": m["acc"],
        "gap": m["gap"],
        "n": m["n"],
        "order_drop": clean["acc"] - shuf["acc"],
        "cos": m["cos"],
    }


def soft_live(bundle, cfg) -> bool:
    return (
        bundle["n"] >= 20
        and bundle["acc"] >= cfg.ext_l2_abs_min - 1e-12
        and bundle["gap"] >= cfg.ext_l2_gap_min - 1e-12
    )


def ft_verb_ing_curve(
    model,
    train_lines_vi,
    stoi,
    word_fps,
    codebook,
    hold_by_sk,
    cfg,
    device,
    zs_protect,
):
    """Light FT on verb_ing train; pick best without crushing noun/polarity."""
    steps = int(cfg.verb_ing_ft_steps)
    every = int(cfg.verb_ing_eval_every)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.verb_ing_ft_lr)
    curve = []
    best = {
        "step": 0,
        "vi_acc": -1.0,
        "vi_gap": -1.0,
        "protect_ok": True,
        "bundles": None,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    protect_pp = float(cfg.verb_ing_protect_pp)

    def _protect_ok(bundles):
        for sk in ("noun_rel", "polarity"):
            if sk not in zs_protect or zs_protect[sk]["n"] < 20:
                continue
            drop = zs_protect[sk]["acc"] - bundles[sk]["acc"]
            if drop > protect_pp + 1e-12:
                return False
        return True

    def _snap(step: int):
        model.eval()
        bundles = {
            sk: eval_l2_bundle(model, hold_by_sk[sk], word_fps, codebook, device)
            for sk in hold_by_sk
        }
        vi = bundles["verb_ing"]
        pok = _protect_ok(bundles)
        pt = {
            "step": step,
            "verb_ing": vi,
            "noun_rel": bundles.get("noun_rel"),
            "polarity": bundles.get("polarity"),
            "protect_ok": pok,
            "soft": soft_live(vi, cfg),
        }
        curve.append(pt)
        # prefer: soft+protect > residual(gap)+protect > any protect > raw acc
        def key(acc, gap, pok, soft):
            return (int(pok), int(soft), int(gap >= cfg.ext_l2_gap_min - 1e-12), gap, acc)

        cand = key(vi["acc"], vi["gap"], pok, soft_live(vi, cfg))
        if best["bundles"] is None:
            take = True
        else:
            cur = key(
                best["vi_acc"], best["vi_gap"], best["protect_ok"],
                soft_live(best["bundles"]["verb_ing"], cfg),
            )
            take = cand > cur
        if take:
            best["step"] = step
            best["vi_acc"] = vi["acc"]
            best["vi_gap"] = vi["gap"]
            best["protect_ok"] = pok
            best["bundles"] = bundles
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        return pt

    _snap(0)
    model.train()
    for step in range(1, steps + 1):
        line = train_lines_vi[random.randrange(len(train_lines_vi))]
        ids = [stoi[w] for w in line["words"] if w in stoi][: model.max_len]
        if len(ids) < 2:
            continue
        fps = word_fps[ids]
        pred = model(fps.unsqueeze(0)).squeeze(0)
        loss = 0.0
        n = 0
        for t in range(len(ids) - 1):
            gold = word_fps[ids[t + 1]]
            pc = pred[t]
            loss = loss + (1.0 - (pc * gold).sum())
            n += 1
            if t + 1 >= 2:
                perm = list(range(t + 1))
                random.shuffle(perm)
                if perm != list(range(t + 1)):
                    ps = model.pred_from_prefix(fps[perm])
                    loss = loss + F.relu(
                        cfg.order_margin - ((pc * gold).sum() - (ps * gold).sum())
                    )
        if n == 0:
            continue
        loss = loss / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            pt = _snap(step)
            print(
                f"    step {step}: vi L2={pt['verb_ing']['acc']*100:.1f}%/"
                f"+{pt['verb_ing']['gap']*100:.1f}pp protect={pt['protect_ok']}",
                flush=True,
            )

    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    model.eval()
    return best, curve



def split_ing_word(word: str) -> list[str]:
    """walking -> [walk, ing]; leave other words intact."""
    w = word.lower()
    if w in MORPH_LEX and MORPH_LEX[w][1] == "ing":
        stem, _ = MORPH_LEX[w]
        return [stem, "ing"]
    if w.endswith("ing") and w != "ing" and len(w) >= 5:
        stem = w[:-3]
        if len(stem) >= 2 and stem.isalpha():
            return [stem, "ing"]
    return [word]


def remap_ing_line(line: dict) -> dict:
    words = []
    for w in line["words"]:
        words.extend(split_ing_word(w))
    out = dict(line)
    out["words"] = words
    out["phrase"] = " ".join(words)
    out["orig_phrase"] = line.get("phrase", "")
    out["ing_split"] = words != list(line["words"])
    return out



def role_of_target(ex, line_words):
    """Where gold sits in the line: verb_ing / rel / right / other."""
    tw = ex["target_word"]
    ws = line_words
    if tw.endswith("ing") and tw != "ing":
        return "verb_ing"
    if tw in RELS:
        return "rel"
    # object after rel
    for i, w in enumerate(ws):
        if w in RELS and i + 1 < len(ws) and ws[i + 1] == tw:
            return "right"
    if tw in ws and ws.index(tw) == 0:
        return "leftish"
    return "other"


def guess_stem_ing(word: str) -> tuple[str | None, str]:
    """Map -ing surface to known morph if possible."""
    w = word.lower()
    if w in MORPH_LEX and MORPH_LEX[w][1] == "ing":
        return MORPH_LEX[w][0], "morph_lex"
    # try invert surface() over MORPH_STEMS
    for stem in list(dict.fromkeys(MORPH_STEMS + PLAIN_EXTRA)):
        try:
            if surface(stem, "ing") == w:
                return stem, "surface_invert"
        except Exception:
            continue
    if w.endswith("ing") and w != "ing" and len(w) >= 5:
        return w[:-3], "naive_strip"
    return None, "none"



def encode_surface(stack, surf: str):
    """Word → stack.w; 'on mat' → phrase_fp."""
    parts = [p for p in surf.split() if p]
    if not parts:
        parts = ["x"]
    if len(parts) == 1:
        return F.normalize(stack.w(parts[0]).detach(), dim=-1)
    return F.normalize(safe_phrase_fp(stack, " ".join(parts)), dim=-1)


def grow_codebook_flex(stack, base_surfaces, base_fps, ext_lines):
    surfaces = list(base_surfaces)
    stoi = {s: i for i, s in enumerate(surfaces)}
    fps = [base_fps[i].detach().cpu() for i in range(len(surfaces))]
    new_surfs = []
    for line in ext_lines:
        for w in line["words"]:
            if w not in stoi:
                stoi[w] = len(surfaces)
                surfaces.append(w)
                fps.append(encode_surface(stack, w).detach().cpu())
                new_surfs.append(w)
    return surfaces, torch.stack(fps, 0), sorted(set(new_surfs))


def merge_rel_tail(words: list[str]) -> list[str]:
    """Merge on|to + next token into one atom."""
    out = []
    i = 0
    while i < len(words):
        if words[i] in RELS and i + 1 < len(words):
            out.append(f"{words[i]} {words[i + 1]}")
            i += 2
        else:
            out.append(words[i])
            i += 1
    return out


def remap_tail_line(line: dict) -> dict:
    words = merge_rel_tail(list(line["words"]))
    out = dict(line)
    out["words"] = words
    out["phrase"] = " ".join(words)
    out["orig_phrase"] = line.get("phrase", "")
    out["tail_merged"] = words != list(line["words"])
    return out



class SideHintFusion(nn.Module):
    """Combine TinySeq pred with phrase_fp hint (same dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, pred, hint):
        x = torch.cat([pred, hint], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class RightHead(nn.Module):
    """ADD-on for object-after-rel: condition TinySeq h on left+rel instance."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, h: torch.Tensor, left_fp: torch.Tensor, rel_fp: torch.Tensor):
        if h.dim() == 1:
            h = h.unsqueeze(0)
            left_fp = left_fp.unsqueeze(0)
            rel_fp = rel_fp.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        x = torch.cat([h, left_fp, rel_fp], dim=-1)
        q = F.normalize(h + self.res_scale * self.net(x), dim=-1)
        return q.squeeze(0) if squeeze else q


def collect_object_pairs(pairs, lines):
    """Pairs where prefix ends with on|to (object prediction slot)."""
    out = []
    for ex in pairs:
        line = lines[ex["line_i"]]
        pl = ex["prefix_len"]
        ws = line["words"]
        if pl < 1 or pl > len(ws):
            continue
        if ws[pl - 1] in RELS:
            out.append(ex)
    return out


def build_rights_index(train_lines):
    """rel -> set of right objects seen in train."""
    by_rel = {r: set() for r in RELS}
    all_rights = set()
    for ln in train_lines:
        ws = ln["words"]
        for i, w in enumerate(ws):
            if w in RELS and i + 1 < len(ws):
                by_rel[w].add(ws[i + 1])
                all_rights.add(ws[i + 1])
    return by_rel, all_rights


def ft_ing_rehearse(
    model,
    train_vi,
    train_live,
    stoi,
    word_fps,
    codebook,
    hold_by_sk,
    cfg,
    device,
    zs_protect,
):
    """FT on ing-split verb_ing with noun/polarity rehearsal."""
    steps = int(cfg.ing_rehearse_steps)
    every = int(cfg.ing_eval_every)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ing_ft_lr)
    curve = []
    best = {
        "step": 0,
        "vi_acc": -1.0,
        "vi_gap": -1.0,
        "protect_ok": True,
        "bundles": None,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    protect_pp = float(cfg.verb_ing_protect_pp)
    frac = float(cfg.ing_rehearse_frac)

    def _protect_ok(bundles):
        for sk in ("noun_rel", "polarity"):
            if sk not in zs_protect or zs_protect[sk]["n"] < 20:
                continue
            if zs_protect[sk]["acc"] - bundles[sk]["acc"] > protect_pp + 1e-12:
                return False
        return True

    def _key(acc, gap, pok, soft):
        return (int(pok), int(soft), int(gap >= cfg.ext_l2_gap_min - 1e-12), gap, acc)

    def _snap(step: int):
        model.eval()
        bundles = {
            sk: eval_l2_bundle(model, hold_by_sk[sk], word_fps, codebook, device)
            for sk in hold_by_sk
        }
        vi = bundles["verb_ing"]
        pok = _protect_ok(bundles)
        soft = soft_live(vi, cfg)
        curve.append({"step": step, "vi": vi, "protect_ok": pok, "soft": soft,
                       "noun_rel": bundles["noun_rel"], "polarity": bundles["polarity"]})
        cand = _key(vi["acc"], vi["gap"], pok, soft)
        if best["bundles"] is None:
            take = True
        else:
            take = cand > _key(
                best["vi_acc"], best["vi_gap"], best["protect_ok"],
                soft_live(best["bundles"]["verb_ing"], cfg),
            )
        if take:
            best.update(step=step, vi_acc=vi["acc"], vi_gap=vi["gap"], protect_ok=pok, bundles=bundles)
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        return curve[-1]

    _snap(0)
    model.train()
    for step in range(1, steps + 1):
        if train_live and random.random() < frac:
            line = train_live[random.randrange(len(train_live))]
        else:
            line = train_vi[random.randrange(len(train_vi))]
        ids = [stoi[w] for w in line["words"] if w in stoi][: model.max_len]
        if len(ids) < 2:
            continue
        fps = word_fps[ids]
        pred = model(fps.unsqueeze(0)).squeeze(0)
        loss = 0.0
        n = 0
        for t in range(len(ids) - 1):
            gold = word_fps[ids[t + 1]]
            pc = pred[t]
            loss = loss + (1.0 - (pc * gold).sum())
            n += 1
            if t + 1 >= 2:
                perm = list(range(t + 1))
                random.shuffle(perm)
                if perm != list(range(t + 1)):
                    ps = model.pred_from_prefix(fps[perm])
                    loss = loss + F.relu(cfg.order_margin - ((pc * gold).sum() - (ps * gold).sum()))
        if n == 0:
            continue
        opt.zero_grad()
        (loss / n).backward()
        opt.step()
        if step % every == 0 or step == steps:
            pt = _snap(step)
            print(
                f"    [70] step {step}: vi={pt['vi']['acc']*100:.1f}%/"
                f"+{pt['vi']['gap']*100:.1f}pp protect={pt['protect_ok']} soft={pt['soft']}",
                flush=True,
            )
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    model.eval()
    return best, curve


@torch.no_grad()
def eval_object_slot(model, pairs, lines, word_fps, codebook, device, fusion=None, stack=None):
    if not pairs:
        return {"acc": 0.0, "gap": 0.0, "n": 0, "cos": 0.0}
    ok = cos_sum = 0.0
    preds, golds = [], []
    for ex in pairs:
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = model.pred_from_prefix(word_fps[ids])
        if fusion is not None and stack is not None:
            line = lines[ex["line_i"]]
            pref = " ".join(line["words"][: ex["prefix_len"]])
            hint = encode_surface(stack, pref)
            pred = fusion(pred, hint)
        dec, _ = codebook.decode(pred)
        ok += int(dec == ex["target_word"])
        cos_sum += float((pred * word_fps[int(ex["target_word_id"])]).sum())
        preds.append(pred)
        golds.append(ex["target_word"])
    sh = list(golds)
    random.Random(DENSE_EVAL_SEED).shuffle(sh)
    sh_ok = sum(int(codebook.decode(preds[i])[0] == sh[i]) for i in range(len(preds)))
    n = max(len(pairs), 1)
    acc = ok / n
    return {"acc": acc, "shuffle_tgt": sh_ok / n, "gap": acc - sh_ok / n, "cos": cos_sum / n, "n": n}


def run_dig70(stack, parent, cfg, device, train_raw, hold_raw):
    print("\n=== DIG 70: ing-atom + rehearsal protect ===", flush=True)
    base_surfaces = list(parent["surfaces"])
    base_fps = parent["word_fps"]
    if not torch.is_tensor(base_fps):
        base_fps = torch.stack(base_fps)

    train_lines = [remap_ing_line(ln) for ln in train_raw]
    hold_lines = [remap_ing_line(ln) for ln in hold_raw]
    surf, fps_cpu, new_s = grow_codebook(stack, base_surfaces, base_fps, train_lines + hold_lines)
    # ensure ing present
    if "ing" not in {s for s in surf}:
        surf, fps_cpu, new_s2 = grow_codebook(stack, surf, fps_cpu, [{"words": ["ing"], "phrase": "ing"}])
        new_s = list(new_s) + list(new_s2)
    word_fps = F.normalize(fps_cpu.to(device), dim=-1)
    stoi = {s: i for i, s in enumerate(surf)}
    codebook = Codebook(surf, word_fps)

    model = TinySeqLM(cfg.dim).to(device)
    model.load_state_dict(parent["tiny_seq"])
    model.eval()

    ho = lines_to_pairs(hold_lines, stoi)
    hold_by_sk = {sk: filter_pairs_subkind(ho, hold_lines, sk) for sk in ("verb_ing", "noun_rel", "polarity")}
    train_vi = [ln for ln in train_lines if ln["subkind"] == "verb_ing"]
    train_live = [ln for ln in train_lines if ln["subkind"] in ("noun_rel", "polarity")]

    zs = {sk: eval_l2_bundle(model, hold_by_sk[sk], word_fps, codebook, device) for sk in hold_by_sk}
    print(f"  ZS vi={zs['verb_ing']['acc']*100:.1f}% live_train={len(train_live)} vi_train={len(train_vi)}", flush=True)

    best, curve = ft_ing_rehearse(
        model, train_vi, train_live, stoi, word_fps, codebook, hold_by_sk, cfg, device, zs,
    )
    final = best["bundles"]
    vi = final["verb_ing"]
    protect_ok = best["protect_ok"]
    live = soft_live(vi, cfg)
    if live and protect_ok:
        verdict = "PASS"
    elif vi["gap"] >= cfg.ext_l2_gap_min and vi["acc"] < cfg.ext_l2_abs_min and protect_ok:
        verdict = "RESIDUAL"
        if vi["acc"] >= zs["verb_ing"]["acc"] + 0.05:
            verdict = "PARTIAL lift"
    elif not protect_ok:
        verdict = "FAIL protect"
    else:
        verdict = "FAIL"

    drops = {
        sk: zs[sk]["acc"] - final[sk]["acc"]
        for sk in ("noun_rel", "polarity") if zs[sk]["n"] >= 20
    }
    lines = [
        "SOTE Stage 70 report (ing-atom + rehearsal protect)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"steps<={cfg.ing_rehearse_steps} rehearse_frac={cfg.ing_rehearse_frac} protect<={cfg.verb_ing_protect_pp*100:.0f}pp",
        f"ZS vi L2={zs['verb_ing']['acc']*100:.1f}%/+{zs['verb_ing']['gap']*100:.1f}pp n={zs['verb_ing']['n']}",
        f"best step={best['step']} vi L2={vi['acc']*100:.1f}%/+{vi['gap']*100:.1f}pp protect={protect_ok}",
        f"noun_rel {final['noun_rel']['acc']*100:.1f}% (dZS={drops.get('noun_rel',0)*100:+.1f}pp)  "
        f"polarity {final['polarity']['acc']*100:.1f}% (dZS={drops.get('polarity',0)*100:+.1f}pp)",
        f"=== Verdict: {verdict} ===",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    OUT_70.write_text(report, encoding="utf-8")
    return {
        "verdict": verdict, "zs": zs, "final": final, "best_step": best["step"],
        "protect_ok": protect_ok, "curve": curve, "drops": drops,
    }


def run_dig71(stack, parent, cfg, device, train_raw, hold_raw):
    print("\n=== DIG 71: side hint phrase_fp(prefix) on object slot ===", flush=True)
    base_surfaces = list(parent["surfaces"])
    base_fps = parent["word_fps"]
    if not torch.is_tensor(base_fps):
        base_fps = torch.stack(base_fps)
    surf, fps_cpu, _ = grow_codebook(stack, base_surfaces, base_fps, train_raw + hold_raw)
    word_fps = F.normalize(fps_cpu.to(device), dim=-1)
    stoi = {s: i for i, s in enumerate(surf)}
    codebook = Codebook(surf, word_fps)

    model = TinySeqLM(cfg.dim).to(device)
    model.load_state_dict(parent["tiny_seq"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    fusion = SideHintFusion(cfg.dim).to(device)
    tr_pairs = lines_to_pairs(train_raw, stoi)
    ho_pairs = lines_to_pairs(hold_raw, stoi)
    # verb_ing object slots
    tr_vi = [p for p in tr_pairs if pair_subkind(p, train_raw) == "verb_ing"]
    ho_vi = [p for p in ho_pairs if pair_subkind(p, hold_raw) == "verb_ing"]
    tr_obj = collect_object_pairs(tr_vi, train_raw)
    ho_obj = collect_object_pairs(ho_vi, hold_raw)
    print(f"  object pairs train={len(tr_obj)} hold={len(ho_obj)}", flush=True)

    base = eval_object_slot(model, ho_obj, hold_raw, word_fps, codebook, device)
    print(f"  baseline TinySeq object-slot: {base['acc']*100:.1f}%/+{base['gap']*100:.1f}pp n={base['n']}", flush=True)

    opt = torch.optim.Adam(fusion.parameters(), lr=cfg.side_hint_lr)
    best = {"acc": -1.0, "gap": -1.0, "step": 0, "state": {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}, "m": base}
    fusion.train()
    for step in range(1, cfg.side_hint_steps + 1):
        if not tr_obj:
            break
        ex = tr_obj[random.randrange(len(tr_obj))]
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        with torch.no_grad():
            pred = model.pred_from_prefix(word_fps[ids])
        line = train_raw[ex["line_i"]]
        hint = encode_surface(stack, " ".join(line["words"][: ex["prefix_len"]]))
        out = fusion(pred, hint)
        gold = word_fps[ex["target_word_id"]]
        loss = 1.0 - (out * gold).sum()
        # mild push vs random other right
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == cfg.side_hint_steps:
            fusion.eval()
            m = eval_object_slot(model, ho_obj, hold_raw, word_fps, codebook, device, fusion, stack)
            print(f"    [71] step {step}: obj={m['acc']*100:.1f}%/+{m['gap']*100:.1f}pp", flush=True)
            if (m["acc"], m["gap"]) > (best["acc"], best["gap"]):
                best.update(acc=m["acc"], gap=m["gap"], step=step, m=m)
                best["state"] = {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}
            fusion.train()

    fusion.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    fusion.eval()
    final = eval_object_slot(model, ho_obj, hold_raw, word_fps, codebook, device, fusion, stack)
    # protect: overall vi L2 should not be required; this dig is object-slot only
    live = final["n"] >= 20 and final["acc"] >= 0.25 and final["gap"] >= 0.10
    lift = final["acc"] - base["acc"]
    if live:
        verdict = "PASS"
    elif lift >= 0.05 and final["gap"] >= 0.10:
        verdict = "PARTIAL lift"
    elif final["gap"] >= 0.10 and final["acc"] < 0.25:
        verdict = "RESIDUAL"
    else:
        verdict = "FAIL"

    lines = [
        "SOTE Stage 71 report (side hint phrase_fp(prefix) on object slot)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        "TinySeq frozen; train SideHintFusion only. Hint=phrase_fp(prefix) includes on|to, not gold object.",
        f"hold object n={ho_obj and len(ho_obj)}",
        f"baseline TinySeq: {base['acc']*100:.1f}%/+{base['gap']*100:.1f}pp n={base['n']}",
        f"best step={best['step']} hint: {final['acc']*100:.1f}%/+{final['gap']*100:.1f}pp  dbase={lift*100:+.1f}pp",
        f"=== Verdict: {verdict} ===",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    OUT_71.write_text(report, encoding="utf-8")
    return {"verdict": verdict, "base": base, "final": final, "lift": lift, "best_step": best["step"]}


def run_dig72(stack, parent, cfg, device, train_raw, hold_raw):
    print("\n=== DIG 72: object shortlist after *ing on|to ===", flush=True)
    base_surfaces = list(parent["surfaces"])
    base_fps = parent["word_fps"]
    if not torch.is_tensor(base_fps):
        base_fps = torch.stack(base_fps)
    surf, fps_cpu, _ = grow_codebook(stack, base_surfaces, base_fps, train_raw + hold_raw)
    word_fps = F.normalize(fps_cpu.to(device), dim=-1)
    stoi = {s: i for i, s in enumerate(surf)}
    codebook = Codebook(surf, word_fps)

    model = TinySeqLM(cfg.dim).to(device)
    model.load_state_dict(parent["tiny_seq"])
    model.eval()

    train_vi_lines = [ln for ln in train_raw if ln["subkind"] == "verb_ing"]
    by_rel, all_rights = build_rights_index(train_vi_lines)
    ho_pairs = lines_to_pairs(hold_raw, stoi)
    ho_vi = [p for p in ho_pairs if pair_subkind(p, hold_raw) == "verb_ing"]
    ho_obj = collect_object_pairs(ho_vi, hold_raw)

    full = eval_object_slot(model, ho_obj, hold_raw, word_fps, codebook, device)

    ok = 0
    rand_ok = 0
    n = 0
    sizes = []
    rng = random.Random(DENSE_EVAL_SEED)
    with torch.no_grad():
        for ex in ho_obj:
            line = hold_raw[ex["line_i"]]
            pl = ex["prefix_len"]
            rel = line["words"][pl - 1]
            gold = ex["target_word"]
            cands = sorted(by_rel.get(rel, set()) | {gold})
            if gold not in stoi:
                continue
            if len(cands) < cfg.shortlist_min_cands_obj:
                cands = sorted(all_rights | {gold})
            cands = [c for c in cands if c in stoi]
            if len(cands) < 2 or gold not in cands:
                continue
            ids = list(ex["prefix_word_ids"][-model.max_len :])
            pred = model.pred_from_prefix(word_fps[ids])
            scores = [(c, float((pred * word_fps[stoi[c]]).sum())) for c in cands]
            pick = max(scores, key=lambda x: x[1])[0]
            ok += int(pick == gold)
            rand_ok += int(rng.choice(cands) == gold)
            n += 1
            sizes.append(len(cands))

    acc = ok / max(n, 1)
    chance = rand_ok / max(n, 1)
    gap = acc - chance
    mean_s = sum(sizes) / max(len(sizes), 1)
    live = n >= 20 and acc >= 0.35 and gap >= 0.12  # soft: above shortlist chance band
    if live:
        verdict = "PASS"
    elif n >= 20 and gap >= 0.12 and acc < 0.35:
        verdict = "PARTIAL"
    elif n >= 20 and gap >= 0.05:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    lines = [
        "SOTE Stage 72 report (object shortlist after *ing on|to)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        "Candidates = train verb_ing rights for same rel (fallback: all vi rights).",
        f"full-vocab object-slot: {full['acc']*100:.1f}%/+{full['gap']*100:.1f}pp n={full['n']}",
        f"shortlist: acc={acc*100:.1f}%  chance~{chance*100:.1f}%  gap={gap*100:+.1f}pp  "
        f"n={n} mean|cands|={mean_s:.1f}",
        f"=== Verdict: {verdict} ===",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    OUT_72.write_text(report, encoding="utf-8")
    return {
        "verdict": verdict, "full": full, "acc": acc, "chance": chance, "gap": gap,
        "n": n, "mean_cands": mean_s,
    }




def parse_verb_triple(line: dict):
    """Extract (verb_ing, rel, right) if present."""
    ws = line["words"]
    for i, w in enumerate(ws):
        if w.endswith("ing") and w != "ing" and i + 2 < len(ws) and ws[i + 1] in RELS:
            return (w, ws[i + 1], ws[i + 2])
    return None


def collect_object_pairs(pairs, lines):
    out = []
    for ex in pairs:
        line = lines[ex["line_i"]]
        pl = ex["prefix_len"]
        ws = line["words"]
        if pl >= 1 and pl <= len(ws) and ws[pl - 1] in RELS:
            out.append(ex)
    return out


@torch.no_grad()
def eval_object_slot(model, pairs, lines, word_fps, codebook, device):
    if not pairs:
        return {"acc": 0.0, "gap": 0.0, "n": 0, "cos": 0.0, "shuffle_tgt": 0.0}
    ok = cos_sum = 0.0
    preds, golds = [], []
    for ex in pairs:
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = model.pred_from_prefix(word_fps[ids])
        dec, _ = codebook.decode(pred)
        ok += int(dec == ex["target_word"])
        cos_sum += float((pred * word_fps[int(ex["target_word_id"])]).sum())
        preds.append(pred)
        golds.append(ex["target_word"])
    sh = list(golds)
    random.Random(DENSE_EVAL_SEED).shuffle(sh)
    sh_ok = sum(int(codebook.decode(preds[i])[0] == sh[i]) for i in range(len(preds)))
    n = max(len(pairs), 1)
    acc = ok / n
    return {"acc": acc, "shuffle_tgt": sh_ok / n, "gap": acc - sh_ok / n, "cos": cos_sum / n, "n": n}



def build_rights_by_rel(lines):
    by = {r: set() for r in RELS}
    all_r = set()
    for ln in lines:
        ws = ln["words"]
        for i, w in enumerate(ws):
            if w in RELS and i + 1 < len(ws):
                by[w].add(ws[i + 1])
                all_r.add(ws[i + 1])
    return by, all_r


@torch.no_grad()
def eval_shortlist_object(model, pairs, lines, word_fps, stoi, by_rel, all_rights, device, seed=DENSE_EVAL_SEED):
    if not pairs:
        return {"acc": 0.0, "chance": 0.0, "gap": 0.0, "n": 0, "mean_cands": 0.0}
    rng = random.Random(seed)
    ok = rand_ok = 0
    sizes = []
    for ex in pairs:
        line = lines[ex["line_i"]]
        pl = ex["prefix_len"]
        rel = line["words"][pl - 1]
        gold = ex["target_word"]
        cands = sorted((by_rel.get(rel) or set()) | {gold})
        cands = [c for c in cands if c in stoi]
        if len(cands) < 2:
            cands = sorted(all_rights | {gold})
            cands = [c for c in cands if c in stoi]
        if gold not in cands or len(cands) < 2:
            continue
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = model.pred_from_prefix(word_fps[ids])
        scores = [float((pred * word_fps[stoi[c]]).sum()) for c in cands]
        pick = cands[int(max(range(len(cands)), key=lambda i: scores[i]))]
        ok += int(pick == gold)
        rand_ok += int(rng.choice(cands) == gold)
        sizes.append(len(cands))
    n = max(len(sizes), 1)
    # use counted n
    n = max(ok + (len(sizes) - ok), 1) if sizes else 1
    n = len(sizes) if sizes else 1
    acc = ok / n
    chance = rand_ok / n
    return {
        "acc": acc,
        "chance": chance,
        "gap": acc - chance,
        "n": n,
        "mean_cands": sum(sizes) / max(len(sizes), 1),
    }


def shortlist_infonce_loss(pred, gold_id, cand_ids, word_fps):
    """pred [D], cand_ids list[int] including gold."""
    fps = word_fps[cand_ids]  # [C,D]
    logits = fps @ pred  # [C]
    # gold index in cand list
    gi = cand_ids.index(gold_id) if not isinstance(cand_ids, torch.Tensor) else int((cand_ids == gold_id).nonzero()[0])
    return F.cross_entropy(logits.unsqueeze(0), torch.tensor([gi], device=pred.device))


@torch.no_grad()
def eval_hop_object_bank(stack, phrases, cfg, hop1_binder, n_eps=64):
    """Write dense phrases; cue what_tail; measure retrieve exact phrase."""
    facts = make_rel_bank(phrases)
    bank = {f["phrase"]: f for f in facts}
    if len(bank) < 5:
        return {"joint": 0.0, "n": 0}
    # sample episodes: write k facts, cue one
    rng = random.Random(DENSE_EVAL_SEED + 11)
    keys = list(bank.keys())
    ok = 0
    n = 0
    ek = min(7, len(keys))
    for _ in range(n_eps):
        ep = rng.sample(keys, ek)
        target = rng.choice(ep)
        fact = bank[target]
        mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
        scratch = ThoughtScratch(stack.device)
        for ph in ep:
            f = bank[ph]
            mem.observe_strict(
                ph, stack.phrase_fp(ph.split()), fact=f, left_fp=stack.w(f["left"]),
            )
        mem.build_chain_links([bank[ph] for ph in ep])
        cue = f"{fact['rel']} {fact['right']}"
        bound = bind_full(stack, "what", cue)
        key, _, _, _, _ = hop1_rich_pick(
            stack, mem, scratch, bound, cue, "what_tail",
            [bank[ph] for ph in ep], cfg, cue, hop1_binder, True, prefer_deep=False,
        )
        ok += int(key == target)
        n += 1
    return {"joint": ok / max(n, 1), "n": n}



ROLE_ORDER = ("verb_ing", "rel", "right", "leftish", "other")


def target_role(ex, line_words) -> str:
    tw = ex["target_word"]
    ws = line_words
    if tw.endswith("ing") and tw != "ing":
        return "verb_ing"
    if tw in RELS:
        return "rel"
    pl = int(ex["prefix_len"])
    if pl >= 1 and pl <= len(ws) and ws[pl - 1] in RELS:
        return "right"
    if tw in ws and ws.index(tw) == 0:
        return "leftish"
    return "other"


@torch.no_grad()
def eval_hitk_by_role(model, pairs, lines, word_fps, surfaces, stoi, device, k: int = 5, min_prefix: int = 2):
    """Codebook-only hit@1/@5 + role tables. Chance@k = k/|V|."""
    from collections import defaultdict

    buckets = defaultdict(lambda: {"n": 0, "h1": 0, "h5": 0, "rank_sum": 0.0})
    V = len(surfaces)
    kk = min(k, V)
    chance5 = kk / max(V, 1)

    for ex in pairs:
        if int(ex["prefix_len"]) < min_prefix:
            continue
        line = lines[ex["line_i"]]
        role = target_role(ex, line["words"])
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = model.pred_from_prefix(word_fps[ids])
        sims = word_fps @ pred
        top_idx = torch.topk(sims, k=kk).indices.tolist()
        top_labs = [surfaces[int(i)] for i in top_idx]
        order = torch.argsort(sims, descending=True)
        rank = int((order == stoi[gold]).nonzero()[0]) + 1
        b = buckets[role]
        b["n"] += 1
        b["h1"] += int(top_labs[0] == gold)
        b["h5"] += int(gold in top_labs)
        b["rank_sum"] += rank
        b = buckets["ALL"]
        b["n"] += 1
        b["h1"] += int(top_labs[0] == gold)
        b["h5"] += int(gold in top_labs)
        b["rank_sum"] += rank

    out = {"chance_at_k": chance5, "k": kk, "V": V, "roles": {}}
    for name, b in buckets.items():
        n = max(b["n"], 1)
        out["roles"][name] = {
            "n": b["n"],
            "hit1": b["h1"] / n if b["n"] else 0.0,
            "hit5": b["h5"] / n if b["n"] else 0.0,
            "mean_rank": b["rank_sum"] / n if b["n"] else 0.0,
        }
    return out


def fmt_role_table(block, title: str) -> list[str]:
    lines = [
        f"=== {title} ===",
        f"  V={block['V']}  k={block['k']}  chance@k={block['chance_at_k']*100:.2f}%",
        f"  {'role':10s} {'n':>5} {'hit@1':>8} {'hit@5':>8} {'mean_rk':>8}",
    ]
    order = ["ALL"] + [r for r in ROLE_ORDER if r in block["roles"]]
    for extra in sorted(block["roles"]):
        if extra not in order:
            order.append(extra)
    for name in order:
        if name not in block["roles"]:
            continue
        r = block["roles"][name]
        if r["n"] == 0:
            continue
        lines.append(
            f"  {name:10s} {r['n']:5d} {r['hit1']*100:7.1f}% {r['hit5']*100:7.1f}% {r['mean_rank']:8.1f}"
        )
    return lines



def eval_hit1_by_role(model, pairs, lines, word_fps, surfaces, stoi, min_prefix: int = 2):
    """Exact@1 only (no top-5). Role tables mandatory."""
    from collections import defaultdict

    buckets = defaultdict(lambda: {"n": 0, "h1": 0})
    for ex in pairs:
        if int(ex["prefix_len"]) < min_prefix:
            continue
        line = lines[ex["line_i"]]
        role = _role(ex, line["words"])
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = model.pred_from_prefix(word_fps[ids])
        dec = surfaces[int((word_fps @ pred).argmax())]
        for name in (role, "ALL"):
            buckets[name]["n"] += 1
            buckets[name]["h1"] += int(dec == gold)
    out = {}
    for name, b in buckets.items():
        out[name] = {"n": b["n"], "hit1": b["h1"] / max(b["n"], 1)}
    return out


def _role(ex, ws):
    tw = ex["target_word"]
    if tw.endswith("ing") and tw != "ing":
        return "verb_ing"
    if tw in RELS:
        return "rel"
    pl = int(ex["prefix_len"])
    if pl >= 1 and pl <= len(ws) and ws[pl - 1] in RELS:
        return "right"
    if tw in ws and ws.index(tw) == 0:
        return "leftish"
    return "other"


def fmt_hit1(roles: dict, title: str) -> list[str]:
    lines = [f"=== {title} ===", f"  {'role':10s} {'n':>5} {'hit@1':>8}"]
    order = ["ALL", "rel", "right", "verb_ing", "leftish", "other"]
    for name in order:
        if name not in roles or roles[name]["n"] == 0:
            continue
        r = roles[name]
        lines.append(f"  {name:10s} {r['n']:5d} {r['hit1']*100:7.1f}%")
    for name in sorted(roles):
        if name in order:
            continue
        r = roles[name]
        if r["n"]:
            lines.append(f"  {name:10s} {r['n']:5d} {r['hit1']*100:7.1f}%")
    return lines


def zs_roles_on(path: Path, parent, stack, device, cfg, label: str):
    """Zero-shot TinySeq from parent57 on external path; exact@1 + roles."""
    raw = read_tiny_external(path)
    for ln in raw:
        ln["subkind"] = line_subkind(ln)
    tr, ho = split_external(raw, cfg.scale_hold_frac, seed=DENSE_EVAL_SEED)
    for ln in tr + ho:
        ln["subkind"] = line_subkind(ln)
    base_surfaces = list(parent["surfaces"])
    base_fps = parent["word_fps"]
    if not torch.is_tensor(base_fps):
        base_fps = torch.stack(base_fps)
    surf, fps_cpu, _ = grow_codebook(stack, base_surfaces, base_fps, raw)
    word_fps = F.normalize(fps_cpu.to(device), dim=-1)
    stoi = {s: i for i, s in enumerate(surf)}
    model = TinySeqLM(cfg.dim).to(device)
    model.load_state_dict(parent["tiny_seq"])
    model.eval()
    ho_pairs = lines_to_pairs(ho, stoi)
    l2 = bucket(ho_pairs, 2)
    cb = Codebook(surf, word_fps)
    roles_all = eval_hit1_by_role(model, l2, ho, word_fps, surf, stoi)
    by = {}
    for sk in ("noun_rel", "polarity", "verb_ing"):
        sub = filter_pairs_subkind(l2, ho, sk)
        by[sk] = eval_hit1_by_role(model, sub, ho, word_fps, surf, stoi)
    clean = eval_next(model, l2, word_fps, cb, device, shuffle_prefix=False)
    shuf = eval_next(model, l2, word_fps, cb, device, shuffle_prefix=True)
    nr = filter_pairs_subkind(l2, ho, "noun_rel")
    nr_clean = eval_next(model, nr, word_fps, cb, device, shuffle_prefix=False)
    nr_shuf = eval_next(model, nr, word_fps, cb, device, shuffle_prefix=True)
    return {
        "label": label,
        "n_lines": len(raw),
        "n_hold": len(ho),
        "roles_all": roles_all,
        "by_subkind": by,
        "L2_acc": clean["acc"],
        "order_drop": clean["acc"] - shuf["acc"],
        "noun_rel_L2": nr_clean["acc"],
        "noun_rel_order_drop": nr_clean["acc"] - nr_shuf["acc"],
        "path_hit1": by["noun_rel"].get("ALL", {}).get("hit1", 0.0),
        "rel_hit1": by["noun_rel"].get("rel", {}).get("hit1", 0.0),
        "right_hit1": by["noun_rel"].get("right", {}).get("hit1", 0.0),
    }


def object_left_rel(ex, ws):
    """For object-slot pair (prefix ends with rel): return (left_word, rel_word) or None."""
    pl = int(ex["prefix_len"])
    if pl < 2 or pl > len(ws):
        return None
    rel = ws[pl - 1]
    left = ws[pl - 2]
    if rel not in RELS:
        return None
    return left, rel


@torch.no_grad()
def pred_object_query(model, head, word_fps, ex, ws, stoi, use_head: bool):
    ids = list(ex["prefix_word_ids"][-model.max_len :])
    h = model.pred_from_prefix(word_fps[ids])
    if not use_head or head is None:
        return h
    lr = object_left_rel(ex, ws)
    if lr is None:
        return h
    left, rel = lr
    if left not in stoi or rel not in stoi:
        return h
    return head(h, word_fps[stoi[left]], word_fps[stoi[rel]])


@torch.no_grad()
def eval_object_slot(model, head, pairs, lines, word_fps, surfaces, stoi, use_head: bool):
    """Exact@1 + mean rank on object-after-rel pairs."""
    rights_set = set()
    for ln in lines:
        ws = ln["words"]
        for i, w in enumerate(ws):
            if w in RELS and i + 1 < len(ws):
                rights_set.add(ws[i + 1])
    conf = {"right_correct": 0, "other_rightish": 0, "rel": 0, "left": 0, "other": 0}
    n = 0
    h1 = 0
    ranks = []
    V = word_fps.size(0)
    for ex in pairs:
        line = lines[ex["line_i"]]
        ws = line["words"]
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        q = pred_object_query(model, head, word_fps, ex, ws, stoi, use_head)
        sims = word_fps @ q
        order = sims.argsort(descending=True)
        top_i = int(order[0])
        pred = surfaces[top_i]
        gold_i = stoi[gold]
        rank = int((order == gold_i).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
        n += 1
        hit = pred == gold
        h1 += int(hit)
        lr = object_left_rel(ex, ws)
        if hit:
            conf["right_correct"] += 1
        elif pred in RELS:
            conf["rel"] += 1
        elif lr and pred == lr[0]:
            conf["left"] += 1
        elif pred in rights_set:
            conf["other_rightish"] += 1
        else:
            conf["other"] += 1
    return {
        "n": n,
        "hit1": h1 / max(n, 1),
        "mean_rank": sum(ranks) / max(len(ranks), 1),
        "median_rank": sorted(ranks)[len(ranks) // 2] if ranks else 0,
        "chance": 1.0 / max(V, 1),
        "V": V,
        "confusion": {k: v / max(n, 1) for k, v in conf.items()},
        "confusion_n": conf,
    }


@torch.no_grad()
def eval_role_with_optional_head(model, head, pairs, lines, word_fps, surfaces, stoi):
    """Path eval: use RightHead only on object (right) steps; plain TinySeq elsewhere."""
    from collections import defaultdict

    buckets = defaultdict(lambda: {"n": 0, "h1": 0})
    for ex in pairs:
        if int(ex["prefix_len"]) < 2:
            continue
        line = lines[ex["line_i"]]
        ws = line["words"]
        role = _role(ex, ws)
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        use_head = role == "right" and head is not None
        q = pred_object_query(model, head, word_fps, ex, ws, stoi, use_head)
        pred = surfaces[int((word_fps @ q).argmax())]
        for name in (role, "ALL"):
            buckets[name]["n"] += 1
            buckets[name]["h1"] += int(pred == gold)
    return {k: {"n": v["n"], "hit1": v["h1"] / max(v["n"], 1)} for k, v in buckets.items()}


def collect_noun_pool(ext_path: Path) -> list[str]:
    """In-lexicon-ish nouns: no rel/polarity/ing; prefer words seen in external + PLAIN."""
    ban = set(RELS) | {"no", "not", "big", "red", "old", "new", "good", "a", "the"}
    words = set(PLAIN_EXTRA)
    if ext_path.exists():
        for raw in ext_path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            for w in normalize_text(s).split():
                words.add(w)
    out = []
    for w in sorted(words):
        if w in ban:
            continue
        if w.endswith("ing") and w != "ing":
            continue
        if len(w) < 2 or any(c not in CHAR2ID or c == " " for c in w):
            continue
        out.append(w)
    return out


def sample_triples(nouns: list[str], n: int, rng: random.Random, forbid: set[tuple]) -> list[tuple]:
    got = []
    seen = set(forbid)
    guard = 0
    while len(got) < n and guard < n * 200:
        guard += 1
        left = rng.choice(nouns)
        right = rng.choice(nouns)
        if left == right:
            continue
        rel = rng.choice(list(RELS))
        t = (left, rel, right)
        if t in seen:
            continue
        seen.add(t)
        got.append(t)
    return got


def build_noun_rel_repeat_corpus(cfg, nouns: list[str], seed: int = 272):
    """
    Fat triples (many copies) + rare-hold triples (never in train).
    Returns train_lines, hold_seen_lines, hold_rare_lines, meta.
    """
    rng = random.Random(seed)
    n_fat = int(getattr(cfg, "nr_n_fat", 120))
    copies = int(getattr(cfg, "nr_fat_copies", 30))
    n_rare = int(getattr(cfg, "nr_n_rare", 60))
    rare_copies = int(getattr(cfg, "nr_rare_copies", 3))
    seen_hold_frac = float(getattr(cfg, "nr_seen_hold_frac", 0.20))

    fat = sample_triples(nouns, n_fat, rng, forbid=set())
    rare = sample_triples(nouns, n_rare, rng, forbid=set(fat))

    train, hold_seen = [], []
    for t in fat:
        phrase = f"{t[0]} {t[1]} {t[2]}"
        n_hold = max(1, int(round(copies * seen_hold_frac)))
        n_tr = copies - n_hold
        for _ in range(n_tr):
            train.append({"phrase": phrase, "words": list(t), "triple": t, "bucket": "fat_train"})
        for _ in range(n_hold):
            hold_seen.append({"phrase": phrase, "words": list(t), "triple": t, "bucket": "fat_hold"})

    hold_rare = []
    for t in rare:
        phrase = f"{t[0]} {t[1]} {t[2]}"
        for _ in range(rare_copies):
            hold_rare.append({"phrase": phrase, "words": list(t), "triple": t, "bucket": "rare_hold"})

    rng.shuffle(train)
    rng.shuffle(hold_seen)
    rng.shuffle(hold_rare)
    meta = {
        "n_fat_triples": len(fat),
        "n_rare_triples": len(rare),
        "fat_copies": copies,
        "rare_copies": rare_copies,
        "n_train": len(train),
        "n_hold_seen": len(hold_seen),
        "n_hold_rare": len(hold_rare),
        "n_nouns": len(nouns),
    }
    return train, hold_seen, hold_rare, meta, fat, rare


def write_nr_corpus(path: Path, train, hold_seen, hold_rare, meta):
    lines = [
        "# SOTE Stage83 noun_rel repeat corpus (left rel right only; no ing)",
        f"# meta: {json.dumps(meta)}",
        "# split markers: TRAIN / HOLD_SEEN / HOLD_RARE",
        "# TRAIN",
    ]
    for ln in train:
        lines.append(ln["phrase"])
    lines.append("# HOLD_SEEN")
    for ln in hold_seen:
        lines.append(ln["phrase"])
    lines.append("# HOLD_RARE")
    for ln in hold_rare:
        lines.append(ln["phrase"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def roles_on_lines(model, lines, word_fps, surfaces, stoi, min_prefix=1):
    """Include L1 so rel-after-left is visible on 3-word noun_rel lines."""
    pairs = lines_to_pairs(lines, stoi)
    return eval_hit1_by_role(model, pairs, lines, word_fps, surfaces, stoi, min_prefix=min_prefix)


@torch.no_grad()
def object_on_lines(model, lines, word_fps, surfaces, stoi):
    pairs = collect_object_pairs(lines_to_pairs(lines, stoi), lines)
    return eval_object_slot(model, None, pairs, lines, word_fps, surfaces, stoi, use_head=False)


def filter_tinystories_chunk(
    raw_path: Path,
    out_path: Path,
    max_lines: int = 4000,
    min_words: int = 3,
    max_words: int = 8,
    seed: int = 272,
    max_word_len: int = 11,
):
    """Lowercase; keep a-z, digits, spaces; sentence windows; prefer on|to.

    Early-stops collecting once pools cover ~2x target (needed for 100k-scale).
    """
    import re

    raw = raw_path.read_text(encoding="utf-8", errors="ignore")
    text = raw.lower()
    # Stage85+: digits kept (not stripped)
    text = re.sub(r"[^a-z0-9\s.!?]+", " ", text)
    chunks = re.split(r"[.!?\n]+", text)
    pathish, other = [], []
    # oversample pool then uniq; stop early for large max_lines
    pool_target = max(max_lines * 2, max_lines + 2000)
    path_target = int(pool_target * 0.70)
    other_target = pool_target - path_target
    for ch in chunks:
        if len(pathish) >= path_target and len(other) >= other_target:
            break
        words = [
            w for w in ch.split()
            if w and all(c in CHAR2ID and c != " " for c in w) and len(w) <= max_word_len
        ]
        if len(words) < min_words:
            continue
        for start in range(0, max(1, len(words) - min_words + 1)):
            win = words[start : start + max_words]
            if len(win) < min_words:
                break
            if len(win) > max_words:
                win = win[:max_words]
            line = " ".join(win)
            if "on" in win or "to" in win:
                if len(pathish) < path_target:
                    pathish.append(line)
            else:
                if len(other) < other_target:
                    other.append(line)
            if start + max_words >= len(words):
                break
            if len(pathish) >= path_target and len(other) >= other_target:
                break
    rng = random.Random(seed)
    rng.shuffle(pathish)
    rng.shuffle(other)
    n_path = min(len(pathish), int(max_lines * 0.70))
    n_other = min(len(other), max_lines - n_path)
    lines = pathish[:n_path] + other[:n_other]
    rng.shuffle(lines)
    seen = set()
    uniq = []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        uniq.append(ln)
    uniq = uniq[:max_lines]
    meta = {
        "source": str(raw_path.name),
        "n_pathish": n_path,
        "n_other": n_other,
        "n_out": len(uniq),
        "min_words": min_words,
        "max_words": max_words,
        "digits_kept": True,
        "max_word_len": max_word_len,
        "pool_path": len(pathish),
        "pool_other": len(other),
    }
    header = [
        "# SOTE TinyStories (a-z + digits + space; Stage85+)",
        f"# meta: {json.dumps(meta)}",
    ]
    out_path.write_text("\n".join(header + uniq) + "\n", encoding="utf-8")
    return uniq, meta


def build_foundation_lexicon(cfg) -> tuple[list[str], list[str], list[str]]:
    """Words (plain+digit), morph surfaces, short phrases for encode_B."""
    words = set(PLAIN_EXTRA)
    for stem in MORPH_STEMS:
        words.add(stem)
        for aff in AFFIXES:
            words.add(surface(stem, aff))
    # digits & alnum mixes
    for i in range(100):
        words.add(str(i))
    for w in list(PLAIN_EXTRA)[:40]:
        words.add(f"{w}2")
        words.add(f"{w}10")
        words.add(f"2{w}" if len(w) + 1 <= cfg.max_word_len else w)
    # harvest from existing corpora if present
    for path in (EXT_1K, ROOT / "data" / "external_tinystories_mini.txt"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            for tok in normalize_text(s).split():
                if 1 <= len(tok) <= cfg.max_word_len and all(c in CHAR2ID and c != " " for c in tok):
                    words.add(tok)
    words = sorted(w for w in words if 1 <= len(w) <= cfg.max_word_len)
    digitish = [w for w in words if any(c.isdigit() for c in w)]
    # morph surfaces
    morph_words = []
    for stem in MORPH_STEMS:
        for aff in AFFIXES:
            morph_words.append(surface(stem, aff))
    morph_words = sorted(set(morph_words))
    # phrases: path-like + digit objects
    phrases = []
    nouns = [w for w in PLAIN_EXTRA if w not in RELS and w not in ("no", "not")]
    for i in range(min(200, len(nouns) * 2)):
        left = nouns[i % len(nouns)]
        right = nouns[(i * 3) % len(nouns)]
        rel = RELS[i % 2]
        phrases.append(f"{left} {rel} {right}")
        phrases.append(f"{left} {rel} {i % 50}")
    phrases = sorted(set(normalize_text(p) for p in phrases))
    return words, digitish, phrases


def compose_batch(encoder, composer, words, device, max_word_len):
    """Return [B,D] fps for list of words (plain compose, trainable)."""
    fps_list, lens = [], []
    max_l = 1
    for w in words:
        chars = [c for c in w if c in CHAR2ID]
        if not chars:
            chars = ["a"]
        chars = chars[:max_word_len]
        ids = torch.tensor([CHAR2ID[c] for c in chars], device=device)
        cf = encoder(ids)
        fps_list.append(cf)
        lens.append(cf.size(0))
        max_l = max(max_l, cf.size(0))
    bsz = len(words)
    mat = torch.zeros(bsz, max_l, encoder.emb.embedding_dim, device=device)
    for i, cf in enumerate(fps_list):
        mat[i, : cf.size(0)] = cf
    lengths = torch.tensor(lens, device=device)
    return composer(mat, lengths)


def train_word_foundation(encoder, composer, words, cfg, device):
    opt = torch.optim.Adam(list(encoder.parameters()) + list(composer.parameters()), lr=cfg.f85_lr)
    rng = random.Random(cfg.seed)
    steps = int(cfg.f85_word_steps)
    every = int(cfg.f85_eval_every)
    bs = int(cfg.f85_batch)
    curve = []
    best = {"step": 0, "exact": -1.0, "state_e": None, "state_c": None}

    def _eval(step):
        encoder.eval()
        composer.eval()
        # small codebook eval on sample
        sample = sorted(rng.sample(words, min(400, len(words))))
        with torch.no_grad():
            fps = []
            for w in sample:
                fps.append(compose_plain(encoder, composer, w, device))
            mat = F.normalize(torch.stack(fps, 0), dim=-1)
            ok = 0
            for i, w in enumerate(sample):
                pred_i = int((mat @ mat[i]).argmax())
                ok += int(pred_i == i)
            # foil: shuffled chars should not match
            foil_ok = 0
            foil_n = 0
            for w in sample[:80]:
                if len(w) < 3:
                    continue
                chars = list(w)
                rng.shuffle(chars)
                sw = "".join(chars)
                if sw == w:
                    continue
                fw = compose_plain(encoder, composer, sw, device)
                gold = compose_plain(encoder, composer, w, device)
                foil_n += 1
                foil_ok += int(float((fw * gold).sum()) < 0.95)
        exact = ok / max(len(sample), 1)
        foil = foil_ok / max(foil_n, 1)
        curve.append({"step": step, "exact": exact, "foil": foil, "n": len(sample)})
        if exact >= best["exact"]:
            best.update(
                step=step,
                exact=exact,
                state_e={k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                state_c={k: v.detach().cpu().clone() for k, v in composer.state_dict().items()},
            )
        print(f"  word step {step:4d}: codebook_exact={exact*100:.1f}% foil_sep={foil*100:.1f}%", flush=True)
        encoder.train()
        composer.train()

    _eval(0)
    encoder.train()
    composer.train()
    for step in range(1, steps + 1):
        batch = [rng.choice(words) for _ in range(bs)]
        # unique-ish
        batch = list(dict.fromkeys(batch))
        if len(batch) < 8:
            continue
        pred = compose_batch(encoder, composer, batch, device, cfg.max_word_len)
        # InfoNCE in-batch
        logits = pred @ pred.T
        labels = torch.arange(pred.size(0), device=device)
        loss = F.cross_entropy(logits / 0.07, labels)
        # soft attractor: each word vs its own (redundant with diag) + perm foil
        foil_loss = torch.tensor(0.0, device=device)
        for w in batch[:16]:
            if len(w) < 3:
                continue
            chars = list(w)
            rng.shuffle(chars)
            sw = "".join(chars)[: cfg.max_word_len]
            if sw == w or not sw:
                continue
            fw = compose_batch(encoder, composer, [sw], device, cfg.max_word_len)[0]
            gw = compose_batch(encoder, composer, [w], device, cfg.max_word_len)[0]
            foil_loss = foil_loss + F.relu((fw * gw).sum() - 0.80)
        loss = loss + 0.25 * foil_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            _eval(step)
    encoder.load_state_dict(best["state_e"])
    composer.load_state_dict(best["state_c"])
    encoder.eval()
    composer.eval()
    return best, curve


def train_morph_foundation(morph, encoder, composer, cfg, device):
    for p in list(encoder.parameters()) + list(composer.parameters()):
        p.requires_grad_(False)
    encoder.eval()
    composer.eval()
    opt = torch.optim.Adam(morph.parameters(), lr=cfg.f85_lr)
    rng = random.Random(cfg.seed + 1)
    pairs = []
    for stem in MORPH_STEMS:
        for aff in AFFIXES:
            pairs.append((stem, aff, surface(stem, aff)))
    steps = int(cfg.f85_morph_steps)
    every = int(cfg.f85_eval_every)
    best = {"step": 0, "cos": -1.0, "state": None}
    curve = []

    def _eval(step):
        morph.eval()
        cos_sum = 0.0
        n = 0
        with torch.no_grad():
            for stem, aff, surf in pairs:
                pred = reattach_morph(morph, encoder, composer, stem, aff, device)
                tgt = compose_plain(encoder, composer, surf, device)
                cos_sum += float((F.normalize(pred, dim=-1) * F.normalize(tgt, dim=-1)).sum())
                n += 1
        cos = cos_sum / max(n, 1)
        # NN among morph surfaces
        surfs = [s for _, _, s in pairs]
        fps = torch.stack([compose_plain(encoder, composer, s, device) for s in surfs], 0)
        fps = F.normalize(fps, dim=-1)
        ok = 0
        for i, (stem, aff, surf) in enumerate(pairs):
            q = F.normalize(reattach_morph(morph, encoder, composer, stem, aff, device), dim=-1)
            pred_i = int((fps @ q).argmax())
            ok += int(surfs[pred_i] == surf)
        exact = ok / max(len(pairs), 1)
        curve.append({"step": step, "cos": cos, "exact": exact})
        if cos >= best["cos"]:
            best.update(step=step, cos=cos, exact=exact, state={k: v.detach().cpu().clone() for k, v in morph.state_dict().items()})
        print(f"  morph step {step:4d}: realize_cos={cos*100:.1f}% surface_NN={exact*100:.1f}%", flush=True)
        morph.train()

    _eval(0)
    morph.train()
    for step in range(1, steps + 1):
        stem, aff, surf = rng.choice(pairs)
        pred = reattach_morph(morph, encoder, composer, stem, aff, device)
        with torch.no_grad():
            tgt = compose_plain(encoder, composer, surf, device)
        loss = 1.0 - (F.normalize(pred, dim=-1) * F.normalize(tgt, dim=-1)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            _eval(step)
    morph.load_state_dict(best["state"])
    morph.eval()
    for p in list(encoder.parameters()) + list(composer.parameters()):
        p.requires_grad_(True)
    return best, curve


def train_phrase_foundation(phrase, encoder, composer, morph, phrases, cfg, device):
    for p in list(encoder.parameters()) + list(composer.parameters()) + list(morph.parameters()):
        p.requires_grad_(False)
    opt = torch.optim.Adam(phrase.parameters(), lr=cfg.f85_lr)
    rng = random.Random(cfg.seed + 2)
    steps = int(cfg.f85_phrase_steps)
    every = int(cfg.f85_eval_every)
    bs = min(32, len(phrases))
    best = {"step": 0, "recon": -1.0, "state": None}
    curve = []

    def _word(w):
        return word_fp(encoder, composer, morph, w, device).detach()

    def _eval(step):
        phrase.eval()
        # write mem and recon
        ok = 0
        sample = phrases[: min(200, len(phrases))]
        with torch.no_grad():
            keys, fps = [], []
            for p in sample:
                ws = p.split()
                if len(ws) > cfg.max_phrase_len - 1:
                    ws = ws[: cfg.max_phrase_len - 1]
                wf = torch.stack([_word(w) for w in ws], 0)
                out, _, _ = phrase(wf.unsqueeze(0), torch.tensor([wf.size(0)], device=device))
                keys.append(p)
                fps.append(out.squeeze(0))
            mat = F.normalize(torch.stack(fps, 0), dim=-1)
            for i in range(len(keys)):
                if int((mat @ mat[i]).argmax()) == i:
                    ok += 1
        recon = ok / max(len(keys), 1)
        curve.append({"step": step, "recon": recon, "n": len(keys)})
        if recon >= best["recon"]:
            best.update(step=step, recon=recon, state={k: v.detach().cpu().clone() for k, v in phrase.state_dict().items()})
        print(f"  phrase step {step:4d}: recon_NN={recon*100:.1f}%", flush=True)
        phrase.train()

    _eval(0)
    phrase.train()
    for step in range(1, steps + 1):
        batch = [rng.choice(phrases) for _ in range(bs)]
        batch = list(dict.fromkeys(batch))
        if len(batch) < 4:
            continue
        outs = []
        for p in batch:
            ws = p.split()[: cfg.max_phrase_len - 1]
            wf = torch.stack([_word(w) for w in ws], 0)
            out, _, _ = phrase(wf.unsqueeze(0), torch.tensor([wf.size(0)], device=device))
            outs.append(out.squeeze(0))
        pred = F.normalize(torch.stack(outs, 0), dim=-1)
        loss = F.cross_entropy(pred @ pred.T / 0.07, torch.arange(pred.size(0), device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            _eval(step)
    phrase.load_state_dict(best["state"])
    phrase.eval()
    return best, curve


def load_foundation_85(device, cfg, path: Path | None = None):
    """Load Stage85 encode stack (dim256 + digits). Not V1 load_stack."""
    path = path or PARENT85
    assert path.exists(), f"missing {path}"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert int(ck.get("dim", cfg.dim)) == int(cfg.dim), "dim mismatch vs stage85"
    encoder = LetterEncoder(len(CHARS), cfg.dim).to(device)
    composer = WordComposer(cfg.dim, int(ck.get("max_word_len", cfg.max_word_len))).to(device)
    morph = MorphModPlus(cfg.dim, len(AFFIXES), max_suffix=4).to(device)
    phrase = PhraseComposer(cfg.dim, cfg.max_phrase_len).to(device)
    binder = CueBinder(cfg.dim).to(device)
    encoder.load_state_dict(ck["encoder"], strict=True)
    composer.load_state_dict(ck["composer"], strict=True)
    morph.load_state_dict(ck["morph"], strict=True)
    phrase.load_state_dict(ck["phrase_composer"], strict=True)
    if "cue_binder" in ck:
        binder.load_state_dict(ck["cue_binder"], strict=False)
    for mod in (encoder, composer, morph, phrase, binder):
        for p in mod.parameters():
            p.requires_grad_(False)
        mod.eval()
    return Stack(encoder, composer, morph, phrase, binder, device), ck


def target_freq_bucket(count: int) -> str:
    if count <= 1:
        return "rare1"
    if count <= 5:
        return "low2_5"
    if count <= 20:
        return "mid6_20"
    return "head21p"


def build_bigram_table(train_lines):
    """Majority next given previous token (train only)."""
    from collections import Counter, defaultdict

    nexts = defaultdict(Counter)
    unigram = Counter()
    for ln in train_lines:
        ws = ln["words"]
        for w in ws:
            unigram[w] += 1
        for i in range(len(ws) - 1):
            nexts[ws[i]][ws[i + 1]] += 1
    majority = {}
    for prev, ctr in nexts.items():
        majority[prev] = ctr.most_common(1)[0][0]
    uni_maj = unigram.most_common(1)[0][0] if unigram else ""
    return majority, unigram, uni_maj


@torch.no_grad()
def eval_capacity_suite(model, hold_lines, train_lines, word_fps, surfaces, stoi, device):
    """SOTE roles + freq buckets + seen/unseen + object + bigram baseline."""
    from collections import Counter, defaultdict

    majority, unigram, uni_maj = build_bigram_table(train_lines)
    pairs = lines_to_pairs(hold_lines, stoi)
    roles = eval_hit1_by_role(model, pairs, hold_lines, word_fps, surfaces, stoi, min_prefix=1)
    obj_pairs = collect_object_pairs(pairs, hold_lines)
    obj = eval_object_slot(model, None, obj_pairs, hold_lines, word_fps, surfaces, stoi, use_head=False)

    # per-example SOTE + bigram
    freq_b = defaultdict(lambda: {"n": 0, "sote": 0, "bigram": 0})
    seen_b = {"seen": {"n": 0, "sote": 0, "bigram": 0}, "unseen": {"n": 0, "sote": 0, "bigram": 0}}
    role_bigram = defaultdict(lambda: {"n": 0, "hit": 0})

    for ex in pairs:
        if int(ex["prefix_len"]) < 1:
            continue
        line = hold_lines[ex["line_i"]]
        ws = line["words"]
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        role = _role(ex, ws)
        # SOTE
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = model.pred_from_prefix(word_fps[ids])
        sote_w = surfaces[int((word_fps @ pred).argmax())]
        sote_ok = int(sote_w == gold)
        # bigram
        prev = ws[ex["prefix_len"] - 1] if ex["prefix_len"] >= 1 else ""
        bg_w = majority.get(prev, uni_maj)
        bg_ok = int(bg_w == gold)
        cnt = unigram.get(gold, 0)
        bucket = target_freq_bucket(cnt)
        freq_b[bucket]["n"] += 1
        freq_b[bucket]["sote"] += sote_ok
        freq_b[bucket]["bigram"] += bg_ok
        sk = "seen" if cnt > 0 else "unseen"
        seen_b[sk]["n"] += 1
        seen_b[sk]["sote"] += sote_ok
        seen_b[sk]["bigram"] += bg_ok
        role_bigram[role]["n"] += 1
        role_bigram[role]["hit"] += bg_ok
        role_bigram["ALL"]["n"] += 1
        role_bigram["ALL"]["hit"] += bg_ok

    def _pack(d):
        out = {}
        for k, v in d.items():
            n = max(v["n"], 1)
            out[k] = {
                "n": v["n"],
                "sote": v["sote"] / n,
                "bigram": v["bigram"] / n,
            }
        return out

    freq = _pack(freq_b)
    seen = _pack(seen_b)
    bg_roles = {
        k: {"n": v["n"], "hit1": v["hit"] / max(v["n"], 1)} for k, v in role_bigram.items()
    }
    # hyp BPE: same data, soft next — expect ~bigram on rare, between bigram and
    # SOTE-on-head for mid; slight lift vs exact NN on collisions. Use midpoint heuristic.
    hyp = {}
    for k, v in freq.items():
        # conservative: BPE soft@1 ≈ 1.15 * bigram capped, or mid(sote, bigram)+small if sote>bigram
        base = v["bigram"]
        if v["sote"] >= base:
            est = min(0.95, base + 0.5 * (v["sote"] - base) + 0.02)
        else:
            est = min(0.95, max(base, v["sote"]) * 1.10)
        hyp[k] = {"n": v["n"], "est_hit1": est, "sote": v["sote"], "bigram": v["bigram"]}

    return {
        "roles": roles,
        "obj": obj,
        "freq": freq,
        "seen": seen,
        "bigram_roles": bg_roles,
        "hyp_bpe_freq": hyp,
        "n_pairs": len(pairs),
        "V": len(surfaces),
        "train_unigram_n": sum(unigram.values()),
        "train_vocab": len(unigram),
    }


def fmt_capacity_block(suite: dict, title: str) -> list[str]:
    lines = [f"=== {title} ==="]
    r = suite["roles"]
    lines.append(
        f"  SOTE ALL@1={r.get('ALL',{}).get('hit1',0)*100:.1f}% "
        f"rel={r.get('rel',{}).get('hit1',0)*100:.1f}% "
        f"right={r.get('right',{}).get('hit1',0)*100:.1f}% "
        f"obj_slot={suite['obj']['hit1']*100:.1f}% (n={suite['obj']['n']})"
    )
    bg = suite["bigram_roles"]
    lines.append(
        f"  bigram ALL={bg.get('ALL',{}).get('hit1',0)*100:.1f}% "
        f"rel={bg.get('rel',{}).get('hit1',0)*100:.1f}% "
        f"right={bg.get('right',{}).get('hit1',0)*100:.1f}%"
    )
    lines.append("  by target freq (train count):  SOTE / bigram / hypBPE~")
    order = ["head21p", "mid6_20", "low2_5", "rare1"]
    for k in order:
        if k not in suite["freq"]:
            continue
        v = suite["freq"][k]
        h = suite["hyp_bpe_freq"][k]["est_hit1"]
        lines.append(
            f"    {k:8s} n={v['n']:5d}  sote={v['sote']*100:5.1f}%  "
            f"bigram={v['bigram']*100:5.1f}%  hypBPE~{h*100:5.1f}%"
        )
    for sk in ("seen", "unseen"):
        v = suite["seen"][sk]
        lines.append(
            f"  target {sk:6s}: sote={v['sote']*100:.1f}% bigram={v['bigram']*100:.1f}% n={v['n']}"
        )
    return lines


def extract_path_triples(lines):
    """Atomic (left, rel, right) from windows containing on|to."""
    trips = []
    for ln in lines:
        ws = ln["words"] if isinstance(ln, dict) else str(ln).split()
        for i, w in enumerate(ws):
            if w in RELS and i >= 1 and i + 1 < len(ws):
                trips.append((ws[i - 1], w, ws[i + 1]))
    return trips


def build_ts_repeat_mix(base_phrases, cfg, seed=272):
    """
    Mix original TinyStories windows with fat-repeated path triples
    extracted from them (83b-style density on SEEN frames).
    """
    rng = random.Random(seed)
    base_lines = []
    for p in base_phrases:
        ws = p.split()
        base_lines.append({"phrase": p, "words": ws, "bucket": "story", "subkind": line_subkind({"words": ws})})

    trips = extract_path_triples(base_lines)
    from collections import Counter

    ctr = Counter(trips)
    # keep triples seen at least once; prefer more frequent for fat pool
    uniq = [t for t, _ in ctr.most_common()]
    n_fat = min(int(getattr(cfg, "c87_n_fat", 150)), len(uniq))
    fat_triples = uniq[:n_fat]
    rare_pool = uniq[n_fat : n_fat + int(getattr(cfg, "c87_n_rare", 60))]
    copies = int(getattr(cfg, "c87_fat_copies", 60))
    rare_copies = int(getattr(cfg, "c87_rare_copies", 2))
    seen_hold_frac = float(getattr(cfg, "c87_seen_hold_frac", 0.20))
    story_keep = float(getattr(cfg, "c87_story_keep_frac", 0.40))

    # story subsample (keep some natural diversity)
    rng.shuffle(base_lines)
    n_story = max(1, int(len(base_lines) * story_keep))
    story = base_lines[:n_story]

    train, hold_seen, hold_rare, hold_story = [], [], [], []
    # split story
    rng.shuffle(story)
    n_sh = max(1, int(len(story) * float(getattr(cfg, "c87_hold_frac", 0.25))))
    hold_story = story[:n_sh]
    train.extend(story[n_sh:])

    for t in fat_triples:
        phrase = f"{t[0]} {t[1]} {t[2]}"
        n_hold = max(1, int(round(copies * seen_hold_frac)))
        n_tr = max(1, copies - n_hold)
        for _ in range(n_tr):
            train.append({
                "phrase": phrase, "words": list(t), "bucket": "fat_train",
                "triple": t, "subkind": "noun_rel",
            })
        for _ in range(n_hold):
            hold_seen.append({
                "phrase": phrase, "words": list(t), "bucket": "fat_hold",
                "triple": t, "subkind": "noun_rel",
            })

    for t in rare_pool:
        phrase = f"{t[0]} {t[1]} {t[2]}"
        for _ in range(rare_copies):
            hold_rare.append({
                "phrase": phrase, "words": list(t), "bucket": "rare_hold",
                "triple": t, "subkind": "noun_rel",
            })

    rng.shuffle(train)
    rng.shuffle(hold_seen)
    rng.shuffle(hold_rare)
    rng.shuffle(hold_story)
    meta = {
        "n_story_train": len(story) - n_sh,
        "n_story_hold": len(hold_story),
        "n_fat_triples": len(fat_triples),
        "fat_copies": copies,
        "n_train": len(train),
        "n_hold_seen": len(hold_seen),
        "n_hold_rare": len(hold_rare),
        "n_path_triples_avail": len(uniq),
        "top_triple_freq": ctr.most_common(3),
    }
    return train, hold_seen, hold_rare, hold_story, meta



class WordIdTransformer(nn.Module):
    """Causal Transformer LM over SOTE word ids (next-atom = next word_id).

    Vocab rows 0..V-1 are real atoms; row V is PAD (batch left-pad only).
    """

    def __init__(
        self,
        n_vocab: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.pad_id = n_vocab
        self.d_model = d_model
        self.max_len = max_len
        self.tok = nn.Embedding(n_vocab + 1, d_model, padding_idx=self.pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        # decode via tied non-pad rows of tok (see forward)

    @torch.no_grad()
    def init_from_fps(self, word_fps: torch.Tensor):
        """Copy SOTE word fingerprints into embedding rows (pad/truncate dim)."""
        V, d_fp = word_fps.shape
        assert V == self.n_vocab
        w = torch.zeros(V + 1, self.d_model, device=word_fps.device, dtype=self.tok.weight.dtype)
        d = min(self.d_model, d_fp)
        w[:V, :d] = word_fps[:, :d].to(dtype=w.dtype)
        w[:V] = F.normalize(w[:V], dim=-1)
        self.tok.weight.copy_(w)

    def forward(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """ids: [B, T] -> logits [B, T, V] (V = real vocab, not pad)."""
        bsz, tmax = ids.shape
        device = ids.device
        pos = torch.arange(tmax, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.tok(ids) + self.pos(pos)
        # causal mask: True = blocked for TransformerEncoder
        mask = torch.triu(torch.ones(tmax, tmax, device=device, dtype=torch.bool), diagonal=1)
        h = self.tr(x, mask=mask, src_key_padding_mask=key_padding_mask)
        return F.linear(h, self.tok.weight[: self.n_vocab])

    def logits_from_prefix(self, id_list) -> torch.Tensor:
        if not id_list:
            raise ValueError("empty prefix")
        ids = torch.tensor([id_list[-self.max_len :]], dtype=torch.long, device=self.tok.weight.device)
        return self.forward(ids)[0, -1]

    def pred_id(self, id_list) -> int:
        return int(self.logits_from_prefix(id_list).argmax())

    def logits_last_from_batch(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """Left-padded batch -> logits at last (rightmost) position [B, V]."""
        return self.forward(ids, key_padding_mask=key_padding_mask)[:, -1, :]


def collate_word_id_batch(examples, stoi, max_len: int, pad_id: int, device):
    """Pack SOTE word-id prefixes: left-pad so last col is end of prefix."""
    prefs, targets = [], []
    for ex in examples:
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        ids = list(ex["prefix_word_ids"][-max_len :])
        if not ids:
            continue
        prefs.append(ids)
        targets.append(stoi[gold])
    if not prefs:
        return None
    tmax = max(len(p) for p in prefs)
    bsz = len(prefs)
    ids = torch.full((bsz, tmax), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.ones((bsz, tmax), dtype=torch.bool, device=device)
    for i, p in enumerate(prefs):
        ids[i, -len(p) :] = torch.tensor(p, dtype=torch.long, device=device)
        pad_mask[i, -len(p) :] = False
    tgt = torch.tensor(targets, dtype=torch.long, device=device)
    return ids, pad_mask, tgt


@torch.no_grad()
def eval_id_hit1_by_role(model, pairs, lines, surfaces, stoi, min_prefix: int = 1):
    from collections import defaultdict

    buckets = defaultdict(lambda: {"n": 0, "h1": 0})
    for ex in pairs:
        if int(ex["prefix_len"]) < min_prefix:
            continue
        line = lines[ex["line_i"]]
        role = _role(ex, line["words"])
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = surfaces[model.pred_id(ids)]
        for name in (role, "ALL"):
            buckets[name]["n"] += 1
            buckets[name]["h1"] += int(pred == gold)
    return {k: {"n": v["n"], "hit1": v["h1"] / max(v["n"], 1)} for k, v in buckets.items()}


@torch.no_grad()
def eval_id_object_slot(model, pairs, lines, surfaces, stoi):
    if not pairs:
        return {"hit1": 0.0, "n": 0}
    h1 = 0
    n = 0
    for ex in pairs:
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = surfaces[model.pred_id(ids)]
        h1 += int(pred == gold)
        n += 1
    return {"hit1": h1 / max(n, 1), "n": n}


@torch.no_grad()
def eval_id_capacity_suite(model, hold_lines, train_lines, surfaces, stoi, device):
    """Roles + freq + object for id-CE Transformer (exact@1, no soft@5)."""
    from collections import defaultdict

    majority, unigram, uni_maj = build_bigram_table(train_lines)
    pairs = lines_to_pairs(hold_lines, stoi)
    roles = eval_id_hit1_by_role(model, pairs, hold_lines, surfaces, stoi, min_prefix=1)
    obj_pairs = collect_object_pairs(pairs, hold_lines)
    obj = eval_id_object_slot(model, obj_pairs, hold_lines, surfaces, stoi)

    freq_b = defaultdict(lambda: {"n": 0, "model": 0, "bigram": 0})
    seen_b = {"seen": {"n": 0, "model": 0, "bigram": 0}, "unseen": {"n": 0, "model": 0, "bigram": 0}}
    role_bigram = defaultdict(lambda: {"n": 0, "hit": 0})

    for ex in pairs:
        if int(ex["prefix_len"]) < 1:
            continue
        line = hold_lines[ex["line_i"]]
        ws = line["words"]
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        role = _role(ex, ws)
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred_w = surfaces[model.pred_id(ids)]
        model_ok = int(pred_w == gold)
        prev = ws[ex["prefix_len"] - 1] if ex["prefix_len"] >= 1 else ""
        bg_ok = int(majority.get(prev, uni_maj) == gold)
        cnt = unigram.get(gold, 0)
        bucket = target_freq_bucket(cnt)
        freq_b[bucket]["n"] += 1
        freq_b[bucket]["model"] += model_ok
        freq_b[bucket]["bigram"] += bg_ok
        sk = "seen" if cnt > 0 else "unseen"
        seen_b[sk]["n"] += 1
        seen_b[sk]["model"] += model_ok
        seen_b[sk]["bigram"] += bg_ok
        role_bigram[role]["n"] += 1
        role_bigram[role]["hit"] += bg_ok
        role_bigram["ALL"]["n"] += 1
        role_bigram["ALL"]["hit"] += bg_ok

    def _pack(d):
        out = {}
        for k, v in d.items():
            n = max(v["n"], 1)
            out[k] = {"n": v["n"], "model": v["model"] / n, "bigram": v["bigram"] / n}
        return out

    return {
        "roles": roles,
        "obj": obj,
        "freq": _pack(freq_b),
        "seen": _pack(seen_b),
        "bigram_roles": {
            k: {"n": v["n"], "hit1": v["hit"] / max(v["n"], 1)} for k, v in role_bigram.items()
        },
        "n_pairs": len(pairs),
        "V": len(surfaces),
    }




import math


def _cosine_lr(step: int, peak: float, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay to peak * min_ratio."""
    if step <= 0:
        return peak * min_ratio
    if warmup > 0 and step <= warmup:
        return peak * (step / float(warmup))
    t = max(total - warmup, 1)
    progress = min(max(step - warmup, 0) / float(t), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cos)


def _warmup_then_constant(step: int, peak: float, warmup: int) -> float:
    if step <= 0:
        return peak * 0.01
    if warmup > 0 and step < warmup:
        return peak * (step / float(warmup))
    return peak


def _warmup_then_constant(step: int, peak: float, warmup: int) -> float:
    if step <= 0:
        return peak * 0.01
    if warmup > 0 and step < warmup:
        return peak * (step / float(warmup))
    return peak


def _warmup_then_constant(step: int, peak: float, warmup: int) -> float:
    if step <= 0:
        return peak * 0.01
    if warmup > 0 and step < warmup:
        return peak * (step / float(warmup))
    return peak


def _run_one_batch(
    *,
    batch_sz: int,
    train, hold_seen, hold_rare, hold_story,
    word_fps, surf, stoi, cfg, device, seed_offset: int,
):
    model = WordIdTransformer(
        n_vocab=len(surf),
        d_model=int(cfg.c95_d_model),
        n_heads=int(cfg.c95_n_heads),
        n_layers=int(cfg.c95_n_layers),
        max_len=int(cfg.c95_max_len),
        dropout=float(cfg.c95_dropout),
    ).to(device)
    model.init_from_fps(word_fps)
    steps = int(cfg.c95_ft_steps)
    peak_lr = float(cfg.c95_ft_lr)
    warmup = int(cfg.c95_warmup)
    every = int(cfg.c95_eval_every)
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    tr_pairs = lines_to_pairs(train, stoi)
    rng = random.Random(int(cfg.seed) + seed_offset)

    def _set_lr(step: int) -> float:
        lr = _warmup_then_constant(step, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    def _eval_hold(lines):
        return eval_id_capacity_suite(model, lines, train, surf, stoi, device) if lines else None

    best = {
        "step": 0,
        "obj_seen": 0.0,
        "all_seen": 0.0,
        "seen": None,
        "rare": None,
        "story": None,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def _snap(step: int, lr_now: float):
        model.eval()
        seen = _eval_hold(hold_seen)
        rare = _eval_hold(hold_rare)
        story = _eval_hold(hold_story)
        obj_s = seen["obj"]["hit1"]
        all_s = seen["roles"].get("ALL", {}).get("hit1", 0.0)
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        curve.append({
            "step": step, "obj_seen": obj_s, "rel_seen": rel_s, "all_seen": all_s,
            "obj_story": story["obj"]["hit1"],
            "head_seen": seen["freq"].get("head21p", {}).get("model", 0.0),
            "lr": lr_now,
        })
        key = (obj_s, rel_s, all_s)
        brel = 0.0 if best["seen"] is None else best["seen"]["roles"].get("rel", {}).get("hit1", 0.0)
        bkey = (best["obj_seen"], brel, best["all_seen"])
        if key >= bkey:
            best.update(
                step=step, obj_seen=obj_s, all_seen=all_s,
                seen=seen, rare=rare, story=story,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        print(
            f"    b{batch_sz} step {step:4d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% "
            f"head={seen['freq'].get('head21p',{}).get('model',0)*100:.1f}% | "
            f"STORY ALL={story['roles'].get('ALL',{}).get('hit1',0)*100:.1f}%",
            flush=True,
        )
        model.train()

    print(f"\n=== batch={batch_sz} ===", flush=True)
    _snap(0, 0.0)
    model.train()
    for step in range(1, steps + 1):
        lr_now = _set_lr(step)
        batch_ex = [rng.choice(tr_pairs) for _ in range(batch_sz)]
        packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        logits = model.logits_last_from_batch(ids, key_padding_mask=pad_mask)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            _snap(step, lr_now)

    objs = [c["obj_seen"] for c in curve if c["step"] > 0]
    swing = (max(objs) - min(objs)) if objs else 0.0
    seen = best["seen"]
    rare = best["rare"]
    story = best["story"]
    return {
        "batch": batch_sz,
        "best_step": best["step"],
        "obj": seen["obj"]["hit1"],
        "rel": seen["roles"].get("rel", {}).get("hit1", 0.0),
        "all": seen["roles"].get("ALL", {}).get("hit1", 0.0),
        "head": seen["freq"].get("head21p", {}).get("model", 0.0),
        "obj_rare": rare["obj"]["hit1"],
        "obj_story": story["obj"]["hit1"],
        "all_story": story["roles"].get("ALL", {}).get("hit1", 0.0),
        "swing": swing,
        "curve": curve,
        "state": best["state"],
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def _warmup_then_constant(step: int, peak: float, warmup: int) -> float:
    if step <= 0:
        return peak * 0.01
    if warmup > 0 and step < warmup:
        return peak * (step / float(warmup))
    return peak


def _warmup_then_constant(step: int, peak: float, warmup: int) -> float:
    if step <= 0:
        return peak * 0.01
    if warmup > 0 and step < warmup:
        return peak * (step / float(warmup))
    return peak


@torch.no_grad()
def run_episode_2hop_cuebinder(
    stack, cfg, facts, fact_a, fact_b, mode, which, dirty_rng=None, use_strict=True,
):
    """2-hop using F85 CueBinder only (no V1 Hop1/Hop2 ADD-on binders)."""
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    scratch = ThoughtScratch(stack.device)
    for f in facts:
        if use_strict:
            mem.observe_strict(
                f["phrase"],
                stack.phrase_fp(f["phrase"].split()),
                fact=f,
                left_fp=stack.w(f["left"]),
            )
        else:
            mem.observe(f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f)
    mem.build_chain_links(facts)

    dirty = mode.startswith("dirty")
    partial_clean, ptype = hop1_cue(fact_a, which, "what_tail")
    partial = partial_clean
    if dirty:
        assert dirty_rng is not None
        if mode == "dirty_hard":
            partial, _ = dirty_token_hard(partial_clean, dirty_rng)
        else:
            dkind = DIRTY_KINDS[dirty_rng.randrange(len(DIRTY_KINDS))]
            partial, _ = dirty_token(partial_clean, dkind, dirty_rng)

    cue = partial_clean if mode == "clean_bind" else partial
    # CueBinder path (dim256 F85)
    bound = bind_full(stack, "what", cue if mode == "clean_bind" else partial)
    # hop1_rich_pick needs a binder arg only for disambig side-paths; pass None-safe dummy:
    # use CueBinder-shaped identity via bind_full already done; hop1_binder unused in rich_pick core.
    class _Dummy:
        def __call__(self, *a, **k):
            return a[0]

    key1, _, _, _, _ = hop1_rich_pick(
        stack, mem, scratch, bound, cue if mode != "clean_bind" else partial_clean,
        ptype, facts, cfg, partial_clean, _Dummy(), True, prefer_deep=False,
    )
    ok1 = key1 == fact_a["phrase"] and answer_right(key1) == fact_a["right"]
    bridge = answer_right(key1) if key1 else None
    ok2 = False
    if bridge is not None:
        bound2 = bind_full(stack, "where", bridge)
        key2, _, _, _, _ = hop2_via_chain(stack, mem, scratch, bound2, key1, bridge, facts, cfg)
        ok2 = key2 == fact_b["phrase"] and answer_right(key2) == fact_b["right"]
    return {"ok1": ok1, "ok2": ok2, "joint": bool(ok1 and ok2), "depth": scratch.depth()}


@torch.no_grad()
def eval_suite_2hop_cuebinder(stack, pairs, bank, cfg, rng, n_eps, modes, episode_k):
    stats = {m: {"ok1": 0, "ok2": 0, "joint": 0} for m in modes}
    for i in range(n_eps):
        facts, a, b = sample_episode(pairs, bank, episode_k, rng)
        which = "first" if (i % 2 == 0) else "second"
        ep_seed = HOLD_EVAL_SEED * 10007 + i * 17
        for m in modes:
            r = run_episode_2hop_cuebinder(
                stack, cfg, facts, a, b, m, which,
                dirty_rng=random.Random(ep_seed), use_strict=True,
            )
            stats[m]["ok1"] += int(r["ok1"])
            stats[m]["ok2"] += int(r["ok2"])
            stats[m]["joint"] += int(r["joint"])
    n = max(n_eps, 1)
    return {m: {k: stats[m][k] / n for k in ("ok1", "ok2", "joint")} for m in modes}


def _warmup_then_constant(step: int, peak: float, warmup: int) -> float:
    if step <= 0:
        return peak * 0.01
    if warmup > 0 and step < warmup:
        return peak * (step / float(warmup))
    return peak


class DirtyWhatRefine(nn.Module):
    """Residual refine of what_tail cue_fp for dirty-robust hop1 retrieve."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, cue_fp: torch.Tensor) -> torch.Tensor:
        if cue_fp.dim() == 1:
            cue_fp = cue_fp.unsqueeze(0)
        h = self.net(cue_fp)
        return F.normalize(cue_fp + self.res_scale * h, dim=-1).squeeze(0)


@torch.no_grad()
def run_episode_2hop_dirty_ft(
    stack, cfg, facts, fact_a, fact_b, mode, which,
    refine, hop2_binder, dirty_rng=None, use_strict=True,
):
    mem = SoftPhraseMemory(cfg.dim, cfg, stack.device)
    scratch = ThoughtScratch(stack.device)
    for f in facts:
        mem.observe_strict(
            f["phrase"], stack.phrase_fp(f["phrase"].split()), fact=f, left_fp=stack.w(f["left"]),
        )
    mem.build_chain_links(facts)

    dirty = mode.startswith("dirty")
    partial_clean, ptype = hop1_cue(fact_a, which, "what_tail")
    partial = partial_clean
    if dirty:
        assert dirty_rng is not None
        toks = partial_clean.split()
        if not toks:
            partial = partial_clean
        elif mode == "dirty_hard":
            last, _ = dirty_token_hard(toks[-1], dirty_rng)
            partial = " ".join(toks[:-1] + [last]) if len(toks) > 1 else last
        else:
            dkind = DIRTY_KINDS[dirty_rng.randrange(len(DIRTY_KINDS))]
            last, _ = dirty_token(toks[-1], dkind, dirty_rng)
            partial = " ".join(toks[:-1] + [last]) if len(toks) > 1 else last

    used = partial_clean if mode == "clean_bind" else partial
    cue_fp = stack.phrase_fp(used.split()) if mode == "clean_bind" else safe_phrase_fp(stack, used)
    # blend CueBinder prior with refine (clean leans binder; dirty leans refine)
    bound_cb = bind_full(stack, "what", used)
    bound_rf = refine(cue_fp)
    if mode == "clean_bind":
        bound = F.normalize(0.7 * bound_cb + 0.3 * bound_rf, dim=-1)
    else:
        bound = F.normalize(0.3 * bound_cb + 0.7 * bound_rf, dim=-1)

    class _Dummy:
        def __call__(self, *a, **k):
            return a[0]

    key1, _, _, _, _ = hop1_rich_pick(
        stack, mem, scratch, bound, used if dirty else partial_clean,
        ptype, facts, cfg, partial_clean, _Dummy(), True, prefer_deep=False,
    )
    ok1 = key1 == fact_a["phrase"] and answer_right(key1) == fact_a["right"]
    bridge = answer_right(key1) if key1 else None
    ok2 = False
    if bridge is not None:
        bound2 = hop2_binder(safe_phrase_fp(stack, bridge).unsqueeze(0)).squeeze(0)
        key2, _, _, _, _ = hop2_via_chain(stack, mem, scratch, bound2, key1, bridge, facts, cfg)
        ok2 = key2 == fact_b["phrase"] and answer_right(key2) == fact_b["right"]
    return {"ok1": ok1, "ok2": ok2, "joint": bool(ok1 and ok2)}


@torch.no_grad()
def eval_suite_2hop_dirty_ft(stack, pairs, bank, cfg, rng, n_eps, modes, episode_k, refine, hop2):
    stats = {m: {"ok1": 0, "ok2": 0, "joint": 0} for m in modes}
    for i in range(n_eps):
        facts, a, b = sample_episode(pairs, bank, episode_k, rng)
        which = "first" if (i % 2 == 0) else "second"
        ep_seed = HOLD_EVAL_SEED * 10007 + i * 17
        for m in modes:
            r = run_episode_2hop_dirty_ft(
                stack, cfg, facts, a, b, m, which, refine, hop2,
                dirty_rng=random.Random(ep_seed), use_strict=True,
            )
            stats[m]["ok1"] += int(r["ok1"])
            stats[m]["ok2"] += int(r["ok2"])
            stats[m]["joint"] += int(r["joint"])
    n = max(n_eps, 1)
    return {m: {k: stats[m][k] / n for k in ("ok1", "ok2", "joint")} for m in modes}


def _subsample(lines, n, seed):
    if not lines or n <= 0 or len(lines) <= n:
        return list(lines)
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    return [lines[i] for i in idx[:n]]


def main():
    cfg = Config()
    # Stage100 knobs
    cfg.c100_max_lines = getattr(cfg, "c100_max_lines", 100000)
    cfg.c100_ft_steps = getattr(cfg, "c100_ft_steps", 50000)
    cfg.c100_ft_lr = getattr(cfg, "c100_ft_lr", 1e-3)
    cfg.c100_eval_every = getattr(cfg, "c100_eval_every", 1000)
    cfg.c100_batch = getattr(cfg, "c100_batch", 8)
    cfg.c100_warmup = getattr(cfg, "c100_warmup", 200)
    cfg.c100_fat_frac = getattr(cfg, "c100_fat_frac", 0.75)
    cfg.c100_eval_seen_n = getattr(cfg, "c100_eval_seen_n", 600)
    cfg.c100_eval_story_n = getattr(cfg, "c100_eval_story_n", 400)
    cfg.c100_eval_rare_n = getattr(cfg, "c100_eval_rare_n", 120)
    cfg.c100_d_model = getattr(cfg, "c100_d_model", 256)
    cfg.c100_n_layers = getattr(cfg, "c100_n_layers", 2)
    cfg.c100_n_heads = getattr(cfg, "c100_n_heads", 4)
    cfg.c100_max_len = getattr(cfg, "c100_max_len", 16)
    cfg.c100_dropout = getattr(cfg, "c100_dropout", 0.1)
    cfg.c100_obj_partial = getattr(cfg, "c100_obj_partial", 0.25)
    cfg.c100_story_lift_pp = getattr(cfg, "c100_story_lift_pp", 0.08)
    cfg.c100_n_fat = getattr(cfg, "c100_n_fat", 200)
    cfg.c100_fat_copies = getattr(cfg, "c100_fat_copies", 40)
    cfg.c100_story_keep_frac = getattr(cfg, "c100_story_keep_frac", 0.50)
    cfg.c100_hold_frac = getattr(cfg, "c100_hold_frac", 0.15)
    # feed into build_ts_repeat_mix (uses c87_*)
    cfg.c87_n_fat = int(cfg.c100_n_fat)
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = int(cfg.c100_fat_copies)
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = float(cfg.c100_story_keep_frac)
    cfg.c87_hold_frac = float(cfg.c100_hold_frac)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Device:", device, f"({torch.cuda.get_device_name(device)})")
    else:
        print("Device: cpu")

    print("Stage 100: scale atom ~100k TinyStories (recipe98; 6h budget)")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    assert parent85.exists(), f"missing {parent85}"
    # load_foundation_85 uses PARENT85 global — patch via load path arg
    stack, f85 = load_foundation_85(device, cfg, path=parent85)
    print(f"  foundation {parent85.name} dim={f85.get('dim')}", flush=True)

    raw100 = ROOT / "data" / "_tinystories_raw_100k.txt"
    raw_fallback = ROOT / "data" / "_tinystories_raw_chunk.txt"
    corpus_path = ROOT / "data" / "external_tinystories_100k_85.txt"

    if not corpus_path.exists():
        raw = raw100 if raw100.exists() and raw100.stat().st_size > 1_000_000 else raw_fallback
        assert raw.exists(), f"need raw TinyStories at {raw100} or {raw_fallback}"
        print(f"  filtering {raw.name} -> {corpus_path.name} max_lines={cfg.c100_max_lines} ...", flush=True)
        phrases, meta_src = filter_tinystories_chunk(
            raw, corpus_path,
            max_lines=int(cfg.c100_max_lines),
            max_word_len=int(cfg.max_word_len),
            seed=272,
        )
        print(f"  filtered n={len(phrases)} meta={meta_src}", flush=True)
    else:
        phrases = [
            ln.strip() for ln in corpus_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        meta_src = {"source": corpus_path.name, "n": len(phrases), "reused": True}
        print(f"  reused corpus n={len(phrases)}", flush=True)

    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    print(f"  mix meta={ {k: meta[k] for k in meta if k != 'top_triple_freq'} }", flush=True)

    all_lines = train + hold_seen + hold_rare + hold_story
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    all_words = sorted({w for ln in all_lines for w in ln["words"]})
    print(f"  building fps V={len(all_words)} ...", flush=True)
    word_fps = F.normalize(torch.stack([stack.w(w).detach() for w in all_words], 0).to(device), dim=-1)
    surf = all_words
    stoi = {s: i for i, s in enumerate(surf)}

    # eval subsets (speed)
    ev_seen = _subsample(hold_seen, int(cfg.c100_eval_seen_n), 1001)
    ev_rare = _subsample(hold_rare, int(cfg.c100_eval_rare_n), 1002)
    ev_story = _subsample(hold_story, int(cfg.c100_eval_story_n), 1003)
    # final eval slightly larger
    fin_seen = _subsample(hold_seen, min(len(hold_seen), max(1200, int(cfg.c100_eval_seen_n))), 2001)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), max(120, int(cfg.c100_eval_rare_n))), 2002)
    fin_story = _subsample(hold_story, min(len(hold_story), max(800, int(cfg.c100_eval_story_n))), 2003)

    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    fat_pairs = lines_to_pairs(fat_lines, stoi) if fat_lines else []
    story_pairs = lines_to_pairs(story_lines, stoi) if story_lines else lines_to_pairs(train, stoi)
    if not fat_pairs:
        fat_pairs = story_pairs
    print(
        f"  train fat_pairs={len(fat_pairs)} story_pairs={len(story_pairs)} "
        f"eval_seen={len(ev_seen)} eval_story={len(ev_story)}",
        flush=True,
    )

    model = WordIdTransformer(
        n_vocab=len(surf),
        d_model=int(cfg.c100_d_model),
        n_heads=int(cfg.c100_n_heads),
        n_layers=int(cfg.c100_n_layers),
        max_len=int(cfg.c100_max_len),
        dropout=float(cfg.c100_dropout),
    ).to(device)
    model.init_from_fps(word_fps)
    n_params = sum(p.numel() for p in model.parameters())

    batch_sz = int(cfg.c100_batch)
    steps = int(cfg.c100_ft_steps)
    peak_lr = float(cfg.c100_ft_lr)
    warmup = int(cfg.c100_warmup)
    fat_frac = float(cfg.c100_fat_frac)
    every = int(cfg.c100_eval_every)
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    rng = random.Random(cfg.seed)

    print(
        f"  model params={n_params/1e6:.2f}M batch={batch_sz} steps={steps} "
        f"warmup={warmup} lr={peak_lr} fat_frac={fat_frac} eval_every={every}",
        flush=True,
    )

    def _set_lr(step: int) -> float:
        lr = _warmup_then_constant(step, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    def _sample_batch():
        n_fat = max(1, int(round(batch_sz * fat_frac))) if fat_pairs else 0
        n_fat = min(n_fat, batch_sz)
        n_st = batch_sz - n_fat
        ex = [rng.choice(fat_pairs) for _ in range(n_fat)]
        ex += [rng.choice(story_pairs) for _ in range(n_st)]
        rng.shuffle(ex)
        return ex

    def _eval_hold(lines):
        return eval_id_capacity_suite(model, lines, train, surf, stoi, device) if lines else None

    best = {
        "step": 0, "obj": 0.0, "rel": 0.0, "story_all": 0.0,
        "seen": None, "rare": None, "story": None,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def _snap(step: int):
        model.eval()
        seen = _eval_hold(ev_seen)
        rare = _eval_hold(ev_rare)
        story = _eval_hold(ev_story)
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({
            "step": step, "obj_seen": obj_s, "rel_seen": rel_s,
            "story_all": st_all, "obj_rare": rare["obj"]["hit1"],
            "obj_story": story["obj"]["hit1"],
            "head_seen": seen["freq"].get("head21p", {}).get("model", 0.0),
        })
        # prefer story lift while keeping joint SEEN
        key = (1 if rel_s >= 0.50 else 0, st_all, obj_s, rel_s)
        bkey = (
            1 if best["rel"] >= 0.50 else 0,
            best["story_all"], best["obj"], best["rel"],
        )
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, rel=rel_s, story_all=st_all,
                seen=seen, rare=rare, story=story,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        print(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% | "
            f"STORY ALL={st_all*100:.1f}% obj={story['obj']['hit1']*100:.1f}% | "
            f"RARE obj={rare['obj']['hit1']*100:.1f}%",
            flush=True,
        )
        model.train()

    print("\n=== FT scale ===", flush=True)
    _snap(0)
    model.train()
    for step in range(1, steps + 1):
        _set_lr(step)
        packed = collate_word_id_batch(_sample_batch(), stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, pad_mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % every == 0 or step == steps:
            _snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    print("\n=== Final eval (larger subsample) ===", flush=True)
    f_seen = _eval_hold(fin_seen)
    f_rare = _eval_hold(fin_rare)
    f_story = _eval_hold(fin_story)
    obj_s = f_seen["obj"]["hit1"]
    rel_s = f_seen["roles"].get("rel", {}).get("hit1", 0.0)
    all_s = f_seen["roles"].get("ALL", {}).get("hit1", 0.0)
    head_s = f_seen["freq"].get("head21p", {}).get("model", 0.0)
    st_all = f_story["roles"].get("ALL", {}).get("hit1", 0.0)
    st_obj = f_story["obj"]["hit1"]
    rare_obj = f_rare["obj"]["hit1"]

    ref98 = {"obj": 0.413, "rel": 0.960, "story_all": 0.092, "rare_obj": 0.133}
    lift_story = st_all - ref98["story_all"]

    if (
        obj_s >= float(cfg.c100_obj_partial)
        and rel_s >= 0.50
        and lift_story >= float(cfg.c100_story_lift_pp)
    ):
        verdict = "PASS"
    elif obj_s >= float(cfg.c100_obj_partial) and rel_s >= 0.50 and lift_story >= 0.03:
        verdict = "PARTIAL"
    elif obj_s + 0.02 >= ref98["obj"] and rel_s >= 0.50 and abs(lift_story) < 0.03:
        verdict = "CEILING"
    elif obj_s >= float(cfg.c100_obj_partial) and rel_s >= 0.50:
        verdict = "PARITY_ATOM"
    else:
        verdict = "FAIL"

    lines = [
        "SOTE Stage 100 — scale atom ~100k TinyStories (recipe98 hparams)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"device: {device}",
        f"foundation: {parent85.name} dim={cfg.dim}",
        f"corpus: {corpus_path.name} n_phrases={len(phrases)} meta={meta_src}",
        f"mix: { {k: meta[k] for k in meta if k != 'top_triple_freq'} }",
        f"V={len(surf)} params={n_params}",
        f"train: batch={batch_sz} steps={steps} warmup={warmup} lr={peak_lr} "
        f"fat_frac={fat_frac} emb=x1 (Stage98 recipe)",
        "NO overwrite of F85 dual-channel FROZEN. NO soft@5.",
        "",
        "=== Final ===",
        f"  best_step(during)={best['step']}",
        f"  SEEN  obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% ALL={all_s*100:.1f}% head={head_s*100:.1f}%",
        f"  RARE  obj={rare_obj*100:.1f}%",
        f"  STORY ALL={st_all*100:.1f}% obj={st_obj*100:.1f}% "
        f"rel={f_story['roles'].get('rel',{}).get('hit1',0)*100:.1f}%",
        "",
        "=== vs Stage98 mini ===",
        f"  SEEN obj {obj_s*100:.1f}% vs {ref98['obj']*100:.1f}%",
        f"  SEEN rel {rel_s*100:.1f}% vs {ref98['rel']*100:.1f}%",
        f"  STORY ALL {st_all*100:.1f}% vs {ref98['story_all']*100:.1f}% (lift {lift_story*100:+.1f}pp)",
        f"  RARE obj {rare_obj*100:.1f}% vs {ref98['rare_obj']*100:.1f}%",
        "",
        "=== Adequacy ===",
        "  Ask: was Stage98 STORY soft-ceiling from mini data?",
        f"  PASS if STORY ALL lift >= {cfg.c100_story_lift_pp*100:.0f}pp with SEEN held.",
        "  CEILING if SEEN held and STORY flat (|lift|<3pp).",
        "",
        f"=== Verdict: {verdict} ===",
    ]
    # freq rows
    lines.append("  SEEN freq model/bigram:")
    for k in ("head21p", "mid6_20", "low2_5", "rare1"):
        if k not in f_seen["freq"]:
            continue
        v = f_seen["freq"][k]
        lines.append(
            f"    {k:8s} n={v['n']:5d}  model={v['model']*100:5.1f}%  bigram={v['bigram']*100:5.1f}%"
        )
    report = "\n".join(lines) + "\n"
    print()
    print(report)
    OUT_TXT.write_text(report, encoding="utf-8")
    (RES / f"stage100_scale_100k_{verdict}.txt").write_text(report, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "stage": 100,
        "verdict": verdict,
        "best_step": best["step"],
        "n_phrases": len(phrases),
        "V": len(surf),
        "meta": {k: meta[k] for k in meta if k != "top_triple_freq"},
        "meta_src": meta_src,
        "seen": {"obj": obj_s, "rel": rel_s, "all": all_s, "head": head_s, "freq": f_seen["freq"]},
        "rare": {"obj": rare_obj},
        "story": {"all": st_all, "obj": st_obj},
        "lift_story_pp": lift_story,
        "ref98": ref98,
        "curve": curve,
        "cfg": {
            "c100_max_lines": cfg.c100_max_lines,
            "c100_ft_steps": cfg.c100_ft_steps,
            "c100_batch": cfg.c100_batch,
            "c100_ft_lr": cfg.c100_ft_lr,
            "c100_warmup": cfg.c100_warmup,
            "c100_fat_frac": cfg.c100_fat_frac,
        },
    }, indent=2, default=str), encoding="utf-8")
    torch.save({
        "stage": 100,
        "verdict": verdict,
        "word_tf": best["state"],
        "surfaces": surf,
        "word_fps": word_fps.detach().cpu(),
        "cfg": asdict(cfg),
        "meta": {k: meta[k] for k in meta if k != "top_triple_freq"},
        "foundation": str(parent85),
        "corpus": str(corpus_path),
    }, OUT_CKPT)
    print(f"Saved: {OUT_CKPT}")


if __name__ == "__main__":
    main()
