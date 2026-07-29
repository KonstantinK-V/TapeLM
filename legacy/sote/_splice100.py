"""Splice Stage 100: scale atom to ~100k windows."""
import re
from pathlib import Path

train_path = Path("train.py")
main_path = Path("_stage100_main.py")

head = '''"""
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
'''

t = train_path.read_text(encoding="utf-8")
a = t.find('"""')
b = t.find('"""', a + 3)
body = t[b + 3 :]
body = re.sub(r'OUT_CKPT = CKPT / "[^"]+"', 'OUT_CKPT = CKPT / "stage100_scale_100k.pt"', body, count=1)
body = re.sub(r'OUT_TXT = RES / "[^"]+"', 'OUT_TXT = RES / "stage100_scale_100k_report.txt"', body, count=1)
body = re.sub(r'OUT_JSON = RES / "[^"]+"', 'OUT_JSON = RES / "stage100_scale_100k_metrics.json"', body, count=1)

knob_end = "# --------------- frozen modules ---------------"
if "c100_ft_steps" not in body:
    insert = """    # Stage100 scale 100k
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

"""
    body = body.replace(knob_end, insert + knob_end)

# Retarget PARENT85 to FROZEN if present
if 'PARENT85 = CKPT / "stage85_foundation.pt"' in body:
    body = body.replace(
        'PARENT85 = CKPT / "stage85_foundation.pt"',
        'PARENT85 = CKPT / "stage85_foundation_FROZEN.pt" if (CKPT / "stage85_foundation_FROZEN.pt").exists() else CKPT / "stage85_foundation.pt"',
        1,
    )

j = body.rfind("\ndef main():\n")
main = main_path.read_text(encoding="utf-8")
train_path.write_text(head + body[:j] + "\n" + main, encoding="utf-8")
print("ok")
