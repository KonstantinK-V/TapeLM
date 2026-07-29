"""How spaces look on the frozen Stage170 pen curve."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _stage170_curve_dynamics as s170

ckpt = torch.load("checkpoints/stage170_curve.pt", map_location="cpu", weights_only=False)
stoi, itos = ckpt["stoi"], ckpt["itos"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = s170.CurveModel(len(itos)).to(device)
model.load_state_dict(ckpt["model"])
pen = model.pen.eval()

text = Path("data/_wikitext103_train.txt").read_text(encoding="utf-8", errors="ignore")[50_000:52_000]
ids = torch.tensor([[stoi.get(c, 0) for c in text]], device=device)
with torch.no_grad():
    z = pen(ids)[0].cpu()
delta = z[1:] - z[:-1]
speed = delta.norm(dim=-1).numpy()
chars = list(text[1:])


def stats(mask: np.ndarray, name: str) -> None:
    s = speed[mask]
    if s.size < 5:
        print(f"{name}: n={s.size} (skip)")
        return
    print(
        f"{name}: n={s.size} speed_mean={s.mean():.4f} med={np.median(s):.4f} "
        f"p10={np.percentile(s, 10):.4f} p90={np.percentile(s, 90):.4f}"
    )


is_space = np.array([c == " " for c in chars])
is_letter = np.array([c.isalpha() for c in chars])
is_punct = np.array([c in ".,!?;:\"'()[]{}" for c in chars])
is_nl = np.array([c == "\n" for c in chars])

print("space in stoi:", " " in stoi, "id=", stoi.get(" "))
for ch in (" ", "a", "A", ".", "\n", "e"):
    if ch in stoi:
        print(f"emb|{ch!r}| norm={float(pen.emb.weight[stoi[ch]].norm()):.4f}")

stats(is_space, "SPACE")
stats(is_letter, "LETTER")
stats(is_punct, "PUNCT")
stats(is_nl, "NEWLINE")
stats(~is_space, "NONSPACE")

idx = np.where(is_space)[0]
idx = idx[(idx > 0) & (idx < len(speed) - 1)]
print(
    "speed at space / prev / next:",
    float(speed[idx].mean()),
    float(speed[idx - 1].mean()),
    float(speed[idx + 1].mean()),
)

# mean direction of space deltas vs letter deltas (cosine between class means)
ms = delta[torch.tensor(is_space)].mean(0)
ml = delta[torch.tensor(is_letter)].mean(0)
print("cos(mean_delta_space, mean_delta_letter)=", float(F.cosine_similarity(ms, ml, dim=0)))
