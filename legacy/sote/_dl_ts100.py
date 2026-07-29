"""Download TinyStories text for Stage100 (~enough for 100k windows)."""
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "_tinystories_raw_100k.txt"
OUT.parent.mkdir(exist_ok=True)

# Stream GPT4 split; stop when char budget reached (~enough for 100k short windows)
CHAR_BUDGET = 25_000_000  # ~25MB text
N_PRINT = 5000

print("loading roneneldan/TinyStories (streaming)...", flush=True)
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

n_docs = 0
n_chars = 0
with OUT.open("w", encoding="utf-8") as f:
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        f.write(text.replace("\n", " "))
        f.write("\n\n")
        n_docs += 1
        n_chars += len(text) + 2
        if n_docs % N_PRINT == 0:
            print(f"  docs={n_docs} chars={n_chars}", flush=True)
        if n_chars >= CHAR_BUDGET:
            break

print(f"saved {OUT} docs={n_docs} chars={n_chars}", flush=True)
