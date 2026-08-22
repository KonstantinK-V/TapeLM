"""Audit: how much natural text does stage-254's global masking destroy?

inject_and_mask() calls masked.replace(f["value"], f"RegionBeta{i}") on the whole
domain corpus. f["value"] is a REAL wikitext entity, so every occurrence anywhere in
the domain gets rewritten, not just the one inside the planted fact sentence.
"""
from __future__ import annotations
import random
import re
from pathlib import Path
import numpy as np
from _stage194_fp_fact_memory import ENT_RE
WIKI = Path('data/_wikitext103_train.txt')
SEED = 254

def main() -> int:
    rng = random.Random(SEED)
    wtext = WIKI.open('r', encoding='utf-8', errors='ignore').read(8000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    print(f'values_pool={len(values_pool)} e.g. {values_pool[:8]}')
    for name in ('data/_stage254_med.txt', 'data/_stage254_news.txt', 'data/_tinystories_raw_100k.txt'):
        p = Path(name)
        if not p.exists():
            continue
        text = p.read_text(encoding='utf-8', errors='ignore')[:4000000]
        counts = []
        for _ in range(200):
            v = values_pool[rng.randrange(len(values_pool))]
            counts.append(len(re.findall(f'\\b{re.escape(v)}\\b', text)))
        c = np.asarray(counts)
        per_domain = c.mean() * 26
        print(f'{p.name}: chars={len(text)} | occurrences per planted value: mean={c.mean():.1f} median={np.median(c):.0f} p90={np.percentile(c, 90):.0f} max={c.max()} | >0 in {np.mean(c > 0):.0%} of draws | expected rewrites/domain ~{per_domain:.0f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())