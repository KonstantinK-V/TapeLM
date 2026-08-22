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
v0 = v3('data/_wikitext103_train.txt')
v1 = 254

def main() -> v2:
    v4 = v21.v9(v1)
    v5 = v0.v27('r', encoding='utf-8', errors='ignore').v10(8000000)
    v6 = v11(v28.v22((v35.v34(1) for v35 in v43.v38(v5) if v36(v35.v34(1)) >= 5)))
    v4.v12(v6)
    v13(f'values_pool={v36(v6)} e.g. {v6[:8]}')
    for v7 in ('data/_stage254_med.txt', 'data/_stage254_news.txt', 'data/_tinystories_raw_100k.txt'):
        v14 = v3(v7)
        if not v14.v29():
            continue
        v15 = v14.v30(encoding='utf-8', errors='ignore')[:4000000]
        v16 = []
        for v17 in v23(200):
            v24 = v6[v4.v37(v36(v6))]
            v16.v31(v36(v44.v39(f'\\b{v44.v45(v24)}\\b', v15)))
        v18 = v32.v25(v16)
        v19 = v18.v33() * 26
        v13(f'{v14.v7}: chars={v36(v15)} | occurrences per planted value: mean={v18.v33():.1f} median={v32.v40(v18):.0f} p90={v32.v41(v18, 90):.0f} max={v18.v42()} | >0 in {v32.v33(v18 > 0):.0%} of draws | expected rewrites/domain ~{v19:.0f}')
    return 0
if v8 == '__main__':
    raise v20(v26())