"""Check of 538: the mark is a role key and a rank, and nothing else travels."""
from __future__ import annotations

from pathlib import Path

import _audit538_role as M

SRC = Path("_audit538_role.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit532_pool import slice_graph" not in src:
        f.append("1. 532 slice_graph reuse missing")
    if "from _audit534_mark import offer" not in src:
        f.append("1. 534 offer reuse missing")
    if "from _audit527_learn import v1_nodes" not in src:
        f.append("1. 527 v1_nodes (the 511 arm) missing")
    if "return (band, peaked, width)" not in src:
        f.append("1. the role key carries a word (band/peaked/width only)")
    if "return (v, band, peaked, width)" in src:
        f.append("1. word leaked into role key")
    if "mark_t = [rec[r]] if live else []" not in src:
        f.append("1. rank resolved on the TEST window's own rec missing")
    if "Q[key][r] += 1" not in src:
        f.append("1. the taught value is a rank, not a node")
    if "mark_r = [rec[j]] if j < len(rec) else []" not in src:
        f.append("1. random-rank control missing")
    if "train_g, test_g = graphs[:n_tr], graphs[n_tr:]" not in src:
        f.append("1. 70/30 window split missing")
    if "if rec[r] in held and rec[r] != maj:" not in src:
        f.append("1. taught rank on residual held (not maj) missing")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    if "key_seen = p_key" in src or "p_key = 1" in src:
        f.append("3. key_seen faked")
    if "paired_d += d_t - d_r" not in src:
        f.append("1. paired pay on co-fired trials missing")
    if "void = n_keys < 3 or n_row < 20 or fire_t < 40" not in src:
        f.append("2. VOID on fewer than 3 role keys missing")
    if "paired_n >= 40 and paired_d > 0" not in src:
        f.append("2. GATE on paired_d > 0 with paired_n >= 40 missing")
    if "p_key >= 0.9" not in src:
        f.append("2. GATE on key_seen >= 0.9 missing")
    if "b['sum_d1']/max(b['n'],1)" not in src:
        f.append("2. baskets printed as means, not sums, missing")
    return f


MUTANTS = (
    ("word back in the key",
     "    return (band, peaked, width)",
     "    return (v, band, peaked, width)",
     "1."),
    ("no random control",
     "            mark_r = [rec[j]] if j < len(rec) else []",
     "            mark_r = []",
     "1."),
    ("two-row table allowed",
     "    void = n_keys < 3 or n_row < 20 or fire_t < 40",
     "    void = n_row < 20 or fire_t < 40",
     "2."),
    ("gate on unpaired shot ratio",
     "    gate = (not void) and p_key >= 0.9 and paired_n >= 40 and paired_d > 0",
     "    gate = (not void) and p_key >= 0.9 and shot_t >= 2.0 * shot_r",
     "2."),
    ("no paired count",
     "            if ft and fr:\n                paired_n += 1\n                paired_d += d_t - d_r",
     "            if ft and fr:\n                pass",
     "1."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {n} times")
            continue
        saved = dict(M.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), M.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            M.__dict__.clear()
            M.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
