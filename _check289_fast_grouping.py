"""Ten-second check on the real machine: the fast grouping must equal the original, exactly.

The restructure in `_tape_speed.install_fast_fp_addresses` is a control-flow change, and the control
flow was proved equal offline over three thousand tie-heavy cases. What could NOT be checked
without torch is the numerics: a batched matmul may accumulate a dot product in a different
order than the per-group one, so a similarity sitting exactly on tau could in principle fall
the other way. This runs both versions on real tensors and compares the grouping itself.

  python _check289_fast_grouping.py
"""
from __future__ import annotations

import random
import sys

import torch
import torch.nn.functional as F

import _stage279_write_decision as s279
import _tape_speed


class FakeBank:
    """No model, no corpus - just deterministic vectors, so this runs in seconds."""

    def __init__(self, d=256, seed=0):
        self.d, self.g = d, torch.Generator().manual_seed(seed)
        self.memo: dict = {}

    def _vec(self, s):
        if s not in self.memo:
            h = abs(hash(s)) % (2 ** 31)
            self.memo[s] = torch.randn(self.d, generator=torch.Generator().manual_seed(h))
        return self.memo[s]

    def ctx_fp(self, text, exclude=None):
        return F.normalize(self._vec(f"ctx:{text}|{exclude}"), dim=-1)

    def fp(self, words):
        return F.normalize(self._vec("fp:" + "|".join(words)), dim=-1).unsqueeze(0)


def main() -> int:
    rnd = random.Random(11)
    bank = FakeBank()
    anchors = [f"anchor {i}" for i in range(40)]
    rels = ["was born in", "played for", "is located in"]
    vals = [f"Value{i}" for i in range(30)]
    worst = 0
    for trial in range(8):
        # cluster-friendly: repeated anchors and repeated context words, so groups really form
        asserts = []
        for _ in range(rnd.randint(50, 300)):
            a = rnd.choice(anchors)
            r = rnd.choice(rels)
            v = rnd.choice(vals)
            asserts.append({"address": f"{a}|{r}", "value": v,
                            "source": "x", "ctx": f"{a} {r} {v} extra words here"})
        for tau in (0.5, 0.9):
            for key in ("two", "set"):
                un = _tape_speed.install_fast_fp_addresses(s279)
                fast_out, fast_addrs = s279.fp_addresses(
                    [dict(a) for a in asserts], bank, tau, 1, 2, addr_key=key)
                un()
                raw_out, raw_addrs = s279.fp_addresses(
                    [dict(a) for a in asserts], bank, tau, 1, 2, addr_key=key)
                if fast_addrs != raw_addrs or fast_out != raw_out:
                    print(f"MISMATCH trial={trial} tau={tau} addr_key={key}: "
                          f"{len(fast_addrs)} vs {len(raw_addrs)} addresses")
                    worst += 1
                else:
                    print(f"  ok trial={trial} tau={tau} addr_key={key}: "
                          f"{len(raw_addrs)} addresses, {len(raw_out)} assertions")
    print("IDENTICAL" if worst == 0 else f"{worst} MISMATCHES - do not use the fast path")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
