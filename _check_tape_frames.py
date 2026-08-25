"""Core frame-tape checks: exact counting, distinct fillers, line boundaries."""
from __future__ import annotations

from pathlib import Path

import _tape_frames as T

SRC = Path("_tape_frames.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "owner[i - w] != owner[i] or owner[i + w] != owner[i]" not in src:
        fails.append("1. frame may cross a line boundary")

    cross = ["p a", "q b", "p c", "q d"]
    keep, toks, _owner = T.frame_keep(cross, frame_max=1, min_fillers=2)
    cross_addrs = {(left, right) for (_w, left, right), _pos in keep}
    if (("p",), ("q",)) in cross_addrs:
        fails.append("1. cross-line p|q address survived")

    local = ["p a q", "p c q"]
    keep, toks, _owner = T.frame_keep(local, frame_max=3, min_fillers=2)
    rows = [
        {toks[i] for i in pos}
        for (_w, left, right), pos in keep
        if left == ("p",) and right == ("q",)
    ]
    if not rows or rows[0] != {"a", "c"}:
        fails.append("1. valid line-local recurrent frame disappeared")

    if "len({toks[i] for i in pos}) >= min_fillers" not in src:
        fails.append("1. min_fillers is not distinct-value count")
    return fails


MUTANTS = (
    (
        "cross-line glue",
        "            if owner[i - w] != owner[i] or owner[i + w] != owner[i]:",
        "            if False:",
        "1.",
    ),
    (
        "row count instead of distinct fillers",
        "if len({toks[i] for i in pos}) >= min_fillers",
        "if len(pos) >= min_fillers",
        "1.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props(src)
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {src.count(old)}")
            continue
        saved = dict(T.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), T.__dict__)
            got = props(mutated)
        except Exception as exc:
            got = [f"{tag}. mutant raised {type(exc).__name__}"]
        finally:
            T.__dict__.clear()
            T.__dict__.update(saved)
        if not any(item.startswith(tag) for item in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for item in fails:
        print("FAIL " + item)
    print(
        f"{len(fails)} failures" if fails else
        f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
