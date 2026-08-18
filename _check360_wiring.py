from pathlib import Path
src = Path("_stage289_derivation.py").read_text(encoding="utf-8")
ok = True
checks = [
    ('ap.add_argument("--min-fillers", type=int, default=2' in src, "argparse default 2"),
    ("MIN_FILLERS = args.min_fillers" in src, "assigned from args"),
    ("lines, FRAME_MAX, MIN_FILLERS," in src, "threaded to frame_assertions"),
    ("lines, FRAME_MAX, 2," not in src, "hard-coded 2 gone from frame_assertions"),
    ('"min_fillers": MIN_FILLERS' in src, "written into the report"),
]
for cond, msg in checks:
    print(("OK  " if cond else "FAIL") + " " + msg)
    ok = ok and cond
print()
print("360 WIRING OK" if ok else "360 WIRING BROKEN")
raise SystemExit(0 if ok else 1)
