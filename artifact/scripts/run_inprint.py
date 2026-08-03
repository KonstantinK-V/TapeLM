#!/usr/bin/env python3
"""
Inprint v0.1 — unified product entry (TapeLM variant A codebase).

  python artifact/scripts/run_inprint.py              # same as demo
  python artifact/scripts/run_inprint.py demo         # memory trunk demo
  python artifact/scripts/run_inprint.py demo --all   # + stage 196 scorecard
  python artifact/scripts/run_inprint.py map          # curated verdicts, no GPU
  python artifact/scripts/run_inprint.py check        # env + manifest
  python artifact/scripts/run_inprint.py verify       # v0.1 evidence files present
  python artifact/scripts/run_inprint.py smoke-glue   # stage 256 smoke (~2 min GPU)
  python artifact/scripts/run_inprint.py ingest --schedule wiki:2,med:2 --run-tag my_run
  python artifact/scripts/run_inprint.py ingest --resume --run-tag my_run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
MANIFEST = REPO / "artifact" / "inprint" / "manifest.json"


def run_py(script: str | Path, extra: list[str] | None = None) -> int:
    p = script if isinstance(script, Path) else SCRIPTS / script
    cmd = [sys.executable, str(p), *(extra or [])]
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _say(m: str) -> None:
    try:
        print(m)
    except UnicodeEncodeError:
        print(m.encode("ascii", "replace").decode("ascii"))


def cmd_check(_: argparse.Namespace) -> int:
    m = load_manifest()
    _say(f"Inprint {m['version']} — {m['tagline']}\n")
    _say("Layers:")
    for layer in m["layers"]:
        st = ", ".join(layer["stages"][:6])
        if len(layer["stages"]) > 6:
            st += ", ..."
        _say(f"  [{layer['id']}] {layer['title']}  ({st})")
    _say("")
    rc = run_py("check_env.py")
    if rc != 0:
        return rc
    p1 = REPO / "checkpoints" / "stage191_p1_curve.pt"
    if not p1.exists():
        print("\nWeights missing — python artifact/scripts/download_checkpoints.py")
        return 2
    print(f"\nOK  P1 weights present")
    return 0


def cmd_verify(_: argparse.Namespace) -> int:
    m = load_manifest()
    missing = []
    for req in m.get("evidence_required", []):
        rel = req["file"]
        if not (REPO / rel).exists():
            missing.append(f"stage {req['stage']}: {rel}")
    if missing:
        print("Inprint v0.1 evidence — missing:\n")
        for x in missing:
            print(f"  - {x}")
        print("\nRun: python artifact/scripts/sync_decisions.py")
        return 1
    print(f"Inprint {m['version']}: all {len(m['evidence_required'])} evidence files present.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.all:
        extra.append("--all")
    if args.smoke:
        extra.append("--smoke")
    if args.skip_cross_domain:
        extra.append("--skip-cross-domain")
    if args.skip_glue:
        extra.append("--skip-glue")
    m = load_manifest()
    print(f"Inprint {m['version']} demo — memory trunk + slot-bias glue\n")
    return run_py("run_memory_demo.py", extra if extra else None)


def cmd_ingest(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(REPO / "_stage255_stream_ingest.py")]
    if args.smoke:
        cmd.append("--smoke")
    if args.schedule:
        cmd.extend(["--schedule", args.schedule])
    if args.run_tag:
        cmd.extend(["--run-tag", args.run_tag])
    if args.resume:
        cmd.append("--resume")
    if args.no_query_train:
        cmd.append("--no-query-train")
    cmd.extend(["--chunk-lines", str(args.chunk_lines)])
    cmd.extend(["--epochs-per-chunk", str(args.epochs_per_chunk)])
    cmd.extend(["--replay-frac", str(args.replay_frac)])
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Inprint v0.1 product")
    sub = ap.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="Memory trunk demo (default)")
    p_demo.add_argument("--all", action="store_true")
    p_demo.add_argument("--smoke", action="store_true")
    p_demo.add_argument("--skip-cross-domain", action="store_true")
    p_demo.add_argument("--skip-glue", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_ing = sub.add_parser("ingest", help="Bounded stream ingest (stage 255)")
    p_ing.add_argument("--smoke", action="store_true")
    p_ing.add_argument("--schedule", default="wiki:2,med:2")
    p_ing.add_argument("--run-tag", default="inprint_ingest")
    p_ing.add_argument("--resume", action="store_true")
    p_ing.add_argument("--no-query-train", action="store_true")
    p_ing.add_argument("--chunk-lines", type=int, default=25000)
    p_ing.add_argument("--epochs-per-chunk", type=float, default=1.0)
    p_ing.add_argument("--replay-frac", type=float, default=0.2)
    p_ing.set_defaults(func=cmd_ingest)

    sub.add_parser("map", help="Curated stage map (no GPU)").set_defaults(
        func=lambda a: run_py("show_map.py")
    )
    sub.add_parser("check", help="Environment + manifest summary").set_defaults(func=cmd_check)
    sub.add_parser("verify", help="Check v0.1 decision artifacts").set_defaults(func=cmd_verify)
    sub.add_parser("smoke-glue", help="Stage 256 glue smoke").set_defaults(
        func=lambda a: run_py(REPO / "_stage256_slot_bias_decode.py", ["--smoke"])
    )
    sub.add_parser("smoke-stream", help="Stage 255 stream smoke").set_defaults(
        func=lambda a: run_py(REPO / "_stage255_stream_ingest.py", ["--smoke", "--run-tag", "inprint_smoke"])
    )

    args = ap.parse_args()
    if args.cmd is None:
        args.cmd = "demo"
        args.all = False
        args.smoke = False
        args.skip_cross_domain = False
        args.skip_glue = False
        return cmd_demo(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
