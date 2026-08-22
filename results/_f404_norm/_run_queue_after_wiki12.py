"""
After wiki:12 (run-tag wiki12) finishes, queue overnight stream tests on GPU.

  python _run_queue_after_wiki12.py
  python _run_queue_after_wiki12.py --skip-ablation
  python _run_queue_after_wiki12.py --skip-scale-ablation
  python _run_queue_after_wiki12.py --scale-ablation-only

1. Poll until results/stage255_decision_wiki12.json has 12 chunks (or log tail).
2. Multi-domain stream: wiki:6,med:3,news:3 -> run-tag stream_wmn_v1
3. Ablation (small, in-domain smoke): wiki:2,med:2 with --no-query-train -> run-tag ablate_no_wq_4ch
4. Ablation (scale-matched vs wiki12): wiki:12 with --no-query-train -> run-tag ablate_no_wq_wiki12
   Same schedule/chunk-lines as the real wiki12 run, so tape_slots line up chunk-for-chunk and
   recall_top1_frozen_vs_bank can be compared directly against stage255_decision_wiki12.json at
   equal bank size — step 3 alone only reaches ~1.2k slots, wiki12 reaches ~2.9k, and the frozen
   recall regression the old (pre-fix) wiki12 run showed was never confirmed at that scale under
   the current eval methodology (see docs/STAGES.md stage 255 note).
5. Stage 257 two-hop composition (full GPU run, after step 4 unless --skip-stage257).

Logs:
  results/_run_queue_after_wiki12.log
  results/_stage255_stream_wmn_full.out
  results/_stage255_ablate_no_wq_full.out
  results/_stage255_ablate_no_wq_wiki12_full.out
  results/_stage257_full.out
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
LOG_W12 = ROOT / 'results' / '_stage255_wiki12_full.out'
DEC_W12 = ROOT / 'results' / 'stage255_decision_wiki12.json'
OUT_WMN = ROOT / 'results' / '_stage255_stream_wmn_full.out'
OUT_ABL = ROOT / 'results' / '_stage255_ablate_no_wq_full.out'
OUT_ABL_W12 = ROOT / 'results' / '_stage255_ablate_no_wq_wiki12_full.out'
OUT_257 = ROOT / 'results' / '_stage257_full.out'
OUT_257_SMOKE = ROOT / 'results' / '_stage257_smoke.out'
OUT_258 = ROOT / 'results' / '_stage258_smoke.out'
OUT_259 = ROOT / 'results' / '_stage259_smoke.out'
DEC_257 = ROOT / 'results' / 'stage257_decision.json'
DEC_259 = ROOT / 'results' / 'stage259_decision.json'
QLOG = ROOT / 'results' / '_run_queue_after_wiki12.log'
POLL_S = 180
WMN_CHUNKS = 10
ABL_CHUNKS = 4
ABL_W12_CHUNKS = 12
STREAM_COMMON = ['--chunk-lines', '25000', '--epochs-per-chunk', '1.0', '--replay-frac', '0.2', '--ckpt-every', '2']

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    try:
        with QLOG.open('a', encoding='utf-8') as f:
            f.write(line)
    except OSError:
        pass

def wiki12_done() -> bool:
    if DEC_W12.exists():
        try:
            d = json.loads(DEC_W12.read_text(encoding='utf-8'))
            if d.get('run_tag') == 'wiki12' and d.get('summary', {}).get('chunks', 0) >= ABL_W12_CHUNKS:
                return True
            if d.get('summary', {}).get('chunks', 0) >= ABL_W12_CHUNKS:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    if LOG_W12.exists():
        tail = LOG_W12.read_text(encoding='utf-8', errors='ignore')[-15000:]
        if 'schedule exhausted' in tail and 'STREAM_INGEST' in tail:
            return True
    return False

def ablate_wiki12_done() -> bool:
    if stream_done('stage255_decision_ablate_no_wq_wiki12.json', ABL_W12_CHUNKS):
        return True
    if OUT_ABL_W12.exists():
        tail = OUT_ABL_W12.read_text(encoding='utf-8', errors='ignore')[-15000:]
        if 'schedule exhausted' in tail and 'STREAM_INGEST' in tail:
            return True
    return False

def stage257_done() -> bool:
    if not DEC_257.exists():
        return False
    try:
        d = json.loads(DEC_257.read_text(encoding='utf-8'))
        return d.get('stage') == 257 and d.get('overall') in ('FP_COMPOSE_OK', 'FP_COMPOSE_PARTIAL', 'FP_COMPOSE_NO')
    except (json.JSONDecodeError, OSError):
        return False

def stage257_cmd() -> list[str]:
    return [sys.executable, str(ROOT / '_stage257_fp_compose.py')]

def stage258_smoke_cmd() -> list[str]:
    return [sys.executable, str(ROOT / '_stage258_semantic_query.py'), '--smoke']

def stage259_smoke_cmd() -> list[str]:
    return [sys.executable, str(ROOT / '_stage259_hot_swap.py'), '--smoke']

def stage257_smoke_cmd() -> list[str]:
    return [sys.executable, str(ROOT / '_stage257_fp_compose.py'), '--smoke']

def stage259_done() -> bool:
    if not DEC_259.exists():
        return False
    try:
        d = json.loads(DEC_259.read_text(encoding='utf-8'))
        return d.get('stage') == 259 and d.get('overall') in ('HOT_SWAP_OK', 'HOT_SWAP_NO')
    except (json.JSONDecodeError, OSError):
        return False

def run_stage257(*, force: bool=False) -> int:
    if not force and stage257_done():
        log('stage257 already complete — skip')
        return 0
    if not force and OUT_257.exists() and (time.time() - OUT_257.stat().st_mtime < 7200):
        log('skip stage257: _stage257_full.out updated in last 2h (running or just finished?)')
        return 0
    return run_stage(stage257_cmd(), OUT_257, 'stage257_fp_compose', ignore_recent=force)

def stream_done(dec_name: str, min_chunks: int) -> bool:
    p = ROOT / 'results' / dec_name
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
        return d.get('summary', {}).get('chunks', 0) >= min_chunks
    except (json.JSONDecodeError, OSError):
        return False

def run_stage(cmd: list[str], out_path: Path, label: str, *, ignore_recent: bool=False) -> int:
    if not ignore_recent and out_path.exists() and (time.time() - out_path.stat().st_mtime < 7200):
        log(f'skip {label}: {out_path.name} updated in last 2h (already running or just finished?)')
        return 0
    log(f'starting {label}')
    log(f"  cmd: {' '.join(cmd)}")
    try:
        with out_path.open('w', encoding='utf-8') as out:
            return subprocess.call(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    except PermissionError:
        log(f'failed {label}: could not open {out_path.name} (locked)')
        return 1

def ingest_cmd(schedule: str, run_tag: str, *, no_query_train: bool=False) -> list[str]:
    cmd = [sys.executable, str(ROOT / '_stage255_stream_ingest.py'), '--schedule', schedule, *STREAM_COMMON, '--run-tag', run_tag]
    if no_query_train:
        cmd.append('--no-query-train')
    return cmd

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-ablation', action='store_true', help='only stream_wmn_v1, no --no-query-train runs')
    ap.add_argument('--skip-scale-ablation', action='store_true', help='skip the wiki:12 --no-query-train ablation (step 4), keep only the small 4-chunk one')
    ap.add_argument('--scale-ablation-only', action='store_true', help='run only ablate_no_wq_wiki12 (wiki:12 --no-query-train), no wait/stream')
    ap.add_argument('--stage257-only', action='store_true', help='wait for ablate_no_wq_wiki12, then run stage 257 (use while scale ablation is already on GPU)')
    ap.add_argument('--stage259-only', action='store_true', help='run stage 259 hot-swap smoke only (needs stage256_slot_bias.pt; no GPU queue wait)')
    ap.add_argument('--quick-smokes', action='store_true', help='259 hot-swap smoke, then 257 --smoke, then 258 --smoke (parallel-safe with wiki12 ablation)')
    ap.add_argument('--skip-stage257', action='store_true', help='do not queue _stage257_fp_compose.py after ablation')
    ap.add_argument('--skip-stage258-smoke', action='store_true', help='after stage 257, do not run _stage258_semantic_query.py --smoke')
    ap.add_argument('--poll-s', type=int, default=POLL_S)
    args = ap.parse_args()
    if args.stage259_only:
        rc259 = run_stage(stage259_smoke_cmd(), OUT_259, 'stage259_smoke', ignore_recent=True)
        log(f'stage259 smoke exit={rc259}')
        return rc259
    if args.quick_smokes:
        log('quick-smokes: 259 -> 257 smoke -> 258 smoke')
        rc259 = run_stage(stage259_smoke_cmd(), OUT_259, 'stage259_smoke', ignore_recent=True)
        log(f'stage259 smoke exit={rc259}')
        rc257 = run_stage(stage257_smoke_cmd(), OUT_257_SMOKE, 'stage257_smoke', ignore_recent=True)
        log(f'stage257 smoke exit={rc257}')
        rc258 = run_stage(stage258_smoke_cmd(), OUT_258, 'stage258_smoke', ignore_recent=True)
        log(f'stage258 smoke exit={rc258}')
        return rc258 if rc258 else rc257 if rc257 else rc259
    if args.stage257_only:
        log(f'stage257-only: waiting for ablate_no_wq_wiki12 (poll {args.poll_s}s)')
        while not ablate_wiki12_done():
            if OUT_ABL_W12.exists():
                lines = [ln for ln in OUT_ABL_W12.read_text(encoding='utf-8', errors='ignore').splitlines() if ln.strip()]
                tail = (lines[-1][:120] if lines else '...').encode('ascii', 'replace').decode('ascii')
                log(f'  ablate_wiki12 tail: {tail}')
            time.sleep(args.poll_s)
        log('ablate_no_wq_wiki12 done — starting stage 257')
        rc = run_stage257(force=True)
        log(f'stage257 exit={rc}')
        if rc == 0 and (not args.skip_stage258_smoke):
            rc258 = run_stage(stage258_smoke_cmd(), OUT_258, 'stage258_smoke', ignore_recent=True)
            log(f'stage258 smoke exit={rc258}')
            return rc258 if rc258 else rc
        return rc
    if args.scale_ablation_only:
        log('scale-ablation-only: ablate_no_wq_wiki12')
        rc = run_stage(ingest_cmd('wiki:12', 'ablate_no_wq_wiki12', no_query_train=True), OUT_ABL_W12, 'ablate_no_wq_wiki12')
        log(f'ablate_no_wq_wiki12 exit={rc}')
        if not args.skip_stage257 and rc == 0:
            rc257 = run_stage257(force=True)
            log(f'stage257 exit={rc257}')
            return rc257 if rc257 else rc
        return rc
    log(f'waiting for wiki:12 to finish (poll {args.poll_s}s)')
    while not wiki12_done():
        if LOG_W12.exists():
            lines = [ln for ln in LOG_W12.read_text(encoding='utf-8', errors='ignore').splitlines() if ln.strip()]
            tail = (lines[-1][:120] if lines else '...').encode('ascii', 'replace').decode('ascii')
            log(f'  wiki12 tail: {tail}')
        time.sleep(args.poll_s)
    log('wiki:12 done — queueing stream_wmn_v1')
    rc = run_stage(ingest_cmd('wiki:6,med:3,news:3', 'stream_wmn_v1'), OUT_WMN, 'stream_wmn_v1')
    log(f'stream_wmn_v1 exit={rc}')
    if args.skip_ablation:
        log('ablation skipped (--skip-ablation)')
        return rc
    if not stream_done('stage255_decision_stream_wmn_v1.json', WMN_CHUNKS):
        log(f'stream_wmn_v1 did not reach {WMN_CHUNKS} chunks in decision — skipping ablations')
        return rc
    rc2 = 0
    if stream_done('stage255_decision_ablate_no_wq_4ch.json', ABL_CHUNKS):
        log('ablate_no_wq_4ch already complete — skip')
    else:
        log('stream_wmn_v1 done — starting ablation (no W_q, 4 chunks)')
        rc2 = run_stage(ingest_cmd('wiki:2,med:2', 'ablate_no_wq_4ch', no_query_train=True), OUT_ABL, 'ablate_no_wq_4ch')
        log(f'ablate_no_wq_4ch exit={rc2}')
    if args.skip_scale_ablation:
        log('scale-matched ablation skipped (--skip-scale-ablation)')
        return rc2 if rc2 else rc
    if stream_done('stage255_decision_ablate_no_wq_wiki12.json', ABL_W12_CHUNKS):
        log('ablate_no_wq_wiki12 already complete — skip')
    else:
        log('ablate_no_wq_4ch done — starting scale-matched ablation (no W_q, wiki:12)')
        rc3 = run_stage(ingest_cmd('wiki:12', 'ablate_no_wq_wiki12', no_query_train=True), OUT_ABL_W12, 'ablate_no_wq_wiki12')
        log(f'ablate_no_wq_wiki12 exit={rc3}')
        rc_out = rc3 if rc3 else rc2 if rc2 else rc
        if args.skip_stage257:
            log('stage257 skipped (--skip-stage257)')
            return rc_out
        rc257 = run_stage257(force=True)
        log(f'stage257 exit={rc257}')
        return rc257 if rc257 else rc_out
    if not args.skip_stage257 and ablate_wiki12_done() and (not stage257_done()):
        log('ablate_no_wq_wiki12 already done — starting stage 257')
        rc257 = run_stage257(force=True)
        log(f'stage257 exit={rc257}')
        return rc257
    return rc2 if rc2 else rc
if __name__ == '__main__':
    raise SystemExit(main())