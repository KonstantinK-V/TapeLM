"""Night queue after 278 n16: stage 281 frames, then 280 if ceiling clears 0.375."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
WAIT_PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CEILING_FLOOR = 0.375

def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        SYNCHRONIZE = 1048576
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def run(cmd: list[str], log_name: str) -> int:
    log = RES / log_name
    print(f"[queue] start {' '.join(cmd)} -> {log}", flush=True)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f'[queue] done {cmd[1]} exit={p.returncode}', flush=True)
    return p.returncode

def main() -> int:
    RES.mkdir(exist_ok=True)
    if WAIT_PID > 0:
        print(f'[queue] waiting for pid {WAIT_PID}', flush=True)
        while alive(WAIT_PID):
            time.sleep(30)
        print('[queue] predecessor finished', flush=True)
    else:
        print('[queue] no wait (pid 0)', flush=True)
    on = RES / 'stage278_decisionon.json'
    if on.exists():
        (RES / 'stage278_decisionon_n16.json').write_bytes(on.read_bytes())
        mini = RES / 'stage278_minion.md'
        if mini.exists():
            (RES / 'stage278_minion_n16.md').write_bytes(mini.read_bytes())
    rc = run([sys.executable, '-u', '_stage281_frames.py'], '_stage281_full.out')
    dec = RES / 'stage281_decision.json'
    if not dec.exists():
        print('[queue] missing stage281_decision.json', flush=True)
        return rc or 1
    d = json.loads(dec.read_text(encoding='utf-8'))
    before = d.get('ceiling_before', {})
    after = d.get('ceiling_after', {})
    br = before.get('reward') if isinstance(before, dict) else None
    ar = after.get('reward') if isinstance(after, dict) else None
    print(f"[queue] 281 {d.get('overall')} ceiling {br} -> {ar}", flush=True)
    if ar is not None and ar >= CEILING_FLOOR:
        print('[queue] ceiling cleared 0.375 — re-smoke 280', flush=True)
        run([sys.executable, '-u', '_stage280_raw_exam.py', '--smoke', '--k-gap', '0.35'], '_stage280_smoke_after281.out')
        fp = RES / 'stage280_decision_fp.json'
        if fp.exists():
            s = json.loads(fp.read_text(encoding='utf-8'))
            ceil = s.get('teacher_ceiling_reward')
            print(f"[queue] 280 smoke ceiling={ceil} overall={s.get('overall')}", flush=True)
            if ceil is not None and ceil >= CEILING_FLOOR:
                print('[queue] starting full 280 --hop fp', flush=True)
                run([sys.executable, '-u', '_stage280_raw_exam.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--hop', 'fp'], '_stage280_full_fp.out')
    else:
        print('[queue] ceiling below 0.375 — no 280 restart (FRAMES wall elsewhere)', flush=True)
    print('[queue] night queue done', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())