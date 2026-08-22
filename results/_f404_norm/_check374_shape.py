"""Synthetic wiring for 374: full B population, backoff, gates readable, VOID path."""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path
LINES = ['der große Mann ist hier und der große Lehrer ist dort und die große Frau ist hier', 'die große Frau ist dort und die kleine Frau ist hier und der große Mann ist auch hier', 'der kleine Mann ist hier und die große Schule ist dort und der große Lehrer ist auch dort', 'die kleine Schule ist hier und der große Student ist dort und die große Studentin ist hier', 'der große Student ist hier und die kleine Studentin ist dort und der kleine Student ist auch hier', 'die große Studentin ist hier und der kleine Lehrer ist dort und die kleine Lehrerin ist hier'] * 80

def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'syn.txt'
        p.write_text('\n'.join(LINES) + '\n', encoding='utf-8')
        r = subprocess.run([sys.executable, '_audit374_shape.py', '--corpus', str(p), '--window-lines', '400', '--max-questions', '200'], capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
        out = r.stdout + r.stderr
        print(out)
        if r.returncode != 0:
            return 1
        ok = ['B PRODUCE' in out and 'VOID' not in out.split('B PRODUCE')[-1].split('\n\n')[0], 'backoff' in out, 'G1' in out and 'G2' in out, 'A SHAPE' in out]
        print('checks:', ok)
        return 0 if all(ok) else 1
if __name__ == '__main__':
    raise SystemExit(main())