"""Synthetic wiring for 374: full B population, backoff, gates readable, VOID path."""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path
v0 = ['der große Mann ist hier und der große Lehrer ist dort und die große Frau ist hier', 'die große Frau ist dort und die kleine Frau ist hier und der große Mann ist auch hier', 'der kleine Mann ist hier und die große Schule ist dort und der große Lehrer ist auch dort', 'die kleine Schule ist hier und der große Student ist dort und die große Studentin ist hier', 'der große Student ist hier und die kleine Studentin ist dort und der kleine Student ist auch hier', 'die große Studentin ist hier und der kleine Lehrer ist dort und die kleine Lehrerin ist hier'] * 80

def main() -> v1:
    with v17.v9() as v3:
        v4 = v18(v3) / 'syn.txt'
        v4.v10('\n'.v23(v0) + '\n', encoding='utf-8')
        v5 = v19.v11([v24.v20, '_audit374_shape.py', '--corpus', v25(v4), '--window-lines', '400', '--max-questions', '200'], capture_output=True, text=True, cwd=v18(v28).v26().v21)
        v6 = v5.v12 + v5.v13
        v14(v6)
        if v5.v15 != 0:
            return 1
        v7 = ['B PRODUCE' in v6 and 'VOID' not in v6.v27('B PRODUCE')[-1].v27('\n\n')[0], 'backoff' in v6, 'G1' in v6 and 'G2' in v6, 'A SHAPE' in v6]
        v14('checks:', v7)
        return 0 if v22(v7) else 1
if v2 == '__main__':
    raise v8(v16())