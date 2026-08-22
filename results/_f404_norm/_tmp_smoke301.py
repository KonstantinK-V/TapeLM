import subprocess, sys
cmd = [sys.executable, '-u', '_stage289_derivation.py', '--tape', 'frames', '--frame-max', '3', '--min-mentions', '1', '--fp', 'hash', '--write-fp', 'hash', '--ink', 'mean', '--write-ink', 'mean', '--words', 'ascii', '--reach', '--reach-no-refuse', '--reach-lookahead', '--frame-fp', 'fillers', '--tape-sample', 'region', '--objective', 'reward', '--addresses', '3000', '--reach-max-q', '2000', '--probe-period', '1000', '--probe-size', '60', '--cpu', '--train-steps', '10', '--seed', '1337', '--run-tag', '301_smoke']
r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
out = (r.stdout or '') + '\n' + (r.stderr or '')
for line in out.splitlines():
    if any((k in line for k in ('tape:', 'frame_pool', 'questions', 'too few', 'node vector', 'Traceback', 'REACH', 'step '))):
        print(line[:200])
print('exit', r.returncode)