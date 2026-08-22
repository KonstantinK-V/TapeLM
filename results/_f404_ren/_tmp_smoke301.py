import subprocess, sys
v0 = [v9.v4, '-u', '_stage289_derivation.py', '--tape', 'frames', '--frame-max', '3', '--min-mentions', '1', '--fp', 'hash', '--write-fp', 'hash', '--ink', 'mean', '--write-ink', 'mean', '--words', 'ascii', '--reach', '--reach-no-refuse', '--reach-lookahead', '--frame-fp', 'fillers', '--tape-sample', 'region', '--objective', 'reward', '--addresses', '3000', '--reach-max-q', '2000', '--probe-period', '1000', '--probe-size', '60', '--cpu', '--train-steps', '10', '--seed', '1337', '--run-tag', '301_smoke']
v1 = v10.v5(v0, capture_output=True, text=True, encoding='utf-8', errors='replace')
v2 = (v1.v13 or '') + '\n' + (v1.v11 or '')
for v3 in v2.v6():
    if v12((v14 in v3 for v14 in ('tape:', 'frame_pool', 'questions', 'too few', 'node vector', 'Traceback', 'REACH', 'step '))):
        v7(v3[:200])
v7('exit', v1.v8)