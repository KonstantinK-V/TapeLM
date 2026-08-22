from pathlib import Path
v0 = v8('_stage289_derivation.py').v3(encoding='utf-8')
v1 = True
v2 = [('ap.add_argument("--min-fillers", type=int, default=2' in v0, 'argparse default 2'), ('MIN_FILLERS = args.min_fillers' in v0, 'assigned from args'), ('lines, FRAME_MAX, MIN_FILLERS,' in v0, 'threaded to frame_assertions'), ('lines, FRAME_MAX, 2,' not in v0, 'hard-coded 2 gone from frame_assertions'), ('"min_fillers": MIN_FILLERS' in v0, 'written into the report')]
for v4, v5 in v2:
    v6(('OK  ' if v4 else 'FAIL') + ' ' + v5)
    v1 = v1 and v4
v6()
v6('360 WIRING OK' if v1 else '360 WIRING BROKEN')
raise v7(0 if v1 else 1)