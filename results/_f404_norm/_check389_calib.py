"""389: does the calibration term actually remove the gauge? No torch needed for most of it, but
torch is used where it exists - the term is eight lines and every one of them can be wrong in a way
that still trains.

WHAT IS BEING CHECKED, and why each property is a wrong NUMBER rather than an exception:

  1. SHIFT INVARIANCE, ACROSS the batch. Adding one constant to every question's score must not
     change the term. If it did, the term would be a level control and could be won by pushing
     everything down - which is exactly how the three per-question refusal attempts were won by
     "always refuse".
  2. SHIFT SENSITIVITY, WITHIN the batch. Raising ONE question's score must change the term. This
     is the whole content: a per-question offset must no longer be free.
  3. DIRECTION. Raising an ANSWERABLE question's score must increase the term; raising an
     UNANSWERABLE one must decrease it. A sign error here trains the mind to be least confident
     exactly where the tape can answer, and nothing would crash.
  4. NO TARGET, NO TERM. All-positive and all-negative batches return exactly zero - not a
     fabricated uniform target, and not a NaN from dividing by npos = 0.
  5. THE OPTIMUM IS NOT A CONSTANT. A flat score vector scores strictly worse than one that
     separates the answerable questions, at the same mean. If a constant were optimal the term
     would teach nothing.
  6. THE LABEL IS THE TAPE'S, NOT THE MIND'S. reach_loss must record `ans` (truth in cands) and
     must NOT record `rt` (the mind's own correctness): training on `right` would be a moving
     target and would leave no held-out target to read the result on.
  7. ONE PICK FEEDS BOTH TERMS. reach_pick is called ONCE when either accumulator is live. Two
     calls would build the staged argmax twice with no guarantee the two terms agree about which
     world was settled on.
  8. THE ACCUMULATOR IS SCOPED. _CALIB_ACC is None outside the batch and is cleared in a
     `finally`, so a raising question cannot leave graph-holding tensors alive across steps.
  9. THE PAIRING IS POSITIONAL. A short accumulator raises rather than silently pairing question
     i's score with question j's label.
 10. B = 1 IS REFUSED. A softmax over one score is 1.0 whatever Phi says, and the gauge it exists
     to remove would still be per-question - the arm would be its own control wearing a flag.
 11. THE ARM IS IN THE TRANSPLANT SIGNATURE. A calibrated and an uncalibrated mind no longer
     measure on the same ruler.
 12. THE BATCH IS NOT QUIETLY SHARED. With both flags on, B is the larger of the two, and each
     term is added only if its own flag asked for it.

    python _check389_calib.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
SRC = Path('_stage289_derivation.py')

def _mini_torch():
    """The four operations calib_term uses, on plain floats. Forward only.

    This exists so the arithmetic of the term is checked in an environment without torch rather
    than taken on trust until the run. It deliberately implements the ops NAIVELY - if the stub
    and the real library ever disagreed, the properties below are about a formula that holds
    either way (shift invariance, sign, the zero case), so a stub that is wrong in the same
    direction as torch is not a way to pass.
    """
    import math

    class V:
        __slots__ = ('x',)

        def __init__(self, x):
            self.x = [float(v) for v in x]
        dtype = None
        device = None

        def __mul__(self, o):
            return V([a * b for a, b in zip(self.x, o.x)])

        def __truediv__(self, k):
            return V([a / float(k) for a in self.x])

        def sum(self):
            return sum(self.x)

        def __len__(self):
            return len(self.x)

    class _T:

        @staticmethod
        def stack(xs):
            return V([float(v) for v in xs])

        @staticmethod
        def tensor(x, device=None, dtype=None, requires_grad=False):
            return V(x) if isinstance(x, (list, tuple)) else float(x)

        @staticmethod
        def zeros(_shape, device=None, dtype=None):
            return 0.0

        @staticmethod
        def device(_name):
            return _name

        @staticmethod
        def log_softmax(v, _dim):
            mx = max(v.x)
            z = math.log(sum((math.exp(a - mx) for a in v.x)))
            return V([a - mx - z for a in v.x])
    return _T()

def main() -> int:
    src = SRC.read_text(encoding='utf-8')
    fails = []
    try:
        import torch
        real_torch = True
    except Exception:
        torch, real_torch = (_mini_torch(), False)
        print('NOTE: torch missing - properties 1-5 run against a numeric stub of the four ops calib_term uses (stack, tensor, zeros, log_softmax). The FORMULA is checked; the gradient sub-check needs the real thing and is skipped.')
    if True:
        ns = {'torch': torch}
        m = re.search('^def calib_term\\(.*?(?=\\n(?:def |@|\\w))', src, re.S | re.M)
        if not m:
            print(f'FAIL: calib_term not found in {SRC}')
            return 1
        exec(compile(m.group(0), 'calib_term', 'exec'), ns)
        term = ns['calib_term']
        dev = torch.device('cpu')

        def T(xs):
            return [torch.tensor(float(x), requires_grad=True) for x in xs]
        base = [0.3, -1.2, 2.0, 0.1]
        lab = [1.0, 0.0, 1.0, 0.0]
        v0 = float(term(T(base), lab, dev))
        v_shift = float(term(T([x + 4.7 for x in base]), lab, dev))
        if abs(v_shift - v0) > 1e-05:
            fails.append(f'1. the term is not shift invariant across the batch: {v0:.6f} -> {v_shift:.6f}. It can be won by a level, like `always refuse` was')
        up_pos = base[:]
        up_pos[0] += 1.0
        up_neg = base[:]
        up_neg[1] += 1.0
        v_pos = float(term(T(up_pos), lab, dev))
        v_neg = float(term(T(up_neg), lab, dev))
        if abs(v_pos - v0) < 1e-06:
            fails.append("2. raising one question's score changed nothing - the per-question offset is still free and the term does not tie the gauge")
        if not v_pos > v0:
            fails.append(f"3. raising an ANSWERABLE question's score did not raise the term ({v0:.6f} -> {v_pos:.6f}) - the sign is inverted")
        if not v_neg < v0:
            fails.append(f"3. raising an UNANSWERABLE question's score did not lower the term ({v0:.6f} -> {v_neg:.6f})")
        for name, ls in (('all-negative', [0.0, 0.0, 0.0, 0.0]), ('all-positive', [1.0, 1.0, 1.0, 1.0])):
            try:
                z = float(term(T(base), ls, dev))
            except Exception as e:
                fails.append(f'4. a {name} batch raised {type(e).__name__}: {e} - the term has no guard for a batch with no target')
                continue
            if z != 0.0 or z != z:
                fails.append(f'4. a {name} batch returned {z} instead of an exact zero')
        flat = float(term(T([0.5, 0.5, 0.5, 0.5]), lab, dev))
        sep = float(term(T([1.5, -0.5, 1.5, -0.5]), lab, dev))
        if not sep > flat:
            fails.append(f'5. a constant score is not worse than a separating one ({flat:.6f} vs {sep:.6f}) - the term teaches nothing')
        if real_torch:
            ts = T(base)
            term(ts, lab, dev).backward()
            if ts[0].grad is None or float(ts[0].grad) == 0.0:
                fails.append('5. no gradient reaches the scores - the term is decorative')
    rl = re.search('^def reach_loss\\(.*?(?=\\n(?:def |@|\\w))', src, re.S | re.M).group(0)
    if '_CALIB_ACC.append((_sc, 1.0 if ans else 0.0))' not in rl:
        fails.append("6. reach_loss does not record (raw score, answerable) - check the label: `rt` is the mind's own correctness and must NOT be the teacher")
    if re.search('_CALIB_ACC\\.append\\(\\(_sc, 1\\.0 if rt', rl):
        fails.append('6. the calibration term is trained on `right`, its own moving correctness, and `right` is then no longer a held-out target')
    if rl.count('reach_pick(q, l1, l2, own, cands, l3, lcands, keep_graph=True)') != 1:
        fails.append('7. reach_pick is not called exactly once in reach_loss - the two terms have no guarantee of agreeing about which world was settled on')
    if 'if _SPEAK_ACC is not None or _CALIB_ACC is not None:' not in rl:
        fails.append('7. the pick is not guarded by BOTH accumulators')
    loop = re.search('if SPEAK_BATCH or CALIB_BATCH:.*?(?=\\n        elif VIEWS)', src, re.S)
    if not loop:
        fails.append('12. the batch block is not driven by both flags - --calib-batch alone would never form a batch and the term would silently never fire')
    else:
        lb = loop.group(0)
        if 'bn = max(SPEAK_BATCH, CALIB_BATCH)' not in lb:
            fails.append("12. B is not the larger of the two batches - one term would be given the other's size without saying so")
        if '_CALIB_ACC = [] if CALIB_BATCH else None' not in lb:
            fails.append('8. the calibration accumulator is opened even when the flag is off')
        if '_SPEAK_ACC = _CALIB_ACC = None' not in lb:
            fails.append('8. the accumulators are not both cleared - a raising question would leave graph-holding tensors alive across steps')
        if 'finally:' not in lb:
            fails.append('8. the accumulators are not cleared in a `finally`')
        if 'calibration batch recorded' not in lb:
            fails.append('9. a short calibration accumulator does not raise - the pairing would stop being positional silently')
        if 'if CALIB_BATCH:' not in lb or 'if SPEAK_BATCH:' not in lb:
            fails.append('12. a term is added without checking its own flag')
    if '--calib-batch needs at least 2 questions' not in src:
        fails.append('10. --calib-batch 1 is not refused, and it is a constant term over a free per-question gauge - the arm would be its own control')
    sig = re.search('"speak_batch": SPEAK_BATCH.*?"constrain": CONSTRAIN', src, re.S)
    if not sig or '"calib_batch": CALIB_BATCH' not in sig.group(0):
        fails.append('11. the calibration arm is missing from the transplant signature - a calibrated mind could be dropped into an uncalibrated run in silence')
    if fails:
        print('FAIL')
        for f in fails:
            print('  ' + f)
        return 1
    print('PASS  the term is invariant to a batch-wide shift and sensitive to a per-question one,')
    print('  it rises on answerable questions and falls on unanswerable ones, a batch with no')
    print('  target returns an exact zero, a constant score is strictly worse than a separating')
    print("  one, the teacher is the tape's `answerable` and never the mind's `right`, one pick")
    print('  feeds both terms, the accumulator is scoped and positional, B=1 is refused, and the')
    print('  arm is in the transplant signature.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())