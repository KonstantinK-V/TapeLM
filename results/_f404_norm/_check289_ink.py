"""Prove the bigram ink before spending six thousand steps on it. Seconds, no model needed.

Four things have to hold before an A/B between `--ink mean` and `--ink bigram` means anything,
and three of them are properties of the binding rather than of the corpus, so a fake encoder
tests them honestly and instantly:

  1 the duplicated tokenisation reproduces the base mean-ink BIT-FOR-BIT (torch.equal). If it
    does not, the arms differ in which words they see and the A/B measures tokenisation.
  2 the binding is non-commutative: `X defeated Y` and `Y defeated X` must stop being the same
    vector. Under mean ink their cosine is exactly 1.0 - that identity is the whole defect.
  3 shared structure still produces similarity: two sentences with a bigram in common must be
    closer than two unrelated ones, or the ink has traded blindness for noise.
  4 the bind does not cancel to a zero vector, and - the failure mode that would actually kill
    this - the cosines do not COLLAPSE to a single value. rank_norm is invariant to scale and
    offset, so a collapsed channel reaches the mind as pure noise while every printed number
    still looks reasonable. Spread is the thing to watch, not mean.

The fifth thing - whether real arc_enc output collapses under elementwise binding - a fake
encoder CANNOT test, because random vectors are the easy case and arc_enc's are correlated.
That one is measured on the real run: `cos_std` in the report, compared against the mean arm.

    python _check289_ink.py
"""
from __future__ import annotations
import hashlib
import math
import torch
import torch.nn.functional as F
from _stage194_fp_fact_memory import WORD_RE
from _tape_speed import INK_DEGENERATE, WORD_UNICODE, BigramBank, HashFp, verify_hash_ink, verify_word_rule
D = 256

class FakeFp:
    """A frozen deterministic encoder with the FpBank interface. Different words get different
    unit vectors; the same word always gets the same one. Nothing here is trained."""

    def __init__(self):
        self.c: dict[str, torch.Tensor] = {}

    def fp(self, ws):
        for w in ws:
            if w not in self.c:
                g = torch.Generator().manual_seed(int.from_bytes(w.encode()[:8], 'little'))
                self.c[w] = F.normalize(torch.randn(D, generator=g), dim=-1)
        return torch.stack([self.c[w] for w in ws], 0)

    def ctx_fp(self, text, exclude=None):
        ws = [w for w in WORD_RE.findall(text) if w != exclude][:40]
        if len(ws) < 3:
            return None
        return F.normalize(self.fp(ws).mean(0), dim=-1)

def main() -> int:
    base = FakeFp()
    bg = BigramBank(base)
    ok = True
    texts = ['canada defeated sweden in the final match of the tournament', 'the swiss firm produced aircraft for the navy prior the war', 'leipzig and weimar were both connected by the same railway line', 'two words']
    v = bg.verify_mean_path(texts)
    ok &= v
    print(f'1 tokenisation faithful (torch.equal): {v}')
    m1, m2 = (base.ctx_fp('canada defeated sweden'), base.ctx_fp('sweden defeated canada'))
    b1, b2 = (bg.ctx_fp('canada defeated sweden'), bg.ctx_fp('sweden defeated canada'))
    cm, cb = (float(m1 @ m2), float(b1 @ b2))
    swap_fixed = cm > 0.999 and cb < 0.5
    ok &= swap_fixed
    print(f'2 swap  mean ink cos = {cm:.6f}   bigram ink cos = {cb:.6f}   fixed: {swap_fixed}')
    p = bg.ctx_fp('canada defeated sweden in the final')
    q = bg.ctx_fp('canada defeated norway in the final')
    r = bg.ctx_fp('entirely unrelated sentence about wooden furniture')
    near, far = (float(p @ q), float(p @ r))
    ok &= near > far
    print(f'3 similarity survives  shared-bigram = {near:.4f}  unrelated = {far:.4f}  ordered: {near > far}')
    corpus = ['canada defeated sweden in the final match of the tournament', 'sweden defeated canada in the final match of the tournament', 'the swiss firm produced aircraft for the navy prior the war', 'leipzig and weimar were both connected by the same railway line', 'weimar and leipzig were both connected by the same railway line', 'the american thriller television series aired for four seasons', 'cambridge won the boat race against oxford by three lengths', 'oxford won the boat race against cambridge by three lengths']
    for mode, bank in (('mean', base), ('bigram', bg)):
        vs = [bank.ctx_fp(t) for t in corpus]
        M = torch.stack(vs, 0)
        C = M @ M.T
        iu = torch.triu_indices(len(vs), len(vs), offset=1)
        cv = C[iu[0], iu[1]]
        print(f'4 {mode:<6} cos  mean={cv.mean():.4f}  std={cv.std():.4f}  min={cv.min():.4f}  max={cv.max():.4f}')
    collapsed = False
    vs = torch.stack([bg.ctx_fp(t) for t in corpus], 0)
    iu = torch.triu_indices(len(corpus), len(corpus), offset=1)
    cv = (vs @ vs.T)[iu[0], iu[1]]
    collapsed = bool(cv.std() < 0.001)
    ok &= not collapsed
    print(f'  collapse (std < 1e-3): {collapsed}')
    deg = INK_DEGENERATE[0] / INK_DEGENERATE[1] if INK_DEGENERATE[1] else float('nan')
    ok &= INK_DEGENERATE[0] == 0
    print(f'5 degenerate binds: {INK_DEGENERATE[0]}/{INK_DEGENERATE[1]}  rate={deg:.4f}')
    same = torch.equal(bg.ctx_fp('canada defeated sweden'), BigramBank(base).ctx_fp('canada defeated sweden'))
    ok &= same
    print(f'6 deterministic across instances: {same}')
    ok &= hash_ink()
    print('\nINK OK' if ok else '\nINK FAILED')
    return 0 if ok else 1

def hash_ink() -> bool:
    """The trainless ink. Every claim in its docstring, checked.

    The digest cross-check is the important one and it is deliberately written twice: once in
    _tape_speed with torch, once here in plain python with hashlib and nothing else. Two
    implementations that agree are evidence; one implementation agreeing with itself is not.
    """
    print('\n-- hash ink --')
    ok = True
    H = HashFp(d=D, n=3)
    v = verify_word_rule()
    ok &= v
    print(f"7 ascii word rule still equals stage194's: {v}")
    v, notes = verify_hash_ink(H)
    ok &= v
    print(f'8 deterministic + digest-faithful: {v}  {notes}')

    def plain(w, d=D, n=3):
        s = w if n < 2 else '^' + w + '$'
        grams = [s] if len(s) <= n else [s[i:i + n] for i in range(len(s) - n + 1)]
        acc = [0.0] * d
        for t in grams:
            b = hashlib.blake2b(t.encode('utf-8'), digest_size=d // 8).digest()
            for k in range(d):
                acc[k] += (1.0 if b[k // 8] >> k % 8 & 1 else -1.0) / math.sqrt(d)
        nrm = math.sqrt(sum((x * x for x in acc)))
        return [x / nrm for x in acc]
    agree = all((abs(a - b) < 1e-06 for w in ('canada', 'listen', 'Россия', '東京', 'a', '1') for a, b in zip(H.fp([w])[0].tolist(), plain(w))))
    ok &= agree
    print(f'9 torch path == independent plain-python path: {agree}')
    solo = H.fp(['a', 'I', 'в', '1'])
    v = all((abs(float(solo[i].norm()) - 1.0) < 1e-06 for i in range(4))) and abs(float(solo[0] @ solo[1])) < 0.3 and (abs(float(solo[0] @ H.fp(['canada'])[0])) < 0.3)
    ok &= v
    print(f"9b one-letter words  grams('a')={H.grams('a')}  cos(a,I)={float(solo[0] @ solo[1]):+.4f}  cos(a,canada)={float(solo[0] @ H.fp(['canada'])[0]):+.4f}  ok: {v}")
    H1 = HashFp(d=D, n=1)
    cjk3 = float(H.fp(['東京'])[0] @ H.fp(['京都'])[0])
    cjk1 = float(H1.fp(['東京'])[0] @ H1.fp(['京都'])[0])
    nosh = float(H1.fp(['東京'])[0] @ H1.fp(['大阪'])[0])
    v = '^' not in ''.join(H1.grams('cat')) and cjk1 > cjk3 and (cjk1 > nosh)
    ok &= v
    print(f"9c n=1  grams('cat')={H1.grams('cat')}  CJK shared-char cos {cjk3:+.4f} -> {cjk1:+.4f}, unshared {nosh:+.4f}  ok: {v}")
    la, si = (H.fp(['listen'])[0], H.fp(['silent'])[0])
    ru = H.fp(['Россия', 'России', 'Германия'])
    cn = H.fp(['東京', '京都'])
    print(f'10 anagram   cos(listen, silent)   = {float(la @ si):+.4f}   (arc_enc: exactly 1)')
    print(f'11 morphology cos(Россия, России)  = {float(ru[0] @ ru[1]):+.4f}')
    print(f'   unrelated  cos(Россия, Германия) = {float(ru[0] @ ru[2]):+.4f}')
    print(f'12 CJK        cos(東京, 京都)        = {float(cn[0] @ cn[1]):+.4f}   (n=3 reaches no further than a trigram: see 9c)')
    morph = float(ru[0] @ ru[1]) > float(ru[0] @ ru[2])
    ok &= morph
    print(f'   inflection closer than unrelated: {morph}')
    oov = H.fp(['unhappiness', 'happiness', 'bicycle'])
    near, far = (float(oov[0] @ oov[1]), float(oov[0] @ oov[2]))
    ok &= near > far
    print(f'13 OOV  cos(unhappiness, happiness) = {near:+.4f}  vs bicycle = {far:+.4f}  ordered: {near > far}')
    v = bool(WORD_UNICODE.findall('Россия победила Швецию в 1917 году'))
    ok &= v
    print(f"14 unicode word rule finds Cyrillic and digits: {WORD_UNICODE.findall('Россия победила Швецию в 1917 году')}")
    corpus = ['canada defeated sweden in the final match of the tournament', 'sweden defeated canada in the final match of the tournament', 'the swiss firm produced aircraft for the navy prior the war', 'leipzig and weimar were both connected by the same railway line', 'cambridge won the boat race against oxford by three lengths', 'oxford won the boat race against cambridge by three lengths']
    for mode, bank in (('hash+mean', H), ('hash+bigram', BigramBank(H))):
        M = torch.stack([bank.ctx_fp(t) for t in corpus], 0)
        iu = torch.triu_indices(len(corpus), len(corpus), offset=1)
        cv = (M @ M.T)[iu[0], iu[1]]
        print(f'15 {mode:<12} cos mean={cv.mean():+.4f} std={cv.std():.4f} min={cv.min():+.4f} max={cv.max():+.4f}')
        if mode == 'hash+bigram':
            collapsed = bool(cv.std() < 0.001)
            ok &= not collapsed
            print(f'   collapse: {collapsed}')
    return ok
if __name__ == '__main__':
    raise SystemExit(main())