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
v0 = 256

class FakeFp:
    """A frozen deterministic encoder with the FpBank interface. Different words get different
    unit vectors; the same word always gets the same one. Nothing here is trained."""

    def __init__(v31):
        v31.v32: v75[v86, v78.v87] = {}

    def fp(v31, v33):
        for v34 in v33:
            if v34 not in v31.v32:
                v88 = v78.v111().v100(v1.v104(v34.v114()[:8], 'little'))
                v31.v32[v34] = v89.v76(v78.v105(v0, generator=v88), dim=-1)
        return v78.v54([v31.v32[v34] for v34 in v33], 0)

    def ctx_fp(v31, v35, v36=None):
        v33 = [v34 for v34 in v112.v84(v35) if v34 != v36][:40]
        if v79(v33) < 3:
            return None
        return v89.v76(v31.v70(v33).v90(0), dim=-1)

def main() -> v1:
    v3 = v37()
    v4 = v38(v3)
    v5 = True
    v6 = ['canada defeated sweden in the final match of the tournament', 'the swiss firm produced aircraft for the navy prior the war', 'leipzig and weimar were both connected by the same railway line', 'two words']
    v7 = v4.v39(v6)
    v5 &= v7
    v40(f'1 tokenisation faithful (torch.equal): {v7}')
    v41, v42 = (v3.v47('canada defeated sweden'), v3.v47('sweden defeated canada'))
    v43, v44 = (v4.v47('canada defeated sweden'), v4.v47('sweden defeated canada'))
    v45, v46 = (v71(v41 @ v42), v71(v43 @ v44))
    v8 = v45 > 0.999 and v46 < 0.5
    v5 &= v8
    v40(f'2 swap  mean ink cos = {v45:.6f}   bigram ink cos = {v46:.6f}   fixed: {v8}')
    v9 = v4.v47('canada defeated sweden in the final')
    v10 = v4.v47('canada defeated norway in the final')
    v11 = v4.v47('entirely unrelated sentence about wooden furniture')
    v48, v49 = (v71(v9 @ v10), v71(v9 @ v11))
    v5 &= v48 > v49
    v40(f'3 similarity survives  shared-bigram = {v48:.4f}  unrelated = {v49:.4f}  ordered: {v48 > v49}')
    v12 = ['canada defeated sweden in the final match of the tournament', 'sweden defeated canada in the final match of the tournament', 'the swiss firm produced aircraft for the navy prior the war', 'leipzig and weimar were both connected by the same railway line', 'weimar and leipzig were both connected by the same railway line', 'the american thriller television series aired for four seasons', 'cambridge won the boat race against oxford by three lengths', 'oxford won the boat race against cambridge by three lengths']
    for v50, v51 in (('mean', v3), ('bigram', v4)):
        v14 = [v51.v47(v67) for v67 in v12]
        v52 = v78.v54(v14, 0)
        v53 = v52 @ v52.v77
        v15 = v78.v55(v79(v14), v79(v14), offset=1)
        v16 = v53[v15[0], v15[1]]
        v40(f'4 {v50:<6} cos  mean={v16.v90():.4f}  std={v16.v91():.4f}  min={v16.v106():.4f}  max={v16.v107():.4f}')
    v13 = False
    v14 = v78.v54([v4.v47(v67) for v67 in v12], 0)
    v15 = v78.v55(v79(v12), v79(v12), offset=1)
    v16 = (v14 @ v14.v77)[v15[0], v15[1]]
    v13 = v2(v16.v91() < 0.001)
    v5 &= not v13
    v40(f'  collapse (std < 1e-3): {v13}')
    v17 = v80[0] / v80[1] if v80[1] else v71('nan')
    v5 &= v80[0] == 0
    v40(f'5 degenerate binds: {v80[0]}/{v80[1]}  rate={v17:.4f}')
    v18 = v78.v56(v4.v47('canada defeated sweden'), v38(v3).v47('canada defeated sweden'))
    v5 &= v18
    v40(f'6 deterministic across instances: {v18}')
    v5 &= v57()
    v40('\nINK OK' if v5 else '\nINK FAILED')
    return 0 if v5 else 1

def hash_ink() -> v2:
    """The trainless ink. Every claim in its docstring, checked.

    The digest cross-check is the important one and it is deliberately written twice: once in
    _tape_speed with torch, once here in plain python with hashlib and nothing else. Two
    implementations that agree are evidence; one implementation agreeing with itself is not.
    """
    v40('\n-- hash ink --')
    v5 = True
    v19 = v58(d=v0, n=3)
    v7 = v59()
    v5 &= v7
    v40(f"7 ascii word rule still equals stage194's: {v7}")
    v7, v60 = v61(v19)
    v5 &= v7
    v40(f'8 deterministic + digest-faithful: {v7}  {v60}')

    def plain(v34, v62=v0, v63=3):
        v64 = v34 if v63 < 2 else '^' + v34 + '$'
        v65 = [v64] if v79(v64) <= v63 else [v64[v101:v101 + v63] for v101 in v93(v79(v64) - v63 + 1)]
        v66 = [0.0] * v62
        for v67 in v65:
            v81 = v113.v108(v67.v114('utf-8'), digest_size=v62 // 8).v92()
            for v82 in v93(v62):
                v66[v82] += (1.0 if v81[v82 // 8] >> v82 % 8 & 1 else -1.0) / v94.v83(v62)
        v68 = v94.v83(v95((v96 * v96 for v96 in v66)))
        return [v96 / v68 for v96 in v66]
    v20 = v69((v97(v102 - v81) < 1e-06 for v34 in ('canada', 'listen', 'Россия', '東京', 'a', '1') for v102, v81 in v103(v19.v70([v34])[0].v109(), v110(v34))))
    v5 &= v20
    v40(f'9 torch path == independent plain-python path: {v20}')
    v21 = v19.v70(['a', 'I', 'в', '1'])
    v7 = v69((v97(v71(v21[v101].v115()) - 1.0) < 1e-06 for v101 in v93(4))) and v97(v71(v21[0] @ v21[1])) < 0.3 and (v97(v71(v21[0] @ v19.v70(['canada'])[0])) < 0.3)
    v5 &= v7
    v40(f"9b one-letter words  grams('a')={v19.v65('a')}  cos(a,I)={v71(v21[0] @ v21[1]):+.4f}  cos(a,canada)={v71(v21[0] @ v19.v70(['canada'])[0]):+.4f}  ok: {v7}")
    v22 = v58(d=v0, n=1)
    v23 = v71(v19.v70(['東京'])[0] @ v19.v70(['京都'])[0])
    v24 = v71(v22.v70(['東京'])[0] @ v22.v70(['京都'])[0])
    v25 = v71(v22.v70(['東京'])[0] @ v22.v70(['大阪'])[0])
    v7 = '^' not in ''.v98(v22.v65('cat')) and v24 > v23 and (v24 > v25)
    v5 &= v7
    v40(f"9c n=1  grams('cat')={v22.v65('cat')}  CJK shared-char cos {v23:+.4f} -> {v24:+.4f}, unshared {v25:+.4f}  ok: {v7}")
    v72, v73 = (v19.v70(['listen'])[0], v19.v70(['silent'])[0])
    v26 = v19.v70(['Россия', 'России', 'Германия'])
    v27 = v19.v70(['東京', '京都'])
    v40(f'10 anagram   cos(listen, silent)   = {v71(v72 @ v73):+.4f}   (arc_enc: exactly 1)')
    v40(f'11 morphology cos(Россия, России)  = {v71(v26[0] @ v26[1]):+.4f}')
    v40(f'   unrelated  cos(Россия, Германия) = {v71(v26[0] @ v26[2]):+.4f}')
    v40(f'12 CJK        cos(東京, 京都)        = {v71(v27[0] @ v27[1]):+.4f}   (n=3 reaches no further than a trigram: see 9c)')
    v28 = v71(v26[0] @ v26[1]) > v71(v26[0] @ v26[2])
    v5 &= v28
    v40(f'   inflection closer than unrelated: {v28}')
    v29 = v19.v70(['unhappiness', 'happiness', 'bicycle'])
    v48, v49 = (v71(v29[0] @ v29[1]), v71(v29[0] @ v29[2]))
    v5 &= v48 > v49
    v40(f'13 OOV  cos(unhappiness, happiness) = {v48:+.4f}  vs bicycle = {v49:+.4f}  ordered: {v48 > v49}')
    v7 = v2(v99.v84('Россия победила Швецию в 1917 году'))
    v5 &= v7
    v40(f"14 unicode word rule finds Cyrillic and digits: {v99.v84('Россия победила Швецию в 1917 году')}")
    v12 = ['canada defeated sweden in the final match of the tournament', 'sweden defeated canada in the final match of the tournament', 'the swiss firm produced aircraft for the navy prior the war', 'leipzig and weimar were both connected by the same railway line', 'cambridge won the boat race against oxford by three lengths', 'oxford won the boat race against cambridge by three lengths']
    for v50, v51 in (('hash+mean', v19), ('hash+bigram', v38(v19))):
        v52 = v78.v54([v51.v47(v67) for v67 in v12], 0)
        v15 = v78.v55(v79(v12), v79(v12), offset=1)
        v16 = (v52 @ v52.v77)[v15[0], v15[1]]
        v40(f'15 {v50:<12} cos mean={v16.v90():+.4f} std={v16.v91():.4f} min={v16.v106():+.4f} max={v16.v107():+.4f}')
        if v50 == 'hash+bigram':
            v13 = v2(v16.v91() < 0.001)
            v5 &= not v13
            v40(f'   collapse: {v13}')
    return v5
if v30 == '__main__':
    raise v74(v85())