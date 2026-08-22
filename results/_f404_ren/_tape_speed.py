"""Three exact speedups shared by every tape stage - and the reason they exist.

289a's resample cost twenty minutes and I guessed at the cause twice, wrongly, before measuring
it. Caching the corpus scan changed nothing; caching the ink changed nothing. The cost was
fp_addresses making a few million tiny GPU round-trips, and compute was never involved. The
lesson is in HANDOFF 9b: time the loop before optimising it.

All three are EXACT. Not "close enough to keep the conclusions" - byte-identical output,
verified against the original paths, with --no-scan-cache / --no-fast-grouping kept so that
verification can be repeated at any time. A speedup that moved a number would quietly
invalidate every gate that depends on it, which is worse than a slow run.

They live here rather than in a stage because 286 needs them too and 289a imports 286: putting
them in a stage would make the import graph a cycle.
"""
from __future__ import annotations
import hashlib
import math
import random
import re
from collections import defaultdict
import torch
import torch.nn.functional as F
v0 = v46.v9('[A-Za-z][a-z]{2,}')
v1 = v46.v9('[^\\W\\d_][^\\W_]*|\\d+', v46.v10)
v2 = {'ascii': v0, 'unicode': v1}
v3 = 40
v4 = 3

def words_of(v11, v12=None, v13=v0):
    return [v31 for v31 in v13.v171(v11) if v31 != v12][:v3]

def verify_word_rule():
    """The ascii rule is duplicated from stage 194. Duplicated rules drift; this catches it."""
    from _stage194_fp_fact_memory import WORD_RE
    return v0.v47 == v105.v47

def digest_of(v14: v48, v15: v6) -> v5:
    return v172.v148(v14.v173('utf-8'), digest_size=v15 // 8).v49()

def sign_bit(v16: v5, v17: v6) -> v6:
    return v16[v17 // 8] >> v17 % 8 & 1

def sign_vector(v16: v5, v15: v6) -> v20.v7:
    """The same mapping as sign_bit, vectorised. A python loop over d coordinates per n-gram
    costs minutes over a real vocabulary, and this stage has already been bitten once by doing
    per-item work that could be done per-batch."""
    v18 = v20.v50(v106(v16), dtype=v20.v107)
    v19 = v18.v149(1) >> v20.v150(8, dtype=v20.v107) & 1
    return (v19.v183(-1).v157(v20.v174) * 2.0 - 1.0) / v151.v108(v15)

def hash_perm(v15: v6, v21: v6=0) -> v20.v7:
    """A permutation of d coordinates derived from a digest rather than from torch's RNG.

    torch.randperm with a seeded generator is reproducible within a version and is NOT promised
    across versions or platforms. That is exactly the fragility hash ink exists to remove, so
    the bigram binding must not reintroduce it one line later: a torch upgrade would silently
    change every context vector while every gate still passed.

    Sorting the coordinates by an independent digest key is a permutation by construction - no
    rejection, no loop, nothing to get wrong - and is uniformly random for the same reason the
    keys are.
    """
    v22 = [v109(f'perm:{v21}:{v17}', 64) for v17 in v140(v15)]
    return v20.v51(v110(v140(v15), key=lambda v17: v22[v17]), dtype=v20.v111)

class CachedBank:
    """The ink is a pure function of the string, so inking the same context 120 times is pure
    waste - and that waste was the whole cost of this stage.

    A resample changes only WHICH addresses rng draws; the corpus does not change, so the
    contexts recur almost perfectly across tapes. Worse, every assertion was inked twice per
    resample: once by fp_addresses to build the key, once by pack_from_corpus to build the
    slot. Memoising on (text, exclude) removes both, changes no number anywhere - the frozen
    arc_enc returns the same vector by construction, and the hash gate still proves it - and
    turns a 20-hour run into a few hours.

    Nothing here is stored per ADDRESS or per identity: the key is the text, and the cache is
    an execution detail with no path into the mind's inputs.
    """
    v23 = 400000

    def __init__(v52, v53):
        v52.v71, v52.v112, v52.v113 = (v53, {}, {})

    def ctx_fp(v52, v11, v12=None):
        v17 = (v11, v12)
        if v17 not in v52.v112:
            if v155(v52.v112) >= v52.v23:
                v52.v112.v156()
            v52.v112[v17] = v52.v71.v152(v11, exclude=v12)
        return v52.v112[v17]

    def fp(v52, v54):
        v17 = v114(v54)
        if v17 not in v52.v113:
            v52.v113[v17] = v52.v71.v126(v54)
        return v52.v113[v17]

    def __getattr__(v52, v55):
        return v115(v52.v71, v55)

class HashFp:
    """Ink with nothing trained in it at all.

    ctx_fp and the frozen arc_enc were introduced at stage 191-194, when the question was
    whether a curve could hold anything and English wikitext was the whole world. Everything
    built since then rests on that choice, and it has three defects that no amount of care
    downstream can repair:

      arc_enc mean-pools characters, so fp("listen") == fp("silent"). Anagrams collide, a
      prefix and a suffix are the same evidence, morphology is invisible.

      stoi was built from an English corpus. A Cyrillic or CJK character maps to index 0, so
      every Russian word inks to nearly the same vector. This is not a regex problem and no
      regex fixes it - the wall to other languages is INSIDE the frozen weights.

      A word not in the training distribution has no meaningful vector, and there is no
      mechanism by which it could get one.

    So take the invariant to its limit. "Whatever is trained may not hold facts" is satisfied
    most simply by an ink that was never trained:

        fp(w)  = normalize( sum over character n-grams t of "^"+w+"$" of  H(t) )
        H(t)   = (bits of blake2b(t, d/8 bytes)) * 2 - 1, over sqrt(d)

    H is a deterministic +/-1 vector read straight out of a cryptographic digest. No generator,
    no torch RNG, no version dependence, no checkpoint: one string, one vector, on any machine,
    forever. Near-orthogonality is a property of the digest, not something tuned.

    What that buys, all at once and all for free:

      order inside the word     "listen" and "silent" share no trigram
      every script and symbol   utf-8 bytes go into the digest; there is no character vocabulary
      no OOV, ever              an unseen word shares n-grams with seen ones, so morphology
                                works by construction - "^un" and "ing$" are features, not gaps
      the freeze question ends  nothing is trained, so nothing can leak, and the mind stops
                                depending on stage191_p1_curve.pt to mean anything

    What it costs is whatever arc_enc actually learned. The evidence says that may be nothing:
    the only channel carrying signal is the cosine of a bag of a bag, which is already a surface
    overlap measure. If a digest ties a trained encoder, the encoder was never contributing
    semantics - and that result is worth more than the accuracy either way.

    The boundary markers matter and are not decoration. Without them "^cat" and "cat$" are the
    same trigram and a prefix cannot be told from a suffix, which is the morphology case. They
    are added only for n >= 2: inside a 1-gram a marker carries no adjacency, it is just a
    constant vector added to every word, which inflates every cosine equally and tells nobody
    anything. n=1 is the setting a CJK corpus would want, so this is not a hypothetical.

    n=3 is an assumption about ALPHABETIC scripts and it is measured, not assumed: two-character
    CJK words share no trigram at all (東京 gives ^東京 and 東京$, 京都 gives ^京都 and 京都$ -
    the character they share never becomes a shared gram), so cos comes out at +0.017. --fp-ngram
    is exposed for exactly that reason. What hash ink removes unconditionally is the character
    VOCABULARY and OOV; how far a gram of a given length reaches into a script is a separate
    question with a separate number.
    """
    v24 = 400000
    v25 = 400000

    def __init__(v52, v56=None, v15=256, v55=3, v13=v0):
        if v15 % 8 or not 8 <= v15 // 8 <= 64:
            raise v153(f'd={v15} must be a multiple of 8 with d/8 in [8, 64] for blake2b')
        if v55 < 1:
            raise v153(f'n={v55} must be at least 1')
        v52.v15, v52.v55, v52.v13 = (v15, v55, v13)
        v52.v56 = v56 or v20.v56('cpu')
        v52.v57: v39[v48, v20.v7] = {}
        v52.v58: v39[v48, v20.v7] = {}

    def _h(v52, v32: v48) -> v20.v7:
        """One gram's sign vector, on the CPU. Deliberately not on the device: building these
        one at a time on the GPU is a few hundred thousand tiny round-trips, which is the exact
        cost that made 289a take twenty minutes a resample. fp() transfers once per batch."""
        v59 = v52.v57.v116(v32)
        if v59 is None:
            if v155(v52.v57) >= v52.v25:
                v52.v57.v156()
            v59 = v52.v57[v32] = v154(v109(v32, v52.v15), v52.v15)
        return v59

    def grams(v52, v31: v48) -> v60[v48]:
        v14 = v31 if v52.v55 < 2 else '^' + v31 + '$'
        if v155(v14) <= v52.v55:
            return [v14]
        return [v14[v102:v102 + v52.v55] for v102 in v140(v155(v14) - v52.v55 + 1)]

    @v20.v63()
    def fp(v52, v61: v60[v48]) -> v20.v7:
        if v155(v52.v58) >= v52.v24:
            v52.v58.v156()
        v62 = [v31 for v31 in v39.v175(v61) if v31 not in v52.v58]
        if v62:
            v117 = v20.v118([v159.v120(v20.v118([v52.v121(v32) for v32 in v52.v188(v31)], 0).v124(0), dim=-1) for v31 in v62], 0).v157(v52.v56)
            for v31, v59 in v158(v62, v117):
                v52.v58[v31] = v59
        return v20.v118([v52.v58[v31] for v31 in v61], 0)

    @v20.v63()
    def ctx_fp(v52, v11: v48, v12: v48 | None=None) -> v20.v7 | None:
        v61 = v119(v11, v12, v52.v13)
        if v155(v61) < v4:
            return None
        return v159.v120(v52.v126(v61).v128(0), dim=-1)

def verify_hash_ink(v26, v27=('canada', 'listen', 'silent', 'Россия', '東京', '1917')):
    """Is the ink actually deterministic, and does the tensor path agree with the digest?

    G_arc_enc_frozen is vacuous in a hash arm - there are no weights to move - so this is the
    gate that replaces it, and it has to be stronger than "it ran twice". Three claims:

      1 a FRESH bank reproduces every probe bit-for-bit (no hidden state, no RNG)
      2 the sign of every coordinate equals the corresponding bit of the digest, checked
        against hashlib directly rather than against the same code path
      3 distinct strings give distinct vectors, and an anagram is not a collision

    Returns (ok, notes) so the caller can log what failed rather than just that something did.
    """
    v28 = {}
    v29 = v64(device=v26.v56, d=v26.v15, n=v26.v55, rule=v26.v13)
    v30 = True
    for v31 in v27:
        v96, v53 = (v26.v126([v31])[0], v29.v126([v31])[0])
        if not v20.v69(v96, v53):
            v30, v28[v31] = (False, 'not reproducible in a fresh bank')
    for v32 in ('^ca', 'abc', '東京京'):
        v16 = v172.v148(v32.v173('utf-8'), digest_size=v26.v15 // 8).v49()
        v59 = v26.v121(v32)
        v65 = [1.0 if v16[v17 // 8] >> v17 % 8 & 1 else -1.0 for v17 in v140(v26.v15)]
        if v122((v179(v123(v59[v17]) * v151.v108(v26.v15) - v65[v17]) > 1e-06 for v17 in v140(v26.v15))):
            v30, v28[v32] = (False, 'tensor path disagrees with the digest')
    v66, v67 = (v26.v126(['listen'])[0], v26.v126(['silent'])[0])
    v28['anagram_cos'] = v68(v123(v66 @ v67), 4)
    if v20.v69(v66, v67):
        v30, v28['anagram'] = (False, 'listen and silent still collide')
    v33 = v70(v26.v15)
    v34 = v6((v33 == v20.v150(v26.v15)).v124())
    v28['perm_fixed_points'] = v34
    if v34 > v26.v15 // 16:
        v30, v28['perm'] = (False, 'the coordinate permutation is close to the identity')
    return (v30, v28)
v8 = [0, 0]

class BigramBank:
    """Ink that can tell `X defeated Y` from `Y defeated X`.

    The edge ablation settled where the signal lives: `rare` contributes nothing (zeroing it
    leaves the run bit-identical, and alone it scores below the counting rival), so the whole
    4.42-sigma paired win sits in `cos` - the cosine between ctx_fp vectors. And ctx_fp is a
    mean over words of a mean over characters, order-blind twice. The mind's only working
    channel therefore cannot see which side of a relation a value stands on, and the errors say
    so: a third of them are exact swap pairs, A->B and B->A both present in the same confusion
    table (Leipzig<->Weimar, California<->Texas, British<->Canadian). It knows the pair and not
    the direction, which is precisely the signature of a bag.

    The fix belongs in the INK, not in the mind. arc_enc stays frozen and untrained, the
    permutation is a fixed function of a seed, and no parameter is added anywhere - the mind is
    bit-for-bit the same 3489 weights reading a sharper input.

        ctx_fp(T) = normalize( mean_i  fp(w_i) * P.fp(w_{i+1}) )

    P is a fixed permutation of the coordinates, applied to the RIGHT member only, which is what
    makes the binding non-commutative: bind(a,b) != bind(b,a). This is standard MAP binding from
    vector-symbolic algebra, not something invented here.

    It stays a bag - a bag of bigrams - and that is deliberate. Position-tagged sums (sum R^i
    fp(w_i)) also carry order, but shifting a phrase by one word changes every coordinate and
    similarity becomes brittle. Local order under global permutation-invariance is exactly what
    a relation needs and all it needs.

    The word list is re-derived here rather than taken from FpBank, which duplicates a rule -
    the 40-word cap and the 3-word minimum - and duplicated rules drift. verify_mean_path
    proves the duplicate is faithful by reproducing the base mean-ink bit-for-bit, and 289
    gates on it before training rather than assuming it.
    """

    def __init__(v52, v53, v21=0, v13=v0):
        v52.v71 = v53
        v52.v72 = v21
        v52.v73 = v13
        v52.v74 = None

    def _words(v52, v11, v12):
        if v52.v73.v47 == v0.v47:
            from _tape_index import context_words
            return v160(v11, exclude=v12)
        return v119(v11, v12, v52.v73)

    def _rot(v52, v15, v56):
        if v52.v74 is None or v155(v52.v74) != v15:
            v52.v74 = v70(v15, v52.v72)
        if v52.v74.v56 != v56:
            v52.v74 = v52.v74.v157(v56)
        return v52.v74

    @v20.v63()
    def ctx_fp(v52, v11, v12=None):
        v61 = v52.v125(v11, v12)
        if v155(v61) < v4:
            return None
        v75 = v52.v71.v126(v61)
        v76 = v52.v127(v75.v161[-1], v75.v56)
        v77 = (v75[:-1] * v75[1:].v180(-1, v76)).v128(0)
        v8[0] += v6(v123(v77.v181()) < 1e-06)
        v8[1] += 1
        return v159.v120(v77, dim=-1)

    @v20.v63()
    def verify_mean_path(v52, v78):
        """Does the tokenisation duplicated above reproduce the base ink exactly?

        Not "closely" - torch.equal. If this is false the bigram arm and the mean arm are not
        comparable and every A/B conclusion drawn from them is about tokenisation, not order.
        """
        for v32 in v78:
            v61 = v52.v125(v32, None)
            v96 = v52.v71.v152(v32)
            v53 = v159.v120(v52.v71.v126(v61).v128(0), dim=-1) if v155(v61) >= v4 else None
            if (v96 is None) != (v53 is None):
                return False
            if v96 is not None and (not v20.v69(v96, v53)):
                return False
        return True

    def __getattr__(v52, v55):
        return v115(v52.v71, v55)

def install_assertion_cache(v35):
    """The corpus scan is deterministic; only WHICH addresses get drawn is not.

    corpus_assertions walks every line with three regexes to build by_addr, and only then does
    `rng.shuffle(addrs); addrs = addrs[:max_addr]`. Nothing before that shuffle depends on rng,
    so a hundred and twenty resamples re-derive the same table a hundred and twenty times. This
    keeps the table and re-draws from it.

    It is exact rather than approximate, which matters more than the speed: the first call runs
    the ORIGINAL function with no address cap, so the cached table is precisely what the
    original would have built. Later calls shuffle the same address list with the caller's own
    rng - same list, same rng state, same draw - and rebuild the assertions in the same order,
    so `source` indices are identical too. No number moves.

    Returns the uninstall function, so a caller that wants the unmemoised path can have it.
    """
    v36 = v35.v37
    v38: v39 = {}

    def cached(v79, v80, v81, v82, v83, v84=3, v85=None, v86=False, v87=1):
        v88 = (v162(v79), v155(v79), v82, v83, v84, v86, v87, v155(v85) if v85 else -1)
        if v88 not in v38:
            v130, v163 = v36(v79, v182.v176(0), 10 ** 9, v82, v83, n_rel=v84, common=v85, require_link=v86, min_frame_words=v87)
            v129: v39 = v141(v60)
            for v96 in v130:
                v129[v96['address']].v147(v96)
            v131 = v60(v140(v155(v163)))
            v182.v176(0).v134(v131)
            v132 = [None] * v155(v163)
            for v102, v164 in v143(v131):
                v132[v164] = v163[v102]
            v38[v88] = (v132, v129)
        v133, v129 = v38[v88]
        v89 = v60(v133)
        v80.v134(v89)
        v89 = v89[:v81]
        v90 = [v39(v96) for v165 in v89 for v96 in v129[v165]]
        return (v90, v89)
    v35.v37 = v40
    return lambda: v135(v35, 'corpus_assertions', v36)

def install_fast_fp_addresses(v35):
    """Same grouping, same order, same ties - one matmul per assertion instead of one per group.

    The original single-link loop is not compute-bound, it is round-trip-bound: for every
    assertion it walks the groups built so far and, for each, does `torch.tensor(mem)` (host to
    device) and `.tolist()` (device to host). With four thousand assertions and a thousand
    groups that is a few million tiny transfers, each costing far more than the 256-wide dot it
    carries. Measured: about twenty minutes per resample, unchanged by caching the scan or the
    ink, because neither was the cost.

    Every index in every group is smaller than i, so ONE product against all previous
    assertions covers every comparison the loop can make. The group walk then reads that vector
    on the CPU in exactly the original order - groups in order, members in order, `>=` so later
    ties still win - and the assignment is identical.

    The one honest caveat: a batched matmul may accumulate in a different order than the
    per-group one, so a dot product can differ in the last bit. That only matters if a
    similarity sits exactly on tau, and it would change a grouping decision rather than corrupt
    one. Verified equal on synthetic data below the level where that can happen.
    """
    v36 = v35.v41

    def fast(v91, v92, v93: v123, v94: v6, v82: v6, v95: v48='two'):
        v22, v136, v137, v54 = ([], [], [], [])
        for v96 in v91:
            v138 = v96['address'].v177('|')[0]
            v139 = v92.v152(v96['ctx'], exclude=v96['value'])
            v17 = v92.v126([v138])[0]
            v22.v147(v159.v120(v17 + v139 if v139 is not None else v17, dim=-1))
            v136.v147(v159.v120(v17, dim=-1))
            v137.v147(v159.v120(v139, dim=-1) if v139 is not None else v159.v120(v17, dim=-1))
            v54.v147({v31.v184() for v31 in v35.v187.v171(v96['ctx']) if v31.v184() not in v35.v185} - {v96['value'].v184()})
        v97 = v159.v120(v20.v118(v22).v123(), dim=-1)
        v98 = v159.v120(v20.v118(v136).v123(), dim=-1)
        v99 = v159.v120(v20.v118(v137).v123(), dim=-1)
        v100: v60[v60[v6]] = []
        v101 = [-1] * v155(v91)
        for v102 in v140(v155(v91)):
            if v100:
                v166 = (v20.v186(v98[:v102] @ v98[v102], v99[:v102] @ v99[v102]) if v95 == 'two' else v97[:v102] @ v97[v102]).v178()
            v167, v168 = (-1, v93)
            for v142, v144 in v143(v100):
                for v164 in v144:
                    if v166[v164] >= v168 and v155(v54[v102] & v54[v164]) >= v94:
                        v167, v168 = (v142, v166[v164])
            if v167 < 0:
                v101[v102] = v155(v100)
                v100.v147([v102])
            else:
                v101[v102] = v167
                v100[v167].v147(v102)
        v103: v39 = v141(v60)
        for v102, v142 in v143(v101):
            v103[v142].v147(v102)
        v90, v89 = ([], [])
        for v142, v144 in v103.v145():
            if v155(v144) < v82:
                continue
            v146 = f"fp{v142}:{v91[v144[0]]['address']}"
            v89.v147(v146)
            for v55, v102 in v143(v144):
                v96 = v39(v91[v102])
                v96['address'] = v146
                v96['source'] = f'wiki:{v146}:{v55}'
                v90.v147(v96)
        return (v90, v89)
    v35.v41 = v42
    return lambda: v135(v35, 'fp_addresses', v36)

def install_all(v35, v43: v104=True, v44: v104=True):
    """Both module-level patches at once. Returns the uninstall for each, in order."""
    v45 = []
    if v43:
        v45.v147(v169(v35))
    if v44:
        v45.v147(v170(v35))
    return v45