"""
Stage 277 — Are word votes just the τ→1 limit of the ink, and is the region τ<1 worth anything?

263 and 261f left the retrieval question in an odd place. Votes beat the context mean 7x on the
open bank (top1 0.246 vs 0.034, median rank 76.5 vs 1036.5), so the shipped dense channel was
abandoned. But votes are exact string match, and exact match has one failure that no amount of
idf fixes: SILENCE. When no query word appears in the gold slot's write context the gold scores
zero, and a zero cannot be ranked - 266 showed what happens when you pretend it can.

The claim this stage tests is that "replace votes with a better vector" is the wrong framing,
because in the limit a perfectly discriminative vector IS a one-hot hash, i.e. IS votes. The
right framing is that votes are one endpoint of a family:

    score(q, slot) = Σ_{w∈q} idf(w) · max_{c∈slot} k(fp(w), fp(c))
    k(x, y) = relu(cos(x, y) − τ)^p

At τ→1 the kernel fires only on an identical fingerprint, the max collapses to an indicator, and
the expression is EXACTLY votes with idf - because content() dedups, so sum and max coincide.
At τ<1 the same expression is votes with a soft kernel: a declension, a typo, a compound form
lands near its neighbour instead of landing nowhere. Nothing is trained; τ is one scalar.

Four scorers over one bank, one item list, one seed:

    votes     exact postings + idf                       the incumbent (261f, 263)
    sum       cos(q, normalize(Σ idf(w)·fp(w)))          idf-weighted averaging, the honest
                                                         version of the mean 261f killed
    sum_sif   the same with the top principal component removed (all-but-the-top)
    maxsim    late interaction, the kernel above          votes generalised, zero trained params
    hybrid    votes + α·maxsim                            the deployable form if neither wins alone

and one diagnostic that has been owed since 264: split the queries where votes go SILENT on the
gold into those with a near-string route to the gold (shared 4-gram or edit distance ≤ 2) and
those with none, then ask what maxsim does on each half. The first half is the part the ink can
close by construction. The second half is paraphrase with no shared characters, and a character
encoder cannot close it - saying so with a number is the point.

Ranks use 266's correction throughout: a gold with zero score ranks LAST, not first, and n-way
is strict (gold must beat every distractor outright, ties are misses).

The first run answered INK_NO, and two things about that answer were wrong. Every gate read top1
on a 4338-slot open bank, which is the wrong instrument for a channel whose job is the tail:
hybrid was beating votes on the strict 20-way exam clean (0.432 vs 0.417) AND under character
noise (0.393 vs 0.352), and under that noise votes' median rank collapsed to the 4338 floor
while hybrid held 635. That IS 204's property; the gate simply was not looking at it. Each claim
is now gated on the metric the claim is about.

The second thing the run exposed is geometry. tau=0 and tau=0.3 scored identically, which can
only mean almost no pair of fingerprints sits below cos 0.3 - the ink is crowded into a narrow
cone and there is no range for a threshold to act on. sum_sif removed the common direction from
the SLOT matrix; the crowding is in the WORD matrix and was never touched. --whiten-fp K removes
the top-K principal directions of the vocabulary (or whitens its covariance with --whiten-mode
zca), fitted to the ink's own shape and to no label, and --cos-hist prints the two distributions
that decide whether any tau can work: cosine between unrelated words, and cosine to the nearest
OTHER word, which is what max_c returns on a miss.

That histogram came back and answered the question, though not in favour of whitening. Unrelated
words sit at cos 0.34 - the space is not uniformly crowded - but the NEAREST other word sits at
0.913 median and at exactly 1.0 for the top 5% of the vocabulary. A cosine of 1 between two
different strings is a collision, and whitening is a linear map: it cannot separate two vectors
that are equal.

The cause is upstream of everything this repo has measured. ArcEncoder pools its character
embeddings BEFORE the feed-forward and has no positional code at all, so fp(w) is a function of
w's normalised character histogram and of nothing else. listen and silent are bit-identical, and
so are ab and abab. That single fact explains 261's low-overlap 0.000, QUERY_MUST_BE_WORDS, the
collapse of the ctx_fp mean into near-constancy, and why maxsim at tau=0.6 fires on every slot:
the max over a slot's sixteen words clears 0.6 by chance alone. It also explains the one thing
the ink was always good at - a typo barely moves a histogram - and shows that the robustness and
the non-discrimination were never two properties, only one.

--fp-ngram N feeds the SAME frozen arc_enc overlapping character N-grams and pools those. Each
gram is still internally a bag, but the multiset of grams is not, so order returns without a
weight moving and with the arc_enc hash gate still passing. A typo corrupts N grams out of
len(w)-N+1, so 204's property survives. G_fp_no_collisions is the gate that says whether it
worked.

--fp-ngram 3 did half of that. Collisions fell from 0.098 to 0.030, so order genuinely returned,
but every cosine moved UP - unrelated pairs 0.329 -> 0.438, nearest 0.913 -> 0.942 - and the
separation shrank from 0.304 to 0.260. Pooling the grams is the same averaging one level down:
the mean of k vectors shrinks variance and drags everything toward the common direction. So two
last measurements before the ink is filed where it belongs.

--gram-maxsim N never averages a vector. A word is a SET of gram vectors; similarity is the mean
over the query's grams of the max over the candidate's grams, and the outer mean is over scalars
taken AFTER the maxima, which is not the same operation as averaging first and comparing once.

The spectrum print settles the width question for good. pooled = mean(emb(chars)) is a convex
combination of the alphabet's embedding rows, so its rank is bounded by the charset and not by
d, and ff is a fixed smooth map - it can fold that manifold but cannot add information to it. If
95% of the variance sits in a few dozen directions then 256 is already an order of magnitude
more room than the ink uses, and no width changes what it can tell apart.

  python _stage277_ink_revival.py --cos-hist                       # the shipped ink + spectrum
  python _stage277_ink_revival.py --cos-hist --gram-maxsim 3       # gram space, no averaging
  python _stage277_ink_revival.py [--smoke] [--gram-maxsim 3] [--tau 0.6] [--whiten-fp 8]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage261_nl_query import collect, ctx_words, jaccard
from _stage261f_word_votes import content, typo
v0 = v11('results')
v1 = v0 / 'stage277_decision.json'
v2 = v0 / 'stage277_mini.md'
v3 = v0 / '_stage277_log.txt'
v4 = v11('checkpoints/stage191_p1_curve.pt')
v5 = v11('data/_wikitext103_train.txt')
v6 = 2770

def log(v12: v34) -> None:
    v13 = v12 if v12.v256('\n') else v12 + '\n'
    try:
        v257(v13, end='', flush=True)
    except v130:
        v257(v13.v374('ascii', 'replace').v351('ascii'), end='', flush=True)
    v3.v258.v131(parents=True, exist_ok=True)
    with v3.v259('a', encoding='utf-8') as v132:
        v132.v260(v13)

def rank_of(v14: v199.v61, v15: v7) -> v7:
    """A gold that scored nothing ranks LAST. 266 read 0.477 top1 off 71 empty answers because
    this line said `1 + (v > gold).sum()` and every tie sat above the gold."""
    v16 = v133(v14[v15])
    if v16 <= 0.0:
        return v7(v14.v51[0])
    return 1 + v7((v14 > v16).v178())

def nway_strict(v14: v199.v61, v15: v7, v17: v33[v7]) -> v7:
    """Pessimistic n-way: the gold must beat every distractor OUTRIGHT. A silent gold loses."""
    v16 = v133(v14[v15])
    if v16 <= 0.0:
        return 0
    return v7(v261((v16 > v133(v14[v172]) for v172 in v17)))

def report(v18: v34, v19: v33[v199.v61], v20: v33[v8], v21: v33[v33[v7]], v22: v133, v23: v7) -> v8:
    v134, v135, v136, v137 = ([], [], [], [])
    for v14, v138, v17 in v139(v19, v20, v21):
        v24 = v262(v14, v138['cid'])
        v134.v263(v24)
        v135.v263(v320(v14, v138['cid'], v17))
        (v137 if v138['overlap'] > v22 else v136).v263(v7(v24 == 1))
    v24 = v199.v140(v134, dtype=v199.v264)
    v25 = {'top1': v133(v199.v173(v24 == 1)), 'mrr': v133(v199.v173(1.0 / v24)), 'median_rank': v133(v199.v310(v24)), f'acc_{v23}way_strict': v133(v199.v173(v135)), 'top1_low_overlap': v133(v199.v173(v136)) if v136 else v133('nan'), 'top1_high_overlap': v133(v199.v173(v137)) if v137 else v133('nan'), 'n': v234(v134)}
    v141(f'  [{v18}] ' + v350.v318(v25))
    return v25

def shares_ngram(v26: v34, v27: v34, v28: v7=4) -> v9:
    if v234(v26) < v28 or v234(v27) < v28:
        return v26 == v27
    v29 = {v26[v143:v143 + v28] for v143 in v180(v234(v26) - v28 + 1)}
    return v142((v27[v143:v143 + v28] in v29 for v143 in v180(v234(v27) - v28 + 1)))

def edit_le(v26: v34, v27: v34, v30: v7=2) -> v9:
    """Bounded Levenshtein: returns True iff distance ≤ k. Bails out as soon as it cannot be."""
    if v265(v234(v26) - v234(v27)) > v30:
        return False
    v31 = v33(v180(v234(v27) + 1))
    for v143, v144 in v145(v26, 1):
        v146 = [v143] + [0] * v234(v27)
        for v172, v266 in v145(v27, 1):
            v146[v172] = v179(v31[v172] + 1, v146[v172 - 1] + 1, v31[v172 - 1] + (v144 != v266))
        if v179(v146) > v30:
            return False
        v31 = v146
    return v31[-1] <= v30

def grams(v32: v34, v28: v7) -> v33[v34]:
    return [v32[v143:v143 + v28] for v143 in v180(v234(v32) - v28 + 1)] if v234(v32) >= v28 else [v32]

class Ink:
    """fp(w), with the option of writing the word in n-grams instead of in letters.

    ArcEncoder pools its character embeddings BEFORE the feed-forward and carries no positional
    code (_stage177_curve_bpe.py:183-187), so fp(w) is a function of w's normalised character
    histogram and nothing else. listen and silent are bit-identical; so are ab and abab. That is
    what the first histogram found - a nearest-neighbour cosine whose p95 is exactly 1.0 is not
    crowding, it is collisions, and no linear whitening separates equal vectors.

    It also explains why the ink is robust to typos: one changed letter barely moves a histogram.
    Robustness and non-discrimination are the same property here, not two.

    Feeding the SAME frozen encoder overlapping n-grams and pooling those restores order without
    touching a weight - arc_enc stays byte-identical and its hash gate still passes. Each gram is
    still internally a bag, but the multiset of grams is not: listen gives {lis,ist,ste,ten} and
    silent gives {sil,ile,len,ent}. A typo still only corrupts n grams out of len(w)-n+1, so the
    204 property survives.
    """

    def __init__(v147, v74: v225, v28: v7, v64):
        v147.v74, v147.v28, v147.v64 = (v74, v28, v64)
        v147.v148: v8[v34, v46.v10] = {}

    def __call__(v147, v149: v33[v34]) -> v46.v10:
        if v147.v28 <= 0:
            return v147.v74.v377(v149).v177(v147.v64).v133()
        v150 = [v32 for v32 in v149 if v32 not in v147.v148]
        if v150:
            v321, v322 = ([], [])
            for v32 in v150:
                v16 = v324(v32, v147.v28)
                v322.v263((v234(v321), v234(v16)))
                v321.v352(v16)
            v267 = v147.v74.v377(v321).v177(v147.v64).v133()
            for v32, (v26, v83) in v139(v150, v322):
                v147.v148[v32] = v270.v153(v267[v26:v26 + v83].v173(0), dim=-1)
        return v46.v268([v147.v148[v32] for v32 in v149], 0)

@v46.v41()
def spectrum(v35: v46.v10) -> v8:
    """How many dimensions does the ink actually occupy?

    pooled = mean(emb(chars)) is a convex combination of the alphabet's embedding rows, so its
    rank is bounded by the CHARSET, not by d, and ff is a fixed smooth map that can fold that
    manifold but cannot add information to it. If 95% of the variance lives in ~30 directions
    then d=256 is already an order of magnitude more room than the ink uses, and no width - 64,
    1024, anything - changes what it can tell apart. This prints the number that settles it.
    """
    v36 = (v35 - v35.v173(0, keepdim=True)).v151()
    v37 = v46.v323.v269(v36) ** 2
    v38 = v37 / v37.v178()
    v39 = v46.v152(v38, 0)
    v25 = {'d': v7(v35.v281(1)), 'top1_var_frac': v133(v38[0])}
    for v40 in (0.9, 0.95, 0.99):
        v25[f'dims_{v7(v40 * 100)}pct'] = v7((v39 < v40).v178()) + 1
    v25['participation_ratio'] = v133(1.0 / (v38 ** 2).v178())
    return v25

def apply_whiten(v42: v46.v10, v43, v44) -> v46.v10:
    v45 = v42 - v43
    return v270.v153(v45 if v44 is None else v45 @ v44, dim=-1)

class GramIndex:
    """MaxSim one level down: a word is a SET of gram vectors, never their mean.

    Averaging the grams cut the collision rate three-fold but pushed every cosine up - the mean
    of k vectors shrinks variance and drags everything toward the common direction, which is the
    same disease the whole-word recipe has, one level lower. This path never averages a vector:

        sim(q, c)      = mean over q's grams of  max over c's grams of  cos
        score(q, slot) = sum_q idf(q) * max over the slot's words of sim(q, c)

    The outer mean is over SCALARS, after the maxima, which is a different operation from
    averaging the vectors first and comparing once.
    """

    def __init__(v147, v154: v33[v34], v74: v225, v28: v7, v64, v155: v7, v156: v34):
        v147.v28, v147.v74, v147.v64 = (v28, v74, v64)
        v157: v8[v34, v7] = {}
        v271, v272 = ([], [])
        for v273, v32 in v145(v154):
            for v16 in v324(v32, v28):
                v271.v263(v273)
                v272.v263(v157.v367(v16, v234(v157)))
        v147.v158 = v274(v157, key=v157.v250)
        v147.v159 = v46.v275(v271, dtype=v46.v325, device=v64)
        v147.v160 = v46.v275(v272, dtype=v46.v325, device=v64)
        v147.v161 = v234(v154)
        v162 = v270.v153(v74.v377(v147.v158).v177(v64).v133(), dim=-1)
        v147.v163 = v276(v162)
        v147.v164 = v164(v162)
        v147.v43, v147.v44 = v277(v162, v155, v156)
        v147.v165 = v278(v162, v147.v43, v147.v44)
        v147.v166 = v276(v147.v165) if v155 else v147.v163
        v147.v167: v8[v34, v46.v10] = {}

    def wordsim(v147, v32: v34) -> v46.v10:
        """Similarity of one query word to every vocabulary word, [n_vocab]."""
        v168 = v147.v167.v250(v32)
        if v168 is not None:
            return v168
        v169 = v278(v270.v153(v147.v74.v377(v324(v32, v147.v28)).v177(v147.v64).v133(), dim=-1), v147.v43, v147.v44)
        v170 = v147.v165 @ v169.v40()
        v25 = v46.v279(v147.v161, dtype=v46.v326, device=v147.v64)
        v171 = v46.v280(v147.v161, dtype=v46.v326, device=v147.v64)
        for v172 in v180(v170.v281(1)):
            v171.v327(-1.0)
            v171.v328(0, v147.v159, v170[v147.v160, v172], reduce='amax')
            v25 += v171
        v25 /= v170.v281(1)
        if v234(v147.v167) < 200000:
            v147.v167[v32] = v25
        return v25

def fit_whiten(v35: v46.v10, v30: v7, v47: v34):
    """Fitted on the vocabulary matrix, applied to queries too. Returns (mean, [d,d] or None).

    abtt  remove the top-k principal directions - the common component every fingerprint shares
    zca   whiten the whole covariance, so no direction carries more variance than any other
    Both are zero-train: nothing here is fitted to a label, only to the shape of the ink itself.
    """
    v43 = v35.v173(0, keepdim=True)
    if v30 <= 0:
        return (v43 * 0.0, None)
    v36 = v35 - v43
    if v47 == 'zca':
        v174 = v36.v40() @ v36 / v313(1, v36.v281(0) - 1)
        v37, v282 = v46.v323.v283(v174.v151())
        v37 = v37.v284(1e-08)
        return (v43, (v282 @ v46.v378(v37.v379()) @ v282.v40()).v177(v35.v329))
    v48 = v46.v323.v330(v36, full_matrices=False).v48[:v30]
    return (v43, v46.v331(v35.v281(1), device=v35.v64, dtype=v35.v329) - v48.v40() @ v48)

@v46.v41()
def cos_stats(v35: v46.v10, v49: v7=200000, v50: v7=11) -> v8:
    """Two distributions, and the second is the one that decides whether a threshold can work.

    random_pair  cos between two unrelated words - where the kernel's noise floor sits
    nearest      cos to the closest OTHER word - what max_c actually returns on a miss

    If the nearest-neighbour cosine is ~1 for everything, max_c saturates and no tau separates a
    morphological variant from an unrelated word: the kernel is thresholding on nothing.
    """
    v28, v175 = v35.v51
    v16 = v46.v332(device='cpu').v176(v50)
    v26 = v46.v333(0, v28, (v49,), generator=v16).v177(v35.v64)
    v27 = v46.v333(0, v28, (v49,), generator=v16).v177(v35.v64)
    v52 = v26 != v27
    v53 = (v35[v26[v52]] * v35[v27[v52]]).v178(-1)
    v12 = v179(v28, 4000)
    v54 = v46.v353(v28, generator=v16)[:v12].v177(v35.v64)
    v55 = []
    for v56 in v180(0, v12, 512):
        v181 = v54[v56:v56 + 512]
        v170 = v35[v181] @ v35.v40()
        v170[v46.v354(v181.v368(), device=v35.v64), v181] = -2.0
        v55.v263(v170.v313(dim=1).v79)
    v57 = v46.v182(v55)
    v58 = [1, 5, 25, 50, 75, 95, 99]

    def pct(v183):
        v14 = v46.v285(v183.v133(), v46.v275([v201 / 100 for v201 in v58], device=v183.v64))
        return {'mean': v133(v183.v173()), 'std': v133(v183.v355()), **{f'p{v201}': v133(v40) for v201, v40 in v139(v58, v14)}}
    return {'random_pair': v286(v53), 'nearest_other': v286(v57), 'n_pairs': v7(v52.v178()), 'collision_rate': v133((v57 >= 0.9999).v133().v173()), 'n_words_sampled': v7(v12)}

def near_string(v59: v33[v34], v60: v33[v34]) -> v9:
    for v58 in v59:
        for v39 in v60:
            if v356(v58, v39) or v357(v58, v39):
                return True
    return False

class Index:
    """One flat posting list shared by every scorer, so an arm can never differ by its data.

    pos_word[i] / pos_slot[i] is the i-th (word, slot) incidence. votes, maxsim and the two dense
    arms are all functions of these two arrays plus a kernel.
    """

    def __init__(v147, v78: v33[v33[v34]], v74: v225, v64, v184: v7=20000, v155: v7=0, v156: v34='abtt', v185: v7=0, v87: v7=0):
        v147.v64 = v64
        v147.v186 = v234(v78)
        v147.v78 = v78
        v187: v8[v34, v33[v7]] = v287(v33)
        for v15, v232 in v145(v78):
            for v32 in v232:
                v187[v32].v263(v15)
        v147.v187 = v187
        v147.v154 = v274(v187)
        v147.v188 = {v32: v143 for v143, v32 in v145(v147.v154)}
        v147.v189 = {v32: 1.0 / v358.v141(2.0 + v234(v187[v32])) for v32 in v147.v154}
        v147.v190 = 1.0 / v358.v141(2.0)
        v288, v128 = ([], [])
        for v15, v232 in v145(v78):
            for v32 in v232:
                v288.v263(v147.v188[v32])
                v128.v263(v15)
        v147.v191 = v46.v275(v288, dtype=v46.v325, device=v64)
        v147.v192 = v46.v275(v128, dtype=v46.v325, device=v64)
        v147.v193 = v234(v288)
        v147.v74 = v74
        v147.v184 = v184
        v147.v185 = v185
        v147.v194 = v289(v74, v185, v64)
        v147.v195: v8[v9, v334] = {}
        v162 = v270.v153(v147.v194(v147.v154), dim=-1)
        v147.v163 = v276(v162)
        v147.v43, v147.v44 = v277(v162, v155, v156)
        v147.v155, v147.v156 = (v155, v156)
        v147.v196 = v147.v290(v162)
        v147.v166 = v276(v147.v196) if v155 else v147.v163
        v147.v164 = v164(v162)
        v147.v197 = v46.v275([v147.v189[v32] for v32 in v147.v154], dtype=v46.v326, device=v64)
        v147.v198 = None
        if v87 > 0:
            v147.v198 = v335(v147.v154, v74, v87, v64, v155, v156)
            v147.v163, v147.v166 = (v147.v198.v163, v147.v198.v166)
            v147.v164 = v147.v198.v164

    def _apply(v147, v42: v46.v10) -> v46.v10:
        v45 = v42 - v147.v43
        if v147.v44 is not None:
            v45 = v45 @ v147.v44
        return v270.v153(v45, dim=-1)

    def embed(v147, v149: v33[v34]) -> v46.v10:
        """Query words take the SAME transform the vocabulary was fitted with - an OOV form has
        to land in the whitened space or the kernel compares two different geometries."""
        return v147.v290(v270.v153(v147.v194(v149), dim=-1))

    def votes(v147, v59: v33[v34]) -> v199.v61:
        v14 = v199.v279(v147.v186, dtype=v199.v264)
        for v32 in v59:
            if v32 in v147.v187:
                v181 = v147.v189[v32]
                for v15 in v147.v187[v32]:
                    v14[v15] += v181
        return v14

    def maxsim(v147, v59: v33[v34], v200: v133, v201: v133) -> v199.v61:
        if not v59:
            return v199.v279(v147.v186, dtype=v199.v264)
        if v147.v198 is not None:
            v291 = v46.v268([v147.v198.v369(v32) for v32 in v59], 1)
        else:
            v291 = v147.v196 @ v147.v296(v59).v40()
        v202 = v46.v292(v291 - v200, min=0.0)
        if v201 != 1.0:
            v202 = v202.v336(v201)
        v25 = v46.v279(v147.v186, dtype=v46.v326, device=v147.v64)
        v171 = v46.v280(v147.v186, dtype=v46.v326, device=v147.v64)
        for v143, v32 in v145(v59):
            v171.v337()
            v171.v328(0, v147.v192, v202[v147.v191, v143], reduce='amax')
            v25 += v147.v189.v250(v32, v147.v190) * v171
        return v25.v151().v359().v293()

    def _dense_bank(v147, v203: v9):
        if v203 in v147.v195:
            return v147.v195[v203]
        v170 = v46.v279(v147.v186, v147.v196.v281(1), dtype=v46.v326, device=v147.v64)
        for v26 in v180(0, v147.v193, v147.v184):
            v27 = v179(v26 + v147.v184, v147.v193)
            v273 = v147.v191[v26:v27]
            v170.v338(0, v147.v192[v26:v27], v147.v196[v273] * v147.v197[v273].v297(1))
        v43 = v170.v173(0, keepdim=True)
        v204 = None
        if v203:
            v294 = v170 - v43
            v204 = v46.v323.v330(v294, full_matrices=False).v48[0]
            v170 = v294 - (v294 @ v204).v297(1) * v204.v297(0)
        v147.v195[v203] = (v270.v153(v170, dim=-1), v43, v204)
        return v147.v195[v203]

    def summed(v147, v59: v33[v34], v203: v9) -> v199.v61:
        v170, v43, v204 = v147.v295(v203)
        if not v59:
            return v199.v279(v147.v186, dtype=v199.v264)
        v205 = v147.v296(v59)
        v206 = v46.v275([v147.v189.v250(v32, v147.v190) for v32 in v59], dtype=v46.v326, device=v147.v64).v297(1)
        v58 = (v205 * v206).v178(0, keepdim=True)
        if v203:
            v58 = v58 - v43
            v58 = v58 - (v58 @ v204).v297(1) * v204.v297(0)
        v58 = v270.v153(v58, dim=-1)
        return (v170 @ v58.v40()).v380(1).v151().v359().v293()

    def repointed(v147, v109: v33[v7]):
        """Popularity floor: every posting keeps its word and its COUNT, only the slot it points
        at is permuted. A well-connected slot still collects hits, so this is a floor on
        connectivity rather than a random floor."""
        v207 = v236.v298(v236)
        v207.v300.v299(v147.v300)
        v201 = v46.v275(v109, dtype=v46.v325, device=v147.v64)
        v207.v192 = v201[v147.v192]
        v207.v187 = {v32: [v109[v39] for v39 in v14] for v32, v14 in v147.v187.v20()}
        v207.v195 = {}
        return v207

def main() -> v7:
    v62 = v301.v208()
    v62.v209('--smoke', action='store_true')
    v62.v209('--entities', type=v7, default=0)
    v62.v209('--distractor-entities', type=v7, default=0)
    v62.v209('--tau', type=v133, default=0.6, help='kernel threshold of the headline arm')
    v62.v209('--kernel-p', type=v133, default=1.0)
    v62.v209('--tau-sweep', type=v34, default='0.0,0.3,0.5,0.6,0.7,0.8,0.9,0.999', help='the curve that shows votes as the tau->1 endpoint of the same kernel')
    v62.v209('--alpha', type=v133, default=0.25, help='hybrid = votes + alpha * maxsim')
    v62.v209('--whiten-fp', type=v7, default=0, metavar='K', help="de-crowd the WORD fingerprint matrix before the kernel: remove the top-K principal directions (abtt) or whiten the covariance (zca). 0 = off. Zero-train - fitted to the ink's own shape, never to a label.")
    v62.v209('--whiten-mode', choices=['abtt', 'zca'], default='abtt')
    v62.v209('--fp-ngram', type=v7, default=0, metavar='N', help='write each word as overlapping character N-grams through the SAME frozen arc_enc instead of as one bag of letters. 0 = the shipped recipe. N=3 is the first setting where an anagram stops colliding.')
    v62.v209('--gram-maxsim', type=v7, default=0, metavar='N', help='score at the GRAM level and never average a vector: a word is a set of N-gram vectors, similarity is mean-over-query-grams of max-over-candidate-grams. Supersedes --fp-ngram for the kernel arm.')
    v62.v209('--cos-hist', action='store_true', help='print the cosine histogram of the word matrix and exit without scoring')
    v62.v209('--n-way', type=v7, default=20)
    v62.v209('--typo-rate', type=v133, default=0.15)
    v62.v209('--no-typo-arm', action='store_true')
    v63 = v62.v210()
    v3.v211('', encoding='utf-8')
    v64 = v46.v64('cuda' if v46.v360.v339() else 'cpu')
    v65 = v302.v212(v6)
    v46.v176(v6)
    v66 = v213.v213()
    v67 = v63.v214 or (60 if v63.v216 else 400)
    v68 = v63.v215 or (400 if v63.v216 else 4000)
    v69 = 3000 if v63.v216 else 25000
    v70 = [v133(v183) for v183 in v63.v361.v340(',') if v183.v307()]
    v141(f'Stage277 ink revival start {v372.v365(v373.v366).v317()} device={v64} tau={v63.v200} p={v63.v252} alpha={v63.v253}')
    v217, v217, v218, v219 = v220()
    v71 = v303.v221(v34(v341.v304))
    v72 = v342(v219, v71.v362()).v177(v64)
    v72.v222(v46.v343(v4, map_location=v64, weights_only=False)['model'])
    v72.v223()
    for v73 in v72.v224():
        v73.v305(False)
    v74 = v225(v72, v218, v64)
    with v5.v259('r', encoding='utf-8', errors='ignore') as v132:
        v226 = v132.v306(3000000 if v63.v216 else 20000000)
    v75 = [v344.v307() for v344 in v226.v340('\n') if 80 <= v234(v344.v307()) <= 400][:v69]
    v76 = v227(v75, v74)
    v77 = v274(v76)[:v67]
    v65.v228(v77)
    v141(f'  entities with >=2 natural mentions: {v234(v76)} (using {v234(v77)})')
    if v234(v77) < 16:
        v141('  not enough multi-mention entities')
        return 1
    v78: v33[v33[v34]] = []
    v79: v33[v34] = []
    v20: v33[v8] = []
    for v80 in v77:
        v229 = v76[v80]
        v26, v27 = (v229[0], v229[1])
        v230 = v26['line'][v313(0, v26['start'] - 140):v179(v234(v26['line']), v26['end'] + 140)]
        v231 = v27['line'][v313(0, v27['start'] - 200):v27['start']].v307()
        v232 = v308(v230, exclude=v80)
        v233 = v308(v231, exclude=v80)
        if v234(v232) < 4 or v234(v233) < 4:
            continue
        v15 = v234(v79)
        v79.v263(v80)
        v78.v263(v232)
        v20.v263({'ent': v80, 'cid': v15, 'qwords': v233, 'overlap': v363(v370(v230, v80), v370(v231, v80))})
    v81 = v234(v79)
    v82 = v235(v79)
    for v83 in v75:
        if v234(v79) >= v81 + v68:
            break
        for v12 in v345.v309(v83):
            v80 = v12.v346(1)
            if v234(v80) < 5 or v80 in v82:
                continue
            v136, v137 = (v313(0, v12.v375() - 140), v179(v234(v83), v12.v376() + 140))
            v232 = v308(v83[v136:v137], exclude=v80)
            if v234(v232) < 4:
                continue
            v79.v263(v80)
            v78.v263(v232)
            v82.v347(v80)
            if v234(v79) >= v81 + v68:
                break
    if v234(v20) < 16:
        v141('  not enough usable pairs')
        return 1
    v84 = v236(v78, v74, v64, whiten_k=v63.v88, whiten_mode=v63.v156, ngram=v63.v254, gram_maxsim=v63.v87)
    v85 = v133(v199.v310([v138['overlap'] for v138 in v20]))
    v141(f'  slots={v84.v186} ({v81} asked + {v84.v186 - v81} distractor) | vocab={v234(v84.v154)} postings={v84.v193} | eval={v234(v20)} overlap median={v85:.3f} | ({v213.v213() - v66:.0f}s)')
    v86 = 'whole word (shipped)' if not v63.v254 else f'{v63.v254}-gram mean'
    if v63.v87:
        v86 = f'{v63.v87}-gram SET (no vector averaging)'
    v141(f'  fp unit = {v86}' + (f' | gram vocab={v234(v84.v198.v158)}' if v84.v198 is not None else ''))
    v141(f'  fp spectrum {v350.v318(v84.v164)}')
    v141(f"  fp cos raw       random_pair {v350.v318(v84.v163['random_pair'])}")
    v141(f"  fp cos raw       nearest     {v350.v318(v84.v163['nearest_other'])}")
    v141(f"  fp collisions (nearest cos >= 0.9999): {v84.v163['collision_rate']:.4f} of {v84.v163['n_words_sampled']} sampled words")
    if v63.v88:
        v141(f"  fp cos whitened  random_pair {v350.v318(v84.v166['random_pair'])} ({v63.v156} k={v63.v88})")
        v141(f"  fp cos whitened  nearest     {v350.v318(v84.v166['nearest_other'])}")
    if v63.v89:
        v141('  --cos-hist: geometry only, no scoring')
        return 0
    v90 = v302.v212(v6 + 5)
    v21 = [[v172 for v172 in v90.v371(v180(v84.v186), v179(v63.v23 * 3, v84.v186)) if v172 != v138['cid']][:v63.v23 - 1] for v138 in v20]
    v91 = [v84.v311(v138['qwords']) for v138 in v20]
    v92 = v237('votes', v91, v20, v21, v85, v63.v23)
    v93 = [v84.v312(v138['qwords'], v63.v200, v63.v252) for v138 in v20]
    v94 = v237(f'maxsim tau={v63.v200}', v93, v20, v21, v85, v63.v23)
    v95 = v237('sum', [v84.v348(v138['qwords'], False) for v138 in v20], v20, v21, v85, v63.v23)
    v96 = v237('sum_sif', [v84.v348(v138['qwords'], True) for v138 in v20], v20, v21, v85, v63.v23)
    v97 = v313((v133(v14.v313()) for v14 in v91 if v14.v281), default=1.0) or 1.0
    v98 = v313((v133(v14.v313()) for v14 in v93 if v14.v281), default=1.0) or 1.0
    v99 = [v14 + v63.v253 * (v97 / v98) * v12 for v14, v12 in v139(v91, v93)]
    v100 = v237(f'hybrid a={v63.v253}', v99, v20, v21, v85, v63.v23)
    v101 = {}
    for v40 in v70:
        v238 = [v84.v312(v138['qwords'], v40, v63.v252) for v138 in v20]
        v239 = v199.v140([v262(v14, v138['cid']) for v14, v138 in v139(v238, v20)], dtype=v199.v264)
        v101[f'{v40:g}'] = {'top1': v133(v199.v173(v239 == 1)), 'median_rank': v133(v199.v310(v239)), 'mrr': v133(v199.v173(1.0 / v239))}
        v141(f'  [sweep tau={v40:g}] ' + v350.v318(v101[f'{v40:g}']))
    v102 = f'{v313(v70):g}'
    v103 = v265(v101[v102]['top1'] - v92['top1']) <= 0.02 if v70 else False
    v104 = [v30 for v30, (v14, v138) in v145(v139(v91, v20)) if v14[v138['cid']] <= 0.0]
    v105 = [v30 for v30 in v104 if v349(v20[v30]['qwords'], v78[v20[v30]['cid']])]
    v106 = v235(v105)
    v107 = [v30 for v30 in v104 if v30 not in v106]

    def sub(v18, v240):
        if not v240:
            return {'n': 0}
        v241 = v199.v140([v262(v93[v30], v20[v30]['cid']) for v30 in v240], dtype=v199.v264)
        v242 = v199.v140([v262(v99[v30], v20[v30]['cid']) for v30 in v240], dtype=v199.v264)
        v243 = {'n': v234(v240), 'maxsim_top1': v133(v199.v173(v241 == 1)), 'maxsim_median_rank': v133(v199.v310(v241)), 'maxsim_top10': v133(v199.v173(v241 <= 10)), 'hybrid_top1': v133(v199.v173(v242 == 1)), 'votes_top1': 0.0, 'votes_median_rank': v133(v84.v186)}
        v141(f'  [silent/{v18}] ' + v350.v318(v243))
        return v243
    v108 = {'n_queries': v234(v20), 'n_votes_silent': v234(v104), 'frac_votes_silent': v234(v104) / v313(1, v234(v20)), 'near_string': v314('near_string', v105), 'purely_semantic': v314('purely_semantic', v107)}
    v109 = v33(v180(v84.v186))
    v302.v212(v6 + 7).v228(v109)
    v110 = v84.v244(v109)
    v111 = [v110.v311(v138['qwords']) for v138 in v20]
    v112 = [v110.v312(v138['qwords'], v63.v200, v63.v252) for v138 in v20]
    v113 = {'votes_top1': v133(v199.v173([v262(v14, v138['cid']) == 1 for v14, v138 in v139(v111, v20)])), 'maxsim_top1': v133(v199.v173([v262(v14, v138['cid']) == 1 for v14, v138 in v139(v112, v20)]))}
    v141('  [popularity floor] ' + v350.v318(v113))
    v114 = {}
    if not v63.v315 and v63.v316 > 0:
        v245 = v302.v212(v6 + 3)
        v246 = [[v364(v32, v63.v316, v245) for v32 in v138['qwords']] for v138 in v20]
        v247 = [v84.v311(v58) for v58 in v246]
        v248 = [v84.v312(v58, v63.v200, v63.v252) for v58 in v246]
        v249 = [v14 + v63.v253 * (v97 / v98) * v12 for v14, v12 in v139(v247, v248)]
        v114 = {'typo_rate': v63.v316, 'votes': v237('votes+typo', v247, v20, v21, v85, v63.v23), 'maxsim': v237('maxsim+typo', v248, v20, v21, v85, v63.v23), 'hybrid': v237('hybrid+typo', v249, v20, v21, v85, v63.v23)}
    v115 = f'acc_{v63.v23}way_strict'
    v116 = v113['maxsim_top1'] <= 0.02 and v113['votes_top1'] <= 0.02
    v117 = v9(v103)
    v118 = v94['top1'] >= v92['top1'] + 0.02
    v119 = v100['top1'] >= v92['top1'] + 0.02 or v100[v115] >= v92[v115] + 0.01
    v120 = v108['near_string'].v250('maxsim_median_rank', v133(v84.v186))
    v121 = v108['near_string'].v250('n', 0) > 0 and v120 <= 0.75 * v84.v186
    v122 = v108['near_string'].v250('maxsim_top10', 0.0) >= 0.2
    v123 = v313(v95['top1'], v96['top1']) < v94['top1']
    v124 = not v114 or (v114['maxsim']['median_rank'] < v114['votes']['median_rank'] or v114['hybrid'][v115] > v114['votes'][v115])
    v125 = v84.v166['nearest_other']['p50'] < 0.95 and v84.v166['random_pair']['p95'] < 0.8
    v126 = v84.v166['collision_rate'] <= 0.01
    if not v116:
        v251 = 'INK_INVALID_FLOOR'
    elif v118 and v117 and v121:
        v251 = 'INK_REPLACES_VOTES'
    elif v119 and v124:
        v251 = 'INK_FILLS_SILENCE'
    elif v121 or v124:
        v251 = 'INK_SIGNAL_ONLY'
    elif not (v125 and v126):
        v251 = 'INK_CROWDED'
    else:
        v251 = 'INK_NO'
    v25 = {'stage': 277, 'overall': v251, 'trained_parameters': 0, 'tau': v63.v200, 'kernel_p': v63.v252, 'alpha': v63.v253, 'n_way': v63.v23, 'whiten_fp': v63.v88, 'whiten_mode': v63.v156, 'fp_ngram': v63.v254, 'gram_maxsim': v63.v87, 'fp_unit': v86, 'slots': v84.v186, 'asked': v81, 'distractors': v84.v186 - v81, 'vocab': v234(v84.v154), 'postings': v84.v193, 'overlap_median': v85, 'gates': {'G_causal_popularity_floor': v116, 'G_kernel_reduces_to_votes': v117, 'G_maxsim_beats_votes': v118, 'G_hybrid_beats_votes': v119, 'G_nonzero_where_votes_silent': v121, 'G_near_string_hole_closed': v122, 'G_sum_still_loses': v123, 'G_typo_favours_ink': v124, 'G_fp_has_dynamic_range': v125, 'G_fp_no_collisions': v126}, 'summary': {'fp_geometry': {'raw': v84.v163, 'spectrum': v84.v164, 'whitened': v84.v166 if v63.v88 else None}, 'votes': v92, 'maxsim': v94, 'sum': v95, 'sum_sif': v96, 'hybrid': v100, 'tau_sweep': v101, 'silence': v108, 'popularity_floor': v113, 'typo_arm': v114, 'reference_261f': {'votes_top1': 0.246, 'votes_median_rank': 76.5, 'mean_top1': 0.034, 'mean_median_rank': 1036.5}}, 'note': "Votes are not an alternative to the fingerprints, they are the tau->1 limit of score(q,slot) = sum_w idf(w) * max_c relu(cos(fp(w),fp(c)) - tau)^p. content() dedups a slot's words, so at tau->1 the max collapses to an indicator and the expression is votes with idf, exactly - G_kernel_reduces_to_votes checks that empirically rather than on paper. The question the sweep answers is whether the interior tau<1 buys anything the endpoint does not. Ranks use 266's correction (a zero-scoring gold ranks LAST) and n-way is strict, because both failures previously read as accuracy. The silence split is the diagnostic owed since 264: near_string is the half a character encoder can reach by construction, purely_semantic is the half it cannot, and reporting the second honestly is what keeps this from being a claim about meaning.", 'timestamp': v372.v365(v373.v366).v317(), 'wall_s': v213.v213() - v66}
    v0.v131(parents=True, exist_ok=True)
    v1.v211(v350.v318(v25, indent=2), encoding='utf-8')
    v127 = v108['near_string']
    v128 = v108['purely_semantic']
    v2.v211(f"# Stage 277 ink revival: votes as the tau->1 limit\n\n**{v251}** slots={v84.v186} vocab={v234(v84.v154)} eval={v234(v20)} tau={v63.v200} trained params **0**\n\n| arm | top1 | median rank | {v63.v23}-way strict |\n|---|---|---|---|\n| votes (incumbent) | {v92['top1']:.3f} | {v92['median_rank']:.0f} | {v92[f'acc_{v63.v23}way_strict']:.3f} |\n| sum (idf-weighted) | {v95['top1']:.3f} | {v95['median_rank']:.0f} | {v95[f'acc_{v63.v23}way_strict']:.3f} |\n| sum + all-but-top | {v96['top1']:.3f} | {v96['median_rank']:.0f} | {v96[f'acc_{v63.v23}way_strict']:.3f} |\n| **maxsim** | **{v94['top1']:.3f}** | {v94['median_rank']:.0f} | {v94[f'acc_{v63.v23}way_strict']:.3f} |\n| hybrid (a={v63.v253}) | {v100['top1']:.3f} | {v100['median_rank']:.0f} | {v100[f'acc_{v63.v23}way_strict']:.3f} |\n\n- kernel reduces to votes at tau={v102}: **{v117}** (maxsim {v101.v250(v102, {}).v250('top1', v133('nan')):.3f} vs votes {v92['top1']:.3f})\n- votes silent on gold: **{v108['n_votes_silent']}/{v234(v20)}** ({v108['frac_votes_silent']:.3f}) -> near-string {v127.v250('n', 0)}, purely semantic {v128.v250('n', 0)}\n- on the near-string half maxsim top1 {v127.v250('maxsim_top1', v133('nan')):.3f}, top10 {v127.v250('maxsim_top10', v133('nan')):.3f}; on the semantic half top10 {v128.v250('maxsim_top10', v133('nan')):.3f}\n- popularity floor: votes {v113['votes_top1']:.3f}, maxsim {v113['maxsim_top1']:.3f}\n- fp geometry raw: random pair p50 {v84.v163['random_pair']['p50']:.3f} / p95 {v84.v163['random_pair']['p95']:.3f}, nearest-other p50 {v84.v163['nearest_other']['p50']:.3f}" + (f" -> whitened ({v63.v156} k={v63.v88}) random p50 {v84.v166['random_pair']['p50']:.3f} / p95 {v84.v166['random_pair']['p95']:.3f}, nearest p50 {v84.v166['nearest_other']['p50']:.3f}" if v63.v88 else '') + f"; range usable **{v125}**\n- fp unit: **{v86}**, collisions (nearest cos >= 0.9999) **{v84.v166['collision_rate']:.4f}** (raw {v84.v163['collision_rate']:.4f})\n- spectrum: d={v84.v164['d']}, 95% of variance in **{v84.v164['dims_95pct']}** dims, participation ratio {v84.v164['participation_ratio']:.1f} - width is not the bottleneck if this is far below d\n" + (f"- typo {v63.v316}: votes {v114['votes']['top1']:.3f} -> maxsim {v114['maxsim']['top1']:.3f}, hybrid {v114['hybrid']['top1']:.3f}\n" if v114 else ''), encoding='utf-8')
    v141(v350.v318({'overall': v251, 'gates': v25['gates']}, indent=2))
    return 0
if v129 == '__main__':
    raise v255(v319())