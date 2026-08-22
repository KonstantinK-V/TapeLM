"""
Stage 279 — Writing is a decision, and the disagreements it finds are the product.

Every stage so far read a tape somebody else built. The write side was named as the open half in
GOAL.md and never implemented, and it is not an append:

    WRITE     the address is empty                     -> a new slot
    CONFIRM   same address, same value                  -> support += 1, nothing is stored
    DISPUTE   same address, a different value           -> a second slot, both kept

Three things follow that no separate mechanism has to provide. The tape stops growing on
repetition, because a repeat is a counter and not a slot. Witnesses stop being a template: a
value said by four sources and contradicted by two IS 4-2, arrived at from the data, which is
exactly the structure 271-278 had to manufacture with WITNESS_TMPL. And a support count is a
reliability estimate that costs nothing to compute.

That last one is the reason this stage is worth running before the harder open items. A tape
built this way is a contradiction detector with provenance attached, and it audits two things
that are otherwise awkward to audit:

    --source corpus   every address where the text disagrees with itself, and who said what
    --source model    the same subject sampled N times from an LM at temperature. The share of
                      samples that agree is the model's uncertainty measured BEHAVIOURALLY, from
                      what it says rather than from its logits, and the corpus majority is
                      available as a check on whether that number means anything.

The calibration gate is the one that matters. If self-consistency predicts agreement with the
corpus, then a tape written this way estimates reliability with no labels at inference time. If
it does not, the support count is decoration and this stage says so.

Matching is exact by default and optionally soft. Soft is where the ink does the one job 277
left it: an address is a coordinate it created at write time, so a value written as "Paris" and
a value written as "paris ." should land together. The soft arm is guarded by a replay control -
writing a corpus twice must produce zero disputes, whatever the matcher - because a matcher that
invents disagreements is worse than one that misses them.

  python _stage279_write_decision.py [--smoke]
  python _stage279_write_decision.py --soft-match 0.9
  python _stage279_write_decision.py --source model --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE
v0 = v12('results')
v1 = v12('checkpoints/stage191_p1_curve.pt')
v2 = v12('data/_wikitext103_train.txt')
v3 = 279
v4 = v0 / '_stage279_log.txt'
v5 = 'The {S} was'

def log(v13: v18) -> None:
    v14 = v13 if v13.v188('\n') else v13 + '\n'
    try:
        v189(v14, end='', flush=True)
    except v95:
        v189(v14.v306('ascii', 'replace').v271('ascii'), end='', flush=True)
    v4.v54.v96(parents=True, exist_ok=True)
    with v4.v190('a', encoding='utf-8') as v97:
        v97.v191(v14)
v15, v16, v17 = ('WRITE', 'CONFIRM', 'DISPUTE')

class Tape:
    """Addresses to slots, where a slot is a value with a support count and its sources.

    The whole write side is `decide`. Everything downstream - deduplication, the witness
    structure, the confidence estimate - is a consequence of it rather than a separate feature.
    """

    def __init__(v98, v40: v171 | None=None, v61: v125=0.0):
        v98.v99: v122[v18, v122[v18, v122]] = v123(v122)
        v98.v40 = v40
        v98.v61 = v61
        v98.v100 = v120()
        v98.v101: v114[v251[v18, v18, v18, v18]] = []

    def _match(v98, v102: v18, v103: v18) -> v18 | None:
        """The value already on the tape that this one is the same as, or None."""
        v104 = v98.v99[v102]
        if v103 in v104:
            return v103
        if v98.v61 <= 0.0 or v98.v40 is None or (not v104):
            return None
        v105 = v114(v104)
        v106 = v211.v132(v98.v40.v261([v103] + v105).v125(), dim=-1)
        v107 = (v106[1:] @ v106[0]).v192()
        v108 = v11(v290.v252(v107))
        return v105[v108] if v107[v108] >= v98.v61 else None

    def decide(v98, v102: v18, v103: v18, v109: v18) -> v18:
        v110 = v98.v193(v102, v103)
        if not v98.v99[v102]:
            v113 = v15
        elif v110 is not None:
            v113 = v16
        else:
            v113 = v17
        v111 = v110 if v110 is not None else v103
        v112 = v98.v99[v102].v194(v111, {'support': 0, 'sources': [], 'forms': v26()})
        v112['support'] += 1
        v112['forms'].v195(v103)
        if v177(v112['sources']) < 8:
            v112['sources'].v196(v109)
        v98.v100[v113] += 1
        v98.v101.v196((v102, v103, v113, v111))
        return v113

    def n_slots(v98) -> v11:
        return v197((v177(v115) for v115 in v98.v99.v142()))

    def disputed(v98) -> v114[v18]:
        return [v38 for v38, v115 in v98.v99.v136() if v177(v115) > 1]

    def verdict(v98, v102: v18):
        """majority value and its consistency, or None where the tape never settled."""
        v115 = v98.v99.v253(v102) or {}
        if not v115:
            return (None, 0.0, 0)
        v116 = v198(v115.v136(), key=lambda v300: -v300[1]['support'])
        v117 = v197((v238['support'] for v238 in v115.v142()))
        v199, v79 = (v116[0][1]['support'], v116[1][1]['support'] if v177(v116) > 1 else 0)
        if v199 == v79:
            return (None, v199 / v117, v117)
        return (v116[0][0], v199 / v117, v117)
v6 = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'meanwhile', 'later', 'between', 'however', 'although', 'during', 'after', 'before', 'since', 'while', 'these', 'those', 'there', 'their', 'another', 'several', 'following', 'according', 'despite', 'instead', 'finally', 'initially', 'originally'}
v7 = v118.v19('[A-Za-z][a-z]{2,}')
v8 = {'was', 'were', 'is', 'are', 'became', 'become', 'named', 'called', 'known', 'appointed', 'elected', 'born', 'died', 'founded', 'located', 'situated', 'built', 'released', 'married', 'served', 'won', 'wrote', 'directed', 'produced', 'played', 'signed', 'joined', 'replaced', 'succeeded', 'awarded', 'seat', 'capital', 'member', 'part', 'author', 'director'}
v9 = v118.v19('\\b(?:[A-Z][a-z]{2,}(?:\\s+(?=[A-Z]))?)+')

def sentence_initial(v20: v18) -> v10:
    """A capitalised word that opens a sentence is capitalised by grammar, not by being a name."""
    v21 = v20.v119()
    return not v21 or v21[-1] in '.!?”"' or v21.v188(' .')

def common_nouns(v22, v23: v11=3) -> v26[v18]:
    """Words the corpus also uses in lowercase are common nouns, whatever their capitalisation
    at the start of a phrase. Without this the anchors are cardinal, county, news and energy -
    titles and categories rather than subjects - and what gets addressed is the collocation of
    two words rather than a fact about somebody."""
    v24 = v120()
    for v25 in v22:
        for v121 in v118.v200('\\b[a-z]{3,}\\b', v25):
            v24[v121] += 1
    return {v121 for v121, v130 in v24.v136() if v130 >= v23}

def corpus_assertions(v22, v27, v28: v11, v29: v11, v30: v18, v31: v11=3, v32: v26[v18] | None=None, v33: v10=False, v34: v11=1):
    """An assertion is a natural (address, value) pair from real text, nothing authored.

    The first run addressed by the anchor word alone and the result was meaningless: "magazine"
    collected November, Constantine and Keats, and "chris" collected Mullin, Meanwhile and
    Turner. An anchor is not an address, because it carries no RELATION - every sentence in the
    corpus that happens to contain the word lands on it, and a dispute rate of 0.93 measures the
    corpus's vocabulary rather than its contradictions.

    The address is now the anchor together with the content words that stand between it and the
    value, which is what says WHAT is being asserted. "Chris Mullin" and "Chris ... Meanwhile"
    stop being the same address. --address anchor restores the old behaviour for comparison.

    Values are also filtered: a capitalised word that opens a sentence is capitalised by grammar,
    and months and discourse markers are not entities however the regex feels about them.
    """
    v35: v122[v18, v114[v251[v18, v18]]] = v123(v114)
    for v25 in v22:
        for v13 in v254.v201(v25):
            v202 = v13.v255(1)
            if v177(v202) < 5 or v202.v258() in v6:
                continue
            v20 = v25[:v13.v301()]
            if v256(v20):
                continue
            v203 = v240(0, v13.v301() - 140)
            v204 = [v38 for v38 in v9.v201(v25[v203:v13.v301()]) if v38.v255(0).v257() and v38.v255(0).v257() != v202]
            if not v204:
                continue
            v38 = v204[-1]
            v137 = v38.v255(0).v257()
            v205 = v137.v260()[-1].v258()
            if v32 and v205 in v32 and (v177(v137.v260()) == 1):
                continue
            v206 = v25[v203 + v38.v307():v13.v301()]
            v207 = [v121.v258() for v121 in v7.v200(v206) if v121.v258() not in v6]
            if v177(v207) < v34:
                continue
            if v33 and (not v302((v121 in v8 for v121 in v207))):
                continue
            v208 = v207[-v31:]
            v209 = v137.v258() if v30 == 'anchor' else v137.v258() + '|' + ' '.v289(v208)
            v35[v209].v196((v202, v25[v203:v13.v307() + 20]))
    v36 = [v38 for v38, v115 in v35.v136() if v177(v115) >= v29]
    v27.v124(v36)
    v36 = v36[:v28]
    v37 = []
    for v38 in v36:
        for v49, (v202, v259) in v135(v35[v38]):
            v37.v196({'address': v38, 'value': v202, 'source': f'wiki:{v38}:{v49}', 'ctx': v259})
    return (v37, v36)

def fp_addresses(v39, v40, v41: v125, v42: v11, v29: v11, v43: v18='two'):
    """Address by the coordinate the ink writes, verified by the words the contexts share.

    A string address demands the same anchor and the same relation words in the same forms, so a
    fact stated twice in two phrasings lands on two addresses and never gets to CONFIRM. The key
    is what the ink is actually for: norm(fp(anchor) + ctx_fp(context)) is a coordinate that a
    different surface form still reproduces.

    277 also showed what happens if that coordinate is trusted alone - nearest_other sits at 0.91
    and unrelated words are one hop apart. So it PROPOSES and the discrete channel disposes: two
    assertions may share an address only if the key cosine clears tau AND their write contexts
    share at least min_overlap content words. Approximate where it proposes, exact where it
    decides.
    """
    v105, v126, v127, v128 = ([], [], [], [])
    for v38 in v39:
        v129 = v38['address'].v260('|')[0]
        v130 = v40.v210(v38['ctx'], exclude=v38['value'])
        v131 = v40.v261([v129])[0]
        v105.v196(v211.v132(v131 + v130 if v130 is not None else v131, dim=-1))
        v126.v196(v211.v132(v131, dim=-1))
        v127.v196(v211.v132(v130, dim=-1) if v130 is not None else v211.v132(v131, dim=-1))
        v128.v196({v121.v258() for v121 in v7.v200(v38['ctx']) if v121.v258() not in v6} - {v38['value'].v258()})
    v44 = v211.v132(v228.v291(v105).v125(), dim=-1)
    v45 = v211.v132(v228.v291(v126).v125(), dim=-1)
    v46 = v211.v132(v228.v291(v127).v125(), dim=-1)
    v47: v114[v114[v11]] = []
    v48 = [-1] * v177(v39)
    for v49 in v133(v177(v39)):
        v212, v213 = (-1, v41)
        for v134, v214 in v135(v47):
            v215 = v228.v219(v214, device=v44.v64)
            v107 = (v228.v221(v45[v215] @ v45[v49], v46[v215] @ v46[v49]) if v43 == 'two' else v44[v215] @ v44[v49]).v192()
            for v108, v262 in v135(v107):
                if v262 >= v213 and v177(v128[v49] & v128[v214[v108]]) >= v42:
                    v212, v213 = (v134, v262)
        if v212 < 0:
            v48[v49] = v177(v47)
            v47.v196([v49])
        else:
            v48[v49] = v212
            v47[v212].v196(v49)
    v50: v122[v11, v114[v11]] = v123(v114)
    for v49, v134 in v135(v48):
        v50[v134].v196(v49)
    v37, v36 = ([], [])
    for v134, v47 in v50.v136():
        if v177(v47) < v29:
            continue
        v137 = f"fp{v134}:{v39[v47[0]]['address']}"
        v36.v196(v137)
        for v147, v49 in v135(v47):
            v38 = v122(v39[v49])
            v38['straddr'] = v38['address']
            v38['address'] = v137
            v38['source'] = f'wiki:{v137}:{v147}'
            v37.v196(v38)
    return (v37, v36)

def fp_addresses_anchor_block(v39, v40, v41: v125, v42: v11, v29: v11, v51: v18='or', v52: v18='ctx'):
    """Merge only inside one normalized anchor; never across names.

    283 showed the global two-channel tau trades foreign for capacity because cosine density
    grows with corpus N. Bare anchors barely crowd (0.018) while context flattens keys, so the
    hard cut belongs on the name: different anchors never share an address. Soft linking runs
    only among mentions of that name, where N is local and a fixed tau does not need retuning.

    link:
      or  — 283b: overlap OR two-channel (recovered capacity, but glued distinct relations → ties).
      and — 283c: overlap AND two-channel. Same numbers as write2, scoped to the block.
    overlap_from:
      ctx — content words of the whole write context (283b).
      rel — words from the address relation tail after '|', so born/died do not share an edge
            through ambient English.

    min_mentions is applied AFTER clustering. Callers should extract with min_mentions=1 so
    paraphrases that each appear once can still form an address once merged.
    """
    if not v39:
        return ([], [])
    if v51 not in ('or', 'and'):
        raise v216(f'unknown link={v51!r}')
    if v52 not in ('ctx', 'rel'):
        raise v216(f'unknown overlap_from={v52!r}')
    v126, v127, v128, v138 = ([], [], [], [])
    for v38 in v39:
        v129 = v38['address'].v260('|')[0]
        v138.v196(v129)
        v130 = v40.v210(v38['ctx'], exclude=v38['value'])
        v131 = v40.v261([v129])[0]
        v126.v196(v211.v132(v131, dim=-1))
        v127.v196(v211.v132(v130, dim=-1) if v130 is not None else v211.v132(v131, dim=-1))
        if v52 == 'rel':
            v208 = (v38['address'].v260('|', 1) + [''])[1]
            v128.v196({v121.v258() for v121 in v7.v200(v208) if v121.v258() not in v6})
        else:
            v128.v196({v121.v258() for v121 in v7.v200(v38['ctx']) if v121.v258() not in v6} - {v38['value'].v258()})
    v45 = v211.v132(v228.v291(v126).v125(), dim=-1)
    v46 = v211.v132(v228.v291(v127).v125(), dim=-1)
    v53: v122[v18, v114[v11]] = v123(v114)
    for v49, v139 in v135(v138):
        v53[v139].v196(v49)
    v54 = v114(v133(v177(v39)))

    def find(v140: v11) -> v11:
        while v54[v140] != v140:
            v54[v140] = v54[v54[v140]]
            v140 = v54[v140]
        return v140

    def union(v140: v11, v141: v11) -> None:
        v217, v218 = (v263(v140), v263(v141))
        if v217 != v218:
            v54[v218] = v217
    for v55 in v53.v142():
        if v177(v55) < 2:
            continue
        v143 = v228.v219(v55, device=v45.v64)
        v144 = v45[v143] @ v45[v143].v220
        v145 = v46[v143] @ v46[v143].v220
        v146 = v228.v221(v144, v145)
        v147 = v177(v55)
        for v38 in v133(v147):
            for v222 in v133(v38 + 1, v147):
                v49, v108 = (v55[v38], v55[v222])
                v264 = v177(v128[v49] & v128[v108]) >= v42
                v265 = v125(v146[v38, v222]) >= v41
                v266 = v264 or v265 if v51 == 'or' else v264 and v265
                if v266:
                    v303(v49, v108)
    v50: v122[v11, v114[v11]] = v123(v114)
    for v49 in v133(v177(v39)):
        v50[v263(v49)].v196(v49)
    v37, v36 = ([], [])
    v56 = 0
    for v47 in v50.v142():
        if v177(v47) < v29:
            continue
        v137 = f"fp{v56}:{v39[v47[0]]['address']}"
        v56 += 1
        v36.v196(v137)
        for v147, v49 in v135(v47):
            v38 = v122(v39[v49])
            v38['straddr'] = v38['address']
            v38['address'] = v137
            v38['source'] = f'wiki:{v137}:{v147}'
            v37.v196(v38)
    return (v37, v36)

def model_assertions(v57, v36, v58: v11, v59: v125, v60: v11):
    """The same question asked N times at temperature. What varies is the model, not the tape."""
    v37 = []
    for v38 in v36:
        v148 = v5.v223(S=v38)
        v149 = v57.v68(v148, return_tensors='pt')
        v149 = {v131: v115.v167(v57.v64) for v131, v115 in v149.v136()}
        v150 = v11(v149['input_ids'].v267[1])
        with v228.v268():
            v224 = v57.v242.v269(**v149, max_new_tokens=v60, do_sample=True, temperature=v59, top_p=0.95, num_return_sequences=v58, pad_token_id=v57.v68.v292)
        for v49 in v133(v224.v270(0)):
            v225 = v57.v68.v271(v224[v49][v150:], skip_special_tokens=True)
            v13 = v254.v293(v225) or v118.v293('[A-Za-z][a-z]{4,}', v225)
            if v13:
                v37.v196({'address': v38, 'value': v13.v255(0).v257(), 'source': f'lm:{v38}:{v49}', 'ctx': v225[:120]})
    return v37

def build(v39, v40, v61):
    v21 = v151(v40, v61)
    for v38 in v39:
        v21.v226(v38['address'], v38['value'], v38['source'])
    return v21

def main() -> v11:
    v62 = v227.v152()
    v62.v153('--smoke', action='store_true')
    v62.v153('--source', choices=('corpus', 'model'), default='corpus')
    v62.v153('--model', type=v18, default='Qwen/Qwen2.5-0.5B-Instruct')
    v62.v153('--addresses', type=v11, default=0)
    v62.v153('--min-mentions', type=v11, default=3)
    v62.v153('--require-link', action='store_true', help="ablation: demand an English link word between anchor and value. Off by default - a word list does not survive paraphrase or dialect, and the structural rule plus 281's statistics do the same job without one.")
    v62.v153('--min-frame-words', type=v11, default=1, help='tokens that must stand between the two names. 1 is the language-free rule that kills the adjacency pairs; 0 restores them.')
    v62.v153('--no-entity-anchors', dest='entity_anchors', action='store_false', help="keep common nouns as anchors (the first fp run's behaviour)")
    v62.v153('--address-tau', type=v125, default=0.9, help='fp mode: key cosine at or above which two assertions may share an address. The ink proposes; the overlap below verifies.')
    v62.v153('--address-overlap', type=v11, default=2, help="fp mode: content words the two write contexts must share before the ink's proposal is accepted")
    v62.v153('--address', choices=('anchor', 'anchor_rel', 'fp'), default='anchor_rel', help='anchor alone is not an address - it carries no relation, so every sentence containing the word lands on it. anchor_rel adds the content words between the anchor and the value. anchor reproduces the first run.')
    v62.v153('--samples', type=v11, default=8, help='model source: samples per address')
    v62.v153('--temperature', type=v125, default=0.9)
    v62.v153('--max-new', type=v11, default=12)
    v62.v153('--soft-match', type=v125, default=0.0, metavar='TAU', help='let the ink propose that two surface forms are one value, at cosine >= TAU within an address. 0 = exact string match only.')
    v63 = v62.v154()
    v4.v54.v96(parents=True, exist_ok=True)
    v4.v155('', encoding='utf-8')
    v64 = v228.v64('cuda' if v228.v286.v247() else 'cpu')
    v27 = v229.v156(v3)
    v228.v157(v3)
    v65 = v158.v158()
    v66 = v63.v159 or (60 if v63.v160 else 600)
    v67 = 3000 if v63.v160 else 25000
    v161(f'Stage279 write decision start {v304.v298(v305.v299).v248()} device={v64} source={v63.v109} soft_match={v63.v176}')
    v162, v162, v163, v164 = v165()
    v68 = v230.v166(v18(v272.v231))
    v69 = v273(v164, v68.v294()).v167(v64)
    v69.v168(v228.v274(v1, map_location=v64, weights_only=False)['model'])
    v69.v169()
    for v70 in v69.v170():
        v70.v232(False)
    v40 = v171(v69, v163, v64)
    with v2.v190('r', encoding='utf-8', errors='ignore') as v97:
        v172 = v97.v233(3000000 if v63.v160 else 20000000)
    v22 = [v275.v257() for v275 in v172.v260('\n') if 80 <= v177(v275.v257()) <= 400][:v67]
    v32 = v234(v22) if v63.v173 else None
    if v32:
        v161(f'  common-noun filter: {v177(v32)} words excluded as anchors')
    v77, v36 = v174(v22, v27, v66, v63.v29, v63.v102, common=v32, require_link=v63.v33, min_frame_words=v63.v34)
    if v63.v102 == 'fp':
        v77, v36 = v235(v77, v40, v63.v236, v63.v237, v63.v29)
        v161(f'  fp addressing: tau={v63.v236} overlap={v63.v237} -> {v177(v36)} addresses, {v177(v77)} assertions')
    v161(f'  corpus: {v177(v36)} addresses, {v177(v77)} assertions ({v158.v158() - v65:.0f}s)')
    if v177(v36) < 8:
        v161('  not enough multi-mention addresses')
        return 1
    v71 = v175(v77, v40, v63.v176)
    v72 = v177(v77)
    v73 = v71.v73()
    v74 = [v238['support'] for v115 in v71.v99.v142() for v238 in v115.v142()]
    v75 = {'assertions': v72, 'addresses': v177(v71.v99), 'slots': v71.v239(), 'actions': v122(v71.v100), 'dedup_rate': v71.v100[v16] / v240(1, v72), 'dispute_rate': v177(v73) / v240(1, v177(v71.v99)), 'slots_saved_vs_append': 1.0 - v71.v239() / v240(1, v72), 'support_mean': v125(v290.v295(v74)) if v74 else 0.0, 'support_max': v11(v240(v74)) if v74 else 0}
    v161('  corpus tape: ' + v288.v249(v75))
    v76 = v151(v40, v63.v176)
    for v38 in v77:
        v76.v226(v38['address'], v38['value'], v38['source'])
    v78 = v122(v76.v100)
    for v38 in v77:
        v76.v226(v38['address'], v38['value'], v38['source'] + '#2')
    v79 = {v131: v76.v100[v131] - v78.v253(v131, 0) for v131 in (v15, v16, v17)}
    v161(f'  replay control (second identical pass): {v288.v249(v79)}')

    def variants(v115: v18):
        return [v115.v258(), v115 + ' ,', ' ' + v115]
    v80 = v151(v40, v63.v176)
    for v38 in v77:
        v80.v226(v38['address'], v38['value'], v38['source'])
    v81 = v122(v80.v100)
    v82 = v229.v156(v3 + 4)
    for v38 in v77:
        v80.v226(v38['address'], v82.v276(v296(v38['value'])), v38['source'] + '~')
    v83 = {v131: v80.v100[v131] - v81.v253(v131, 0) for v131 in (v15, v16, v17)}
    v84 = v83[v16] / v240(1, v197(v83.v142()))
    v161(f'  paraphrase control: {v288.v249(v83)} merged={v84:.3f}')
    v178, v179 = ({}, {})
    if v63.v109 == 'model':
        import _stage262_trunk_swap as s262
        if v297.v137 == 'nt' and v277(v278, 'ensure_short_hf_home'):
            v278.v279()
        v57 = v278.v241(v63.v242, v64)
        v180 = v36[:12 if v63.v160 else 120]
        v181 = v243(v57, v180, v63.v244, v63.v59, v63.v60)
        v182 = v175(v181, v40, v63.v176)
        v183 = v182.v73()
        v178 = {'model': v63.v242, 'samples_per_address': v63.v244, 'temperature': v63.v59, 'assertions': v177(v181), 'addresses': v177(v182.v99), 'slots': v182.v239(), 'actions': v122(v182.v100), 'self_dispute_rate': v177(v183) / v240(1, v177(v182.v99))}
        v161('  model tape: ' + v288.v249(v178))
        v184 = []
        for v38 in v182.v99:
            v280, v281, v282 = v182.v283(v38)
            v284, v162, v162 = v71.v283(v38)
            if v284 is None or v282 < 2:
                continue
            v184.v196((v281, v11(v280 is not None and v280.v258() == v284.v258())))
        if v177(v184) >= 8:
            v184.v285(key=lambda v308: v308[0])
            v245 = v240(1, v177(v184) // 3)
            v203 = v125(v290.v295([v130 for v162, v130 in v184[:v245]]))
            v246 = v125(v290.v295([v130 for v162, v130 in v184[-v245:]]))
            v179 = {'n': v177(v184), 'acc_low_consistency': v203, 'acc_high_consistency': v246, 'lift': v246 - v203, 'consistency_mean': v125(v290.v295([v130 for v130, v162 in v184]))}
        else:
            v179 = {'n': v177(v184), 'note': 'too few addresses shared with the corpus tape'}
        v161('  calibration: ' + v288.v249(v179))
        del v57
        if v228.v286.v247():
            v228.v286.v287()
    v85 = v79[v17] == 0 and v79[v15] == 0
    v86 = v75['dedup_rate'] >= 0.1
    v87 = v75['dispute_rate'] >= 0.05
    v88 = v75['slots_saved_vs_append'] > 0.0
    v89 = None if v63.v176 <= 0.0 else v84 >= 0.5
    v90 = v63.v109 == 'model' and v179.v253('n', 0) >= 8
    v91 = None if not v90 else v179.v253('lift', -1.0) >= 0.15
    if not v85:
        v185 = 'WRITE_UNSOUND'
    elif not (v86 and v88):
        v185 = 'NO_DEDUP'
    elif v91 is None and v63.v109 == 'model':
        v185 = 'SUPPORT_UNTESTED'
    elif v63.v109 == 'model' and (not v91):
        v185 = 'SUPPORT_NOT_CALIBRATED'
    elif v87 and v91 is not False:
        v185 = 'WRITE_DECISION_OK'
    else:
        v185 = 'WRITE_DECISION_PARTIAL'
    v92 = []
    for v38 in v73[:12]:
        v186 = v198(v71.v99[v38].v136(), key=lambda v300: -v300[1]['support'])
        v92.v196({'address': v38, 'values': [{'value': v115, 'support': v238['support'], 'sources': v238['sources'][:3]} for v115, v238 in v186[:4]]})
    v37 = {'stage': 279, 'overall': v185, 'source': v63.v109, 'soft_match': v63.v176, 'trained_parameters': 0, 'min_mentions': v63.v29, 'address_mode': v63.v102, 'smoke': v63.v160, 'seed': v3, 'gates': {'G_no_false_dispute_on_replay': v85, 'G_dedup_happens': v86, 'G_tape_compresses': v88, 'G_disputes_found': v87, 'G_soft_match_merges_forms': v89, 'G_support_predicts_truth': v91}, 'corpus_tape': v75, 'replay_second_pass': v79, 'paraphrase_second_pass': {**v83, 'merged_fraction': v84}, 'model_tape': v178, 'calibration': v179, 'disputed_examples': v92, 'note': "The write side, which GOAL.md named as the open half. WRITE / CONFIRM / DISPUTE against an address, and the three consequences that follow without any further mechanism: the tape stops growing on repetition, the witness structure that 271-278 had to manufacture with a template arrives from the data as a support count, and that count is a reliability estimate computed for free. The replay control is the validity gate - writing the same corpus twice must produce no disputes and no new slots, whatever the matcher, because a matcher that invents disagreement is worse than one that misses it. With --source model the same subject is sampled N times at temperature and the share of samples that agree is the model's uncertainty measured from what it says rather than from its logits; G_support_predicts_truth checks that number against the corpus majority, and if the lift is absent the support count is decoration and the verdict says so.", 'timestamp': v304.v298(v305.v299).v248(), 'wall_s': v158.v158() - v65}
    v0.v96(parents=True, exist_ok=True)
    v93 = '' if v63.v109 == 'corpus' else '_model'
    (v0 / f'stage279_decision{v93}.json').v155(v288.v249(v37, indent=2), encoding='utf-8')
    (v0 / f'stage279_mini{v93}.md').v155(f"# Stage 279 write as a decision ({v63.v109})\n\n**{v185}**{(' · SMOKE' if v63.v160 else '')} · trained parameters **0**\n\n- {v75['assertions']} assertions -> **{v75['slots']}** slots over {v75['addresses']} addresses ({v75['slots_saved_vs_append']:.1%} saved against an append)\n- actions {v288.v249(v75['actions'])}, dedup **{v75['dedup_rate']:.3f}**, disputed addresses **{v75['dispute_rate']:.3f}**\n- replay control (identical second pass): {v288.v249(v79)}\n" + (f"- model self-dispute **{v178.v253('self_dispute_rate', v125('nan')):.3f}**, consistency lift **{v179.v253('lift', v125('nan')):.3f}** (low {v179.v253('acc_low_consistency', v125('nan')):.3f} -> high {v179.v253('acc_high_consistency', v125('nan')):.3f})\n" if v178 else '') + '\n## Gates\n\n' + ''.v289((f'- {v131}: **{v115}**\n' for v131, v115 in v37['gates'].v136())), encoding='utf-8')
    v161(v288.v249({'overall': v185, 'gates': v37['gates']}, indent=2))
    return 0
if v94 == '__main__':
    raise v187(v250())