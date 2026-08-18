# TapeLM / Inprint — STATE at 353. Self-contained. Written to survive compaction.

The local HANDOFF.md is truncated, so this file repeats everything load-bearing rather than
pointing at it. Nothing here is new; it is the record, compressed.

---

## 0. HOW WE WORK (standing, do not re-derive)

- Answer as briefly as possible. Tokens are expensive.
- NO HEURISTICS. Everything must scale and must be a count.
- On a wall: propose a STRUCTURAL change, never parameter fitting.
- Goal: a real mind. Knowledge separate, decision separate. Not a reference book.
- Everything runs on Kostya's Windows machine. Claude has no torch and no corpora: it writes
  code, audits it statically, and reads returned JSON/logs.
- Repo sync: do nothing, the working copy is local. Do not spend tokens on the GitHub API.
- Branch `claude/286-json-prognoza-wnt8r5`. No PRs unless asked.
- PUSH RETURNS 403 AT THE GATEWAY, ALL SESSION. Read verified working (`git ls-remote` OK),
  proxy clean. Everything is delivered as `git format-patch` files. ~251 commits unpushed.

## 1. THE RULES FOR SPENDING EFFORT (344, amended twice)

- ONE LEVER = one mechanism, decided by 4 seeds against a gate DECLARED BEFORE THE RUN.
- A STEP gets 3-5 levers. Then THE VIEW OF THE PROBLEM CHANGES - mandatory, not optional.
- A BUG FIX IS NOT A LEVER. A bug is the code failing to do what it was designed to do. A
  mechanism that did what it was designed to do and lost IS a spent lever.
- A LEVER THAT WORKS AND FALLS SHORT MAY BE CONTINUED: directionally right on >= 3 of 4 seeds
  and below threshold. Not "one seed looked good", not "a different quantity looked good".
- NO NEW METRIC MAY DECIDE A STEP. A measurement that cannot change the next step is not run.
- The invariant test (transplant + native control + shuffle null) rides along with every arm.
  It is the instrument, never a step of its own.

## 2. READING DISCIPLINE (each of these has already cost a wrong conclusion)

- VOID BEFORE FAILED. If the mind never takes an option (constrain_rate 0.003, step_rate 0),
  the comparison measures nothing. 299_hash, 311's first pair, 339, 345.
- READ RATIOS, NOT z, WHEN n DIFFERS. z grows with the discordant count. Caught at 352price.
- A KNOB MUST MOVE ITS QUANTITY before anything below it is read (DEAD KNOB). The dead
  `--addresses` flag in 335, and `--bytes` failing to thicken a place in 346.
- A NULL MUST BE READ ON AN ABSOLUTE QUANTITY. A conditional rate collapses its own denominator
  and stays flat while the signal dies (342a).
- IDENTICAL DIGITS AT TWO SETTINGS = a collided tag or a dead knob, UNLESS the other columns
  moved too (then it is genuine mind-invariance, as walk-only is per 336).

## 3. ESTABLISHED, NEVER RETRACTED

- SEPARATION, controlled four independent ways: 5633 params and `not_loaded` ink; transplant to
  two foreign corpora with route and pick intact; fine-tuning on the foreign corpus buys
  nothing; a NATIVE mind on news is indistinguishable from the transplant (73/71 discordant of
  8000, z +0.17); and the shuffle-tape null erases everything.
- The mind ROUTES better than every counting router at matched budget, PICKS better than every
  rival of the closed set, is CALIBRATED, and GATES the honest half better than any count.
- PHI HAS NO ABSOLUTE SCALE. 329, 337 and 340 agree from three unrelated experiments: the raw
  score is a coin (AUC 0.44-0.49), only the GAP to the runner-up carries anything. Phi is
  defined only relative to other completions of the same world.

## 4. ARCHITECTURE LAWS (measured, not assumed)

- **THE 4x LAW.** A SECOND OBJECTIVE COSTS ~4x THE ROUTE, AND IT IS NOT CAPACITY:
      321  bisection teacher, d=32                      4.00x
      341  speaking price,    d=32 news  11.31 -> 2.90  3.90x
      352  speaking price,    d=64 wiki  15.15 -> 4.04  3.75x   (3.8x the parameters)
  Two unrelated terms, two widths, one factor. Every remaining capability must live INSIDE the
  one objective. gamma^2 at depth is the proof it can: one price, one head, largest result here.
- CAPACITY IS NOT THE LEVER for one objective (342a: d=32 vs d=64 flat, invariant holds at
  21505 params).
- CORPUS SIZE IS NOT THE LEVER (335: the gap OPENS with width, share shown falls).
- RETENTION / PLACE SELECTION IS NOT THE LEVER (338: an exact count beat the mind at it).
- THICKNESS IS NOT THE LEVER (347: support2+ x2.2 but the offer x5.4, so present@8 FALLS;
  in_own 0.26 -> 0.51, i.e. thickness helps the INDEX).
- THE TAPE'S ONLY USABLE RELATION IS SUBSTITUTION. Two fillers relate when they can fill the
  SAME HOLE - alternation, not association. 349 (same-line) reaches half as much and resolves
  5x worse; 350 (directed triples) is worse still and buys reach only by widening the offer.
  A substitution relation RANKS ALTERNATIVES AND CANNOT PRODUCE CONTENT.

## 5. CLOSED (do not reopen without a named new reason)

composition as a single joint world (310, factorises by identity) · the line channel (304) ·
`--stage2-always` (314) · bisection as a faculty (321; it IS a compression, 91% of the pick at
~2.9 comparisons) · the stepping stone (333) · dilution/two-way at depth 1 (334) · mind-chosen
retention (338) · the speaking price as a second term (341, 352) · the constraint interface:
raw count (345 L1), share (345 L2), two lenses intersected (346, closed unbuilt by audit) ·
thicker places (347) · a second tape relation (349, 350).

## 6. THE BIG ONE — 352: DEPTH WAS CLOSED WRONGLY

                       depth 1   depth 2
    reachable            0.127     0.539     x4.2
    hit_rate             0.068     0.458     x6.7
    hit_of_walk_only     0.180     0.877
    step_rate            0.0115    0.980
    RANK right AUC       0.866     0.969
    ROUTER               11.31x     1.00x    DEAD
    WALK-ONLY pooled  61/33 z+2.89  ->  7056/62 of 8503  z +82.90
    PICK vs COUNT                        1878/21         z +42.61
    GATE-WO 25%       0.663/0.158   ->  0.9976 vs share 0.7332, random 0.8296
    CONFIRM pooled   100/74 z+1.97  ->   191/354         z -6.98   DESTROYED

THREE ARTEFACT CHECKS, ALL PASSED:
- the mind-rooted denominator: 333's `--deep-root first` (mind-independent) reaches MORE
  (0.55-0.58), so 0.54 is the two-hop walk's property, not Phi moving its own goalposts;
- the rival is not blind: since 317 `reach_count_rival` takes candidates and places from EVERY
  depth. Its 0.012 is below chance for 317's own reason - deep candidates are rare by
  construction and a count rival prefers frequent fillers;
- the step logit is not inflated: 325 fixed exactly that, with a static guard on the ORDER of
  two statements in reach_logits. step_rate 0.98 is the policy being CORRECT (0.54 vs 0.12).

THE DEFECT, NAMED: it cannot keep home while going deep. That is the CARDINALITY FAULT FOR THE
THIRD TIME (line cardinality 304, the step logit under depth 325, now stay-vs-go). Under plain
lookahead stage one is a softmax over [own_1..own_k, step] where step = max(l2): EACH OWN WORLD
COMPETES INDIVIDUALLY AGAINST A MAXIMUM OVER EIGHT, and a maximum grows with the number of
samples. At depth 1 there was nothing behind the step so it cost little; at depth 2 there is
0.54 behind it.

322 saw the same CONFIRM collapse and closed depth. The right reading: DEPTH WORKS and the
policy carrying it cannot also mind the home - a routing failure, not a capability one.

## 7-RESULT. 353 PASSED. THE FAULT WAS CARDINALITY, AND `--two-way` IS THE CARRIER

Read on 4 seeds (the first pool double-counted: out/ and results/ hold the same file, so it
printed "8 runs" for 4 seeds and halved the honest z; `_read299.py` now dedupes by name).
RATIOS, not z - z rises with the discordant count.

                     WALK-ONLY     WO ABSOLUTE     CONFIRM      ROUTER
                     mind/rival    (random@25%)    mind/rival
    352deep          7056/62 113.8    0.8296       191/354 0.54   1.02   home LOST
    353 two-way max  7258/87  83.4    0.8347       334/261 1.28   1.00   PASS, clean
    353 two-way marg 6237/54 115.5    0.7155       330/203 1.63   1.10   PASS, but pays

BOTH clear the pre-registered gate. THE WINNER IS `max`, and it is not the bigger CONFIRM:
- `max` recovers home (0.54 -> 1.28, from losing to winning) AT NO COST TO DEPTH: the walk-only
  ABSOLUTE accuracy is 0.835 against 352deep's 0.830. Its lower discordant RATIO is a ratio of
  discordant pairs on an unchanged absolute - exactly the reading error section 2 warns about.
- `margin` buys more home (1.63) and wakes the router (1.10x), but the walk-only absolute FALLS
  0.83 -> 0.72, flat across all four gate fractions, i.e. a genuine base-rate drop on the same
  number of walk-only questions (17324 vs 17330). It pays for home WITH depth. That is the
  trade the gate's "either alone is not a pass" clause exists to catch, showing up inside an
  arm that passes both halves.

SO: the stay-vs-go asymmetry WAS the defect. Equal counts on both sides fixed it. A branch is
worth its BEST world, not the gap between its best two - the 0.969 margin signal ranks worlds
INSIDE a branch and is the wrong summary BETWEEN branches.

DEPTH IS NOW ESTABLISHED, not merely un-closed: reach x4.2, hit x6.7, home kept, one price, one
head, 5633 parameters. This is the largest result the project has recorded, and it arrived from
a STRUCTURAL change inside the single objective - not capacity, not corpus, not a new relation.

DEPTH ABSOLUTES, 4 seeds, held_out - `max` is not a trade at all, it dominates on every column:

                 reachable   hit     wo_hit   step
    352deep        0.532    0.423    0.835    0.947
    353 max        0.541    0.445    0.838    0.970     >= on all four
    353 margin     0.540    0.396    0.725    0.751     reaches the same, walks less, hits worse

THE STANDING ARM IS NOW `--reach-depth 2 --two-way` (max is the default summary). Every control
from here compares against it, not against depth 1.

## 7-WAS. THE PRE-REGISTRATION, kept for the record

`--two-way` makes stay-vs-go a comparison of two summaries over EQUAL counts
(`m = min(len(l1), len(l2))`) - one price, one head, inside the same objective. Not a second
term, so the 4x law does not apply. It failed at depth 1 where it was correcting an asymmetry
that barely existed; at depth 2 there is 0.54 behind the step.

`--two-way-by {max,margin}`: a branch is worth its best world (max, every run to date) or the
GAP between its best two (margin). The margin is the AUC 0.969 signal deciding something for the
first time. Declared rule, not fitted; `max` reproduces every earlier number exactly.

    <arm> --reach-depth 2 --two-way                     --seed <s> --run-tag 353twoway_s<s>
    <arm> --reach-depth 2 --two-way --two-way-by margin --seed <s> --run-tag 353margin_s<s>
    4 seeds each, wiki, --reach-max-q 8000, depth 2. CONTROL = 352deep, already in hand.

PRE-REGISTERED GATE:
  PASS  CONFIRM back to z >= 0 pooled AND walk-only stays z >= +10 pooled. Home kept, depth
        kept. EITHER ALONE IS NOT A PASS - recovering CONFIRM by abandoning depth is the old
        arm with extra steps, and 352deep already holds the depth half.
  ROUTER above 1.00x is the DIAGNOSTIC (stay/go became a decision again), not the gate.
  FAIL  both levers leave CONFIRM negative -> the defect is not cardinality, and the question
        becomes whether home and depth can share one policy at all.

## 8-RESULT. 354 FIRST PASS: POOLED MEMORY LOSES. The oracle was not measured and now is

                      own      pooled W   null R   gain/null   chain_only   lost
    situation (l40)  0.1716     0.1336    0.1332    +0.0004      0.0100     0.0480
    whole tape (l0)  0.1779     0.1458    0.1383    +0.0075      0.0129     0.0450

Three things, in order of what they cost:

1. WRITING ANSWERS BACK IS NEGATIVE, not merely flat: -0.038 and -0.032. The offer more than
   doubles (56 -> 138) and the truth falls out of the top-8. LOST 0.048 against CHAIN_ONLY
   0.010 - a written-back answer costs almost five times what it buys. Not VOID: the knob moves
   its quantity hard, so this is a real loss.
2. THE TRUE ANSWER AND A WRONG ONE ARE INTERCHANGEABLE AS LENSES: 0.1336 vs 0.1332. Identical
   to three decimals while the other columns moved, which by section 2 means the knob is alive
   and the two are genuinely the same. The IDENTITY of what was answered carries nothing.
3. THE SITUATION IS WORSE THAN THE CONTROL. Tightening a session to 40 lines gains LESS over
   the null than drawing across the window. Locality buys nothing, which is the opposite of
   what a session hypothesis predicts. (Caveat on the control: the tape is already a 400-line
   window, so l0 is "independent within the window", not 324's whole corpus. The direction
   still runs against locality.)

AND THIS IS THE THIRD TIME SUMMING LENSES HAS LOST: 345 (a constraint by count, then by share),
346 (two lenses intersected), now 354 (answers pooled into the offer). 353's fix for the same
shape of defect was not a better sum - it was making the comparison cardinality-correct. A
memory that works here would have to be a SELECTION of one lens, never an accumulation.

WHAT THE INSTRUMENT MISSED, and it is an instrument defect, not a spent lever. 351 measured the
ORACLE ceiling (some path lands on the truth) because Phi's whole job is CHOOSING. 354's first
pass measured memory USED INDISCRIMINATELY - every previous answer pooled with equal weight -
which is a job this architecture would never be given. `_audit354_session.py` now also reports:

    oracle_W   the truth reachable through ONE previous answer, chosen perfectly
    oracle_R   the same with WRONG answers - the null, load-bearing, because with five
               answers in hand a wrong one also gets five shots at the top-m
    held       how many answers the selection is over. The size of the job

  GATE  oracle_gain = oracle_W - oracle_R > 0.05.
  If the oracle is flat too, the identity of a previous answer carries nothing under ANY
  policy, memory is closed on the substrate rather than on the pooling, and 324 stands.

    python _audit354_session.py --session 6 --session-lines 40
    python _audit354_session.py --session 6 --session-lines 0

## 8-RESULT-2. 354 CLOSED: REMEMBERED CONTENT IS WORTH NOTHING, ORACLE INCLUDED

                      own      pooled   null    oracle   oracle null   gain
    situation (l40)  0.1716    0.1336   0.1332  0.1829     0.1833    -0.0004
    whole tape (l0)  0.1779    0.1458   0.1383  0.1908     0.1883    +0.0025
    choosing among 2.5 held answers, matched budget, the null given the same number of shots.

A TRAINED LEVEL CANNOT BEAT ITS OWN ORACLE. Perfect memory plus perfect selection is +0.0025,
so no level, no head, no capacity added to Phi makes remembered CONTENT pay. MEMORY OF FACTS IS
CLOSED ON THE SUBSTRATE, and 324's verdict now rests on a measurement 324 never made.

## 8-NEXT. 355: THE OTHER KIND OF MEMORY, the one this project's split actually allows

354 measured a memory of FACTS - and facts belong to the tape, where a memory of them is just
a lens, which is why it lost three times over (345, 346, 354). The split says the other half is
POLICY, and a memory of READING - "the tape is thin here, do not speak" - holds no facts and
breaks no invariant. It has never been measured.

`_audit355_carry.py`, torch-free:
    lift     p(reach | the previous question in this situation reached) - p(reach | it missed)
    null     the same with the previous question drawn from ANOTHER situation
    WITHIN   the same inside strata of the CURRENT question's own top1-minus-top2 count margin.
             THE DECIDING NUMBER: the part of the situation the question does not already
             announce. Reports its own COVERAGE, because a within of 0.0 on strata that hold
             only hits or only misses is VOID, not flat.

  GATE  WITHIN - |within null| > 0.05. Then a carried state is worth a level, and the level
        holds no facts.
  RAW LIFT LARGE, WITHIN FLAT -> situations differ and the question already says so. The state
        is redundant with what Phi sees. BUILD NOTHING.
  BOTH FLAT -> answerability is not a property of a situation, and memory is closed for content
        (354) and for reading (355). The loop as a source of capability is finished, and what
        remains of 351 is the reward and the choice of question, not state.

    python _audit355_carry.py --session 6 --session-lines 40
    python _audit355_carry.py --session 6 --session-lines 40 --bins 3

## 8-RESULT-3. 355 CLOSED TOO. AND WITH IT THE LOOP, AND WITH IT THE VIEW

    base reach 0.1652, 600 situations, 2978 pairs, coverage 2978 of 2978 (NOT void)
    LIFT    after a hit 0.1513   after a miss 0.1681    -0.0168
    NULL    previous from ANOTHER situation             +0.0006
    WITHIN  inside strata of the question's own margin  -0.0111 (bins 5) / -0.0111 (bins 3)

Answerability is not a property of a situation. One question's outcome says nothing about the
next - the tiny sign is NEGATIVE, and the cross-situation null is zero. Memory is closed for
CONTENT (354, oracle included) and for READING (355). The loop is not a source of capability
here, and by the standing rule the VIEW OF THE PROBLEM NOW CHANGES.

## 8-NEXT. THE CHANGE OF VIEW: WE HAVE ONLY EVER MEASURED ACCURACY, NEVER COVERAGE

For 350 steps every number - reach, hit, pick, gate, transplant, invariant - has been computed
over questions drawn from `keep`. `keep` is the positions whose FRAME RECURS and which have TWO
OR MORE DISTINCT FILLERS. Everything else was dropped in the write path and has never appeared
in any measurement, as question or as answer. NOBODY HAS STATED WHAT FRACTION OF THE CORPUS
THAT IS. Every result may be excellent accuracy inside a sliver.

AND THERE IS A RULE IN THE WRITE PATH PHI NEVER GOT A SAY IN. `frame_keep` reads each position
at its WIDEST RECURRING frame. Widest = fewest, most precise fillers; narrowest = most
positions addressable, vaguest paradigm. WIDTH IS COVERAGE, the trade has cardinality on both
sides, it was fixed by fiat in the first commit, and it is the LAST decision in the pipeline
still made by the tape rather than by the mind. That is the structural change this angle opens.

`_audit356_coverage.py` measures the fiat's price first, torch-free, no gate to pass because it
is a description of the denominator - but it decides the next step outright:

    addressable   positions whose frame recurs at some width
    on_tape       and whose place has >= min_fillers distinct fillers   THE REAL DENOMINATOR
    by width      where the widest recurring frame lands
    by rarity     P(on_tape | corpus frequency of the token in the hole)   THE MONEY ROW

  A FACT IS A RARE TOKEN. If rare tokens are essentially never on the tape, the knowledge half
  is a lattice of CATEGORIES by construction, no amount of mind reaches a fact, and the
  project's honest claim changes shape - separation is proven for the substitutable part of
  language, which is not the part that carries facts.
  If rarity is NOT the filter, coverage is simply the denominator everything so far sat inside,
  and the next lever is giving the WIDTH to Phi: a fixed rule (always widest, always narrowest)
  is the rival, it lives inside one objective, and it buys coverage with precision.

  The audit re-runs frame_keep's widening loop locally so the WIDTH of each position is visible,
  and asserts the place count and addresses match frame_keep exactly - verified on a synthetic
  corpus before shipping.

    python _audit356_coverage.py
    python _audit356_coverage.py --frame-max 1

## 8-RESULT-4. 356: COVERAGE IS 0.56, AND MY PREDICTION WAS WRONG

    corpus 3192733 tokens, 83219 types, 25000 lines. frame_keep agrees on places and addresses.
    ADDRESSABLE  0.6827 (identical at frame-max 1 and 3, as it must be)
    ON TAPE      0.5594 at frame-max 3   |   0.5937 at frame-max 1
    WIDTH        w1 83.8%   w2 12.8%   w3 3.5%
    P(on tape | frequency of the token in the hole):
        1        0.6836        6-20     0.6921
        2        0.6755        21-100   0.6941
        3-5      0.6853        101+     0.5260

THREE THINGS, AND THE FIRST TWO ARE GOOD NEWS THE PROJECT HAD NEVER EARNED THE RIGHT TO CLAIM:

1. COVERAGE IS A MAJORITY, NOT A SLIVER. 56% of every interior token position in wikitext is
   askable. The honest end-to-end figure, stated for the first time: coverage x hit at depth 2
   = 0.559 x 0.445 = 0.25 OF ALL TOKEN POSITIONS, recovered by counts plus 5633 parameters.
2. RARITY IS NOT THE FILTER - I PREDICTED THE OPPOSITE AND WAS WRONG. A once-seen token reaches
   the tape MORE often (0.684) than a token seen 101+ times (0.526). A frequent token fails
   because its hole has ONE filler and is dropped: the filter is PREDICTABILITY, not rarity.
   The tape is not a lattice of categories. Facts are present.
3. THE WIDTH FIAT IS CHEAP: 3.4 points of coverage, touching only the 16.2% of positions that
   reach width 2 or 3. "GIVE THE WIDTH TO PHI" IS DEAD ON ITS OWN NUMBERS, before a run.

## 8-NEXT-2. 357: ONE POSITION, ONE ADDRESS - the part of the fiat that is not about coverage

`frame_keep` writes `best_key[i] = the widest recurring frame` and DELETES the narrower ones.
A width-3 position is also a member of its width-2 and width-1 paradigms, and those are the
COARSE ones: more fillers, more bridges, exactly the edges a walk needs to leave a narrow
paradigm and land in another. For 16% of positions the write path deletes their only wide
connections. NOT A SECOND RELATION - 349 and 350 added a different kind of edge and lost; this
is the SAME substitution relation kept at more than one granularity.

`_audit357_resolution.py`, torch-free:
    reach_wide    one address per position                                  (today)
    reach_multi   every recurring width kept as an address
    reach_null    the coarse SEATS permuted among the positions that hold them - the null tape
                  is structurally IDENTICAL to the multi tape (same addresses, same sizes, the
                  same natives) and differs only in who sits in each coarse seat. Verified on a
                  synthetic corpus: same place count, same size multiset, same members.
  GATE  reach_multi - reach_null > 0.05 at matched top-m.
  Every place the question's position sits in is banned AT EVERY WIDTH, or the coarse layer
  hands the question its own answer back.

    python _audit357_resolution.py
    python _audit357_resolution.py --window-lines 3200

## 8-RESULT-5. 357 CLOSED, AND WITH IT THE i.i.d. EXAM ITSELF

    thin  (400):  wide 0.1965  multi 0.2190  null 0.1810  gain/null +0.0380
    thick (3200): wide 0.1885  multi 0.1845  null 0.1520  gain/null +0.0325, vs wide NEGATIVE
Under the gate on both, shrinking with thickness. One position, one address costs nothing.

THREE AUDITS IN A ROW (354, 355, 357) say the same thing: under INDEPENDENT cloze questions
the substrate has nothing more to give. 354/355 closed memory ON AN EXAM THAT IS i.i.d. BY
CONSTRUCTION - both question and tape drawn from one finished world. That is a property of the
exam, not of minds.

## 8-NEXT-3. THREE NON-STANDARD DOORS, one built (358)

1. **READING IS WRITING (358, built).** Wikitext's real dependence is SELF-REFERENCE WITHIN A
   DOCUMENT: an article introduces its subject once and repeats it for forty lines. The first
   mention is unanswerable from outside; later mentions are answerable FROM THE ARTICLE'S OWN
   EARLIER LINES. The project never saw this: the tape is built once, nothing is ever NEW,
   nothing is ever ingested. The loop is not "remember answers" (354) - it is reading a new
   document INTO THE WORLD line by line, the answer becoming corpus, not lens.
   EVADES 354's closure: there the tape already indexed everything either question needed;
   here the document is NOT in the old world and its own past is the only place its later
   fillers stand. NULL: the same-length prefix of a DIFFERENT document - same mass, same added
   places, wrong content. Fixed absolute denominator: every token of the question line, all
   three arms (the 342a lesson).
   GATE  pooled ingest - null > 0.05 AND the late third of the document above the early.
   Verified on a constructed case: truth enters the offer only through the ingested line
   (base (8,1,0) -> ingest (8,1,1)).
   CAVEAT NAMED BEFORE THE RUN: min_fillers >= 2 deletes places whose fillers are all one
   token - the "title paradigm" of pure self-reference. Ingestion here can only pay through
   mixed places. If the gate fails NARROWLY, the one declared suspect is that filter.

       python _audit358_ingest.py
       python _audit358_ingest.py --docs 20 --doc-lines 16    # quick

2. **THE WALK IS THE LIFE (not built).** Stop drawing questions. The mind walks the lattice
   continuously; at each place it may speak or step; reward per spoken truth under a speak
   budget over a trajectory. State = WHERE IT STANDS - policy state, not a fact, so the
   invariant holds by construction. This turns GATE-WO 0.99 into agency: choosing WHERE to
   speak is "choosing its own question", the last unearned item that is testable here.
   Rival: a random walk with the same speak budget.

3. **DEPTH 3 BY RECURSION (standard control).** 353's rule is recursive: a branch is worth its
   best world over equal counts. Torch-free reach ceiling first; if reach saturates at depth
   2, never built.

## 8-RESULT-6. 358 FAILED ITS GATE AND NAMED THE REAL DEFECT

    askable   base .2612   null .2627   ingest .2743     thirds  .2684 .2694 .2849
    reach     base .0384   null .0384   ingest .0434     thirds  .0417 .0421 .0461
    base and null FLAT down the document; ingest RISES on both rows, monotonically.
    gain over null +0.0049, late-early +0.0064.   Gate wanted +0.05.

THE MECHANISM IS ALIVE AND CORRECTLY SHAPED - a document's own past makes its later holes exist
and be reached, and only its own past does. It is small because THE WRITE PATH DELETES ALMOST
EVERYTHING INGESTION CREATES. The suspect declared before the run is confirmed and is bigger
than one filter:

  1. `min_fillers >= 2` deletes CONSTANT frames. Self-reference produces exactly those.
  2. Even kept, THE EXAM CANNOT ASK ABOUT ONE: the lens is the place's other fillers and the
     offer excludes the lens value (`w != v`), so a constant place's truth is excluded from its
     own offer by construction. Verified: a frame holding only XARWIN gives lens ['XARWIN'] and
     an offer that cannot contain XARWIN.

TOGETHER THEY DEFINE THE EXAM AS "GUESS THE ALTERNATIVE", NEVER "RECALL THE VALUE". Every
number in this project, for 350 steps, is substitution. A FACT IS A CONSTANT FRAME. Recall is
count-trivial, and by this project's own thesis that is correct - it is the KNOWLEDGE half
finally holding something. The mind's job becomes WHICH CHANNEL THIS HOLE WANTS.

## 8-NEXT-4. 359: the recall channel (built, torch-free, min_fillers=1)

    RECALL   the place's other POSITIONS, this position excluded, the place KEPT
    SUBST    today's offer, unchanged
    ORACLE   either                                  the ceiling of a perfect channel pick
    RULE     a count: recall when the place's others agree, else substitute
    NULL     recall from a RANDOM OTHER PLACE of the same size
  GATE  oracle - max(always recall, always subst) > 0.05  AND  recall - null > 0.05.
        The first says a decision exists; the second says recall is real and not frequency.

    python _audit359_recall.py
    python _audit359_recall.py --min-fillers 2      # the old world, for contrast

IF IT PASSES, 358 IS RE-RUN WITH RECALL ADMITTED - ingestion and recall are the same story,
since what a document writes about itself is constant, and that is why 358 was flat.

## 8-RESULT-7. 359 PASSED BOTH GATES, AND IT IS THE BIGGEST FINDING OF THE PROJECT

                    recall   subst   null    oracle  both  only_rec  only_sub  decision
    min_fillers 1   0.3955  0.1428  0.0085   0.4770  .061   .3343     .0815     +0.0815
    min_fillers 2   0.2118  0.1923  0.0065   0.3220  .082   .1298     .1103     +0.1103

1. RECALL IS REAL AND ENORMOUS: 0.3955 against a same-size random-place null of 0.0085. A
   factor of 46. Reading a place's own past is not token frequency.
2. RECALL BEATS SUBSTITUTION ON THE TAPE AS IT IS WRITTEN TODAY. At min_fillers 2 - the world
   every result was measured in - 0.2118 against 0.1923. THE PROJECT HAS SPENT 350 STEPS
   MEASURING THE WEAKER OF THE TWO CHANNELS, and the stronger one was excluded by two lines in
   the write path.
3. THE CHANNELS ARE NEARLY DISJOINT: both .082, only-recall .130, only-subst .110 at mf2.
   Neither subsumes the other. The oracle over the pair is 0.322 against 0.212 for the best
   fixed channel: +0.1103 of headroom for CHOOSING, which is the mind's half.
4. THE NAIVE COUNT RULE LOSES TO BOTH FIXED CHANNELS at mf2 (0.144). Not a mystery: at
   min_fillers 2 "the others agree" means THIS POSITION IS THE ODD ONE OUT, so the filter
   inverts the rule. Named, kept, not fitted around.

BEFORE PHI IS GIVEN THIS DECISION, EVERY COUNT MUST FAIL AT IT. 359 now also plays: top1 (the
louder channel wins), margin (the bigger gap wins), repeated (recall if there is anything
repeated to recall). The gate is now HEADROOM = oracle - the best rival OF ANY KIND > 0.05.
Note the shape: the two channels' scores are not commensurable, and Phi's known property since
329/337/340 is that it has no absolute scale and only the gap carries - which is exactly the
property this decision needs.

RIVALS RUN, BOTH GATES STILL PASS:

                 recall  subst  agree   top1  margin repeat  best    oracle  HEADROOM
    mf 1         .3955  .1428  .3535  .1888  .2223  .2355  .3955    .4770   +0.0815
    mf 2         .2118  .1923  .1440  .1888  .1863  .2245  .2245    .3220   +0.0975

  - At mf2 a count DOES beat both fixed channels: `repeated` (recall if there is anything
    repeated to recall) reads .2245. It is now the rival to beat, not always-recall.
  - At mf1 every rule LOSES to always-recall, because a two-position place gives the others
    a single token, `repeated` sends it to substitution, and substitution is the weak channel.
  - Headroom survives on both settings, which is the only thing the gate asked.

ONE MORE CONTROL BEFORE ANYTHING IS BUILT, and it is the shape that killed 354: MAYBE THE TWO
CHANNELS DO NOT NEED CHOOSING BETWEEN AT ALL. Added and pending: `merge` (sum both offers) and
`interleave` (alternate their tops), BOTH AT THE SAME top-m A CHOOSER GETS. If either reaches
the oracle, Phi is owed nothing and recall is simply a second lens to pool. Note the counts are
not commensurable - place-sized against thousands - so merge should collapse into substitution
and interleave is the real no-decision rival.

    python _audit359_recall.py                    # re-run: merge + interleave
    python _audit359_recall.py --min-fillers 2

## 8-RESULT-8. THE NO-PICK RIVAL WON, AND I HAD THE ARCHITECTURE WRONG. TWO CORRECTIONS

    mf 1   recall .3955  subst .1428  merge .2622  INTERLEAVE .4510  oracle .4770  head +0.0260
    mf 2   recall .2118  subst .1923  merge .2128  INTERLEAVE .2903  oracle .3220  head +0.0318

CORRECTION 1 - THE CHANNEL DECISION IS NOT WORTH A HEAD. Interleaving the two channels at the
budget a chooser would get lands within +0.026 / +0.032 of a PERFECT channel pick. Under the
gate on both settings. Merge collapses toward substitution exactly as predicted (the counts are
incommensurable), but interleave is scale-free and takes nearly all of it. Phi is owed nothing
here.

CORRECTION 2, AND IT IS MINE TO OWN - I SAID "THE TAPE HAS NEVER BEEN ALLOWED TO HOLD A FACT"
AND THAT WAS WRONG ABOUT THE ARCHITECTURE. The recall channel has been in the stage since the
reach verb existed:
    `own` (line ~1479) = the values at the question's place's OTHER rows. That IS recall.
    `truth_in_own` is a reported column and reads ~0.13 in runs.
    CONFIRM is the home half, WALK-ONLY the away half - THE TWO CHANNELS, already split.
    353's two-way is the cardinality-correct comparison between them, already built, already
    passing. "It cannot keep home while going deep" was a CHANNEL problem all along.
So 359 rediscovered, torch-free, a structure the stage already has - and the interleave result
says that structure is the right one.

WHAT SURVIVES IS ONE LITERAL, AND IT IS LARGE. `min_fillers` was HARD-CODED TO 2 at the
frame_assertions call - not a flag, never varied since the first frame commit. Constant frames
have never been on the stage's tape. Admitting them: own .2118 -> .3955, union .322 -> .477.

## 8-NEXT-5. 360: `--min-fillers 1`. One literal, one lever, no new mechanism

DONE IN THE STAGE: `--min-fillers` added (DEFAULT 2, so every existing arm is unchanged),
threaded to `frame_assertions`, and written into the report because two arms differing only in
it produce entirely different tapes and nothing else in the file would say so. Static check
`360 WIRING OK`; _check301 and _check337 still pass.

    <arm> --min-fillers 1 --seed <s> --run-tag 360fill_s<s>      4 seeds
    CONTROL = 352deep/353twoway, already in hand (both are --min-fillers 2 by construction).

PRE-REGISTERED GATE:
  PASS   answerable and hit_rate both rise materially, AND the mind still beats every count
         rival (PICK, ROUTER) on the new tape.
  THE RISK, NAMED FIRST: min_fillers 1 admits FIXED PHRASES ("of the"), which are constant
         frames that carry nothing. They would inflate truth_in_own with junk and make the
         exam EASIER rather than the tape RICHER. THE CONTROL IS THE ONE THAT ALWAYS RIDES
         ALONG: the count rivals get the same tape. If a count closes the gap to Phi, the tape
         got easier and the lever is spent.
  Read `own_hit_rate` and CONFIRM against `hit_of_walk_only` - if home rises while walk-only
  falls, the tape has been diluted, not enriched.

## 8-RESULT-9. 360 PASSED ITS GATE. THE TAPE GOT RICHER, AND THE MIND WALKED AWAY FROM HOME

4 seeds, held_out, `--min-fillers 1` on the 353 arm (depth 2, two-way):

                      353 (mf 2)   360 (mf 1)
    reachable            0.541       0.626      +0.085
    hit_rate             0.445       0.549      +0.104
    own_hit              ~0.13       0.313      the constant frames arriving, as 359 predicted
    step                 0.970       0.674      and now 0.44-0.87 ACROSS SEEDS
    hit_of_walk_only     0.838       0.793      0.56-0.95 across seeds
    PICK pooled                   1031/1 of 1316  z +32.06
    ROUTER            1.00x       1.13-1.32x mind vs 0.66-1.10x counting - ALIVE AGAIN
    CONFIRM pooled                 8/141 of 1025  z -10.90   HOME LOST TO A COUNT

GATE AS WRITTEN: PASS. answerable and hit_rate both rise materially; the mind beats every count
rival on the new tape by more than ever (hit 0.514 vs own-count 0.235 vs walk-count 0.158 at
s1337; walk-only 1031/1 pooled). THE NAMED RISK DID NOT HAPPEN - the counts did not close the
gap, so the tape got RICHER, not easier.

THE HEADLINE, END TO END: coverage 0.5937 x hit 0.549 = 0.326 OF EVERY TOKEN POSITION IN
WIKITEXT, against 0.25 before. From deleting one hard-coded literal.

THE DEFECT, AND IT IS NOT WHAT I FIRST WROTE. I guessed the own worlds were rowless and Phi
could not see multiplicity. WRONG - `reach_world` puts the whole question place, all its rows
and their repeats, into EVERY world, so the count is visible. The mind is not blind, it is
OVER-STEPPING: step 0.674 on a tape where home now answers 31% of questions by itself, so it
walks away from free wins and the own-count rival takes them (own_rival_of_own 0.84-0.92
against the mind's hit_of_own 0.53). This is a POLICY imbalance, not an information one, and
CONFIRM is a conditional on the home subset while hit_rate - the absolute - went UP.

## 8-NEXT-6. 361: is depth still needed once the tape holds constants?

The cheapest honest question, and it needs NO new code. Depth 2 was won (352/353) on a tape
whose one-hop reach was 0.127. At `--min-fillers 1` one hop reaches far more, and the mind
steps on two thirds of questions while home sits unclaimed. DEPTH AND CONSTANT FRAMES MAY BE
SUBSTITUTES, not complements.

    <arm without --reach-depth 2> --min-fillers 1 --seed <s> --run-tag 361flat_s<s>   4 seeds
    CONTROL = 360fill, in hand.

  PASS  hit_rate at least 360fill's 0.549 AND CONFIRM back to z >= 0 pooled. Then the walk was
        compensating for a starved tape, the standing arm loses --reach-depth 2, and depth
        rejoins the closed list with a reason.
  FAIL  hit_rate falls materially -> depth and the constants are complements, both stay, and
        the over-stepping is its own lever (the stay-vs-go summary under a rich home).
  Either way READ THE ABSOLUTE: hit_rate first, CONFIRM second. CONFIRM is conditional on a
  subset whose size moves between arms.

## 8-RESULT-10. 361 FAILED CLEANLY. DEPTH AND CONSTANT FRAMES ARE COMPLEMENTS

4 seeds, `--min-fillers 1` WITHOUT `--reach-depth 2`:

                       360 (depth 2)   361 (depth 1)
    reachable              0.626           0.237
    hit_rate               0.549           0.278       HALVED
    hit_of_own             0.528           0.885       home reclaimed
    own_rival_of_own       0.885           0.885       and it TIES the count exactly
    step                   0.674           0.034       it stopped walking
    CONFIRM pooled      8/141  z -10.90   11/5  z +1.50
    WALK-ONLY pooled                       0/2 of 35   VOID, not a result

READ THE VOID FIRST. With step at 0.03 the walk is never taken, so GATE-WO (n 3-12 per seed),
PICK (n 0-1), DEPTH and every walk-only column measure NOTHING here. Only hit_rate and CONFIRM
are readable from this arm, which is exactly why the gate was written on the absolute.

VERDICT: FAIL on the pre-registered gate - hit_rate had to hold 0.549 and it fell to 0.278.
MY HYPOTHESIS WAS WRONG: one hop at min_fillers 1 reaches 0.237, better than min_fillers 2's
0.127 but nowhere near the 0.626 that depth 2 gives. The constants do not replace the walk;
they add to it. DEPTH 2 STAYS.

AND THE SECOND HALF OF THE GATE FIRED AS WRITTEN: home came back (CONFIRM z +1.50), and it came
back by DEGENERATION - the mind stops walking, answers home, and TIES the own-count exactly
(0.885 vs 0.885). It does not beat the count at home; it only ever beat counts by ROUTING.
That also settles what 360's home deficit was: hit_of_own 0.885 when it stays against 0.528
when it may step, so the loss is STEPPING AWAY from free wins, not mis-picking at home.

THE STANDING ARM IS `--reach-depth 2 --two-way --min-fillers 1`, hit_rate 0.549, end to end
0.5937 x 0.549 = 0.326 of every token position.

## 8-NEXT-7. 362: over-stepping is now its own lever, as 361 pre-registered

Headroom is measured, not assumed: ceiling 0.6703 against hit 0.5137 at s1337 - about 0.16
lost between "the answer is available" and "the mind said it", almost all of it routing.

THE LEVER: `--two-way-by margin` at min_fillers 1, depth 2. IT IS A SPENT LEVER BEING RE-RUN,
AND THE NEW REASON IS NAMED: at min_fillers 2 the home branch had no constant frames, so the
gap between its best two own worlds was noise and margin was scoring nothing - it lost, and it
paid with depth (walk-only absolute 0.83 -> 0.72). At min_fillers 1 a constant place gives the
best own world a large, REAL gap over the second, and that gap is precisely the quantity that
says "home holds it". Margin is the only summary that can see it; max cannot, because max is
what the walk also reports and the two are incommensurable (329/337/340: Phi has no absolute
scale, only the gap carries).

    <standing arm> --two-way-by margin --seed <s> --run-tag 362marg_s<s>    4 seeds
    CONTROL = 360fill, in hand.

  PASS  hit_rate >= 0.549 AND CONFIRM pooled z >= 0. Home reclaimed without paying for it.
  FAIL-AND-CLOSE  hit_of_walk_only ABSOLUTE falls the way it did at min_fillers 2. Then margin
        does the same thing on any tape, the summary question is closed for good, and the next
        lever must change what the branches ARE rather than how they are summarised.
  Read hit_rate first. CONFIRM is conditional on a subset that moves between arms, and 361 just
  showed a way to win it that costs half the accuracy.

## 8-RESULT-11. 362 HIT THE FAIL-AND-CLOSE CLAUSE. THE SUMMARY QUESTION IS SHUT

                       360 (max)   362 (margin)
    hit_rate             0.549       0.436        FAIL, needed >= 0.549
    hit_of_walk_only     0.793       0.398        THE CLAUSE: the absolute collapsed again
    step                 0.674       0.294        it did shift routing home, as designed
    CONFIRM pooled     8/141 -10.90  26/36 -1.27  better, still negative
    PICK pooled                      541/2 of 1258  z +23.13
    GATE-WO 5%/10%                   1.0000 / 1.0000 vs share .54/.45

MARGIN DOES THE SAME THING ON EVERY TAPE. At min_fillers 2 it bought home with walk-only
0.83 -> 0.72; at min_fillers 1 it bought home with 0.79 -> 0.40. Twice, on two different tapes,
for the same reason: the gap between a branch's best two worlds is a property INSIDE the branch
and says nothing about the branch against another. `max` stands. HOW TO SUMMARISE A BRANCH IS
CLOSED - by pre-registration, not after the fact. The next lever must change WHAT THE BRANCHES
ARE.

Note what did NOT break: PICK 541/2 and GATE-WO 1.0000 at the top decile. Phi's ranking and its
knowledge of its own correctness survive every routing change. It is only the stay-vs-go
comparison that has never been solved.

STANDING ARM UNCHANGED: `--reach-depth 2 --two-way --min-fillers 1`, hit_rate 0.549.

## 8-NEXT-8. 363: THE MISSING CRITERION - the reason the fourth trait was never tested

Three of step 0's four traits were tested and closed. The fourth - PRODUCE WHAT WAS NEVER
OFFERED - was NEVER TESTED, and not because of the substrate. EVERY REWARD HERE IS "SAY THE
TOKEN THAT WAS IN THE HOLE". A produced value that is not that token scores zero whether it is
brilliant or gibberish, so generation cannot fail - IT CANNOT BE MEASURED. No architecture
fixes an exam.

`_audit363_useful.py` tests a label-free criterion, torch-free. A VALUE IS WORTH SAYING IF
SAYING IT CONNECTS:

    U(c) = positions elsewhere holding c that sit at a place already sharing a filler with this
           one. The question's own place never counts - an answer cannot corroborate itself.

Defined for ANY value in the vocabulary, including ones the offer never proposed, which is the
whole point: a count cannot score a value it has never seen at such a place.

TESTED WHERE IT IS HARDEST AND WHERE IT MATTERS: the holes NEITHER CHANNEL REACHES - truth not
among the place's own fillers, not in the walk's offer. Every number this project has printed
is zero there. The truth is dropped into a pool of random vocabulary and U must find it.

    AUC_U      truth against --pool random values by U
    AUC_FREQ   the same by raw corpus frequency        the dumb label-free rival
    AUC_NULL   U against a RANDOM OTHER PLACE's neighbourhood - same shape, wrong place
  GATE  AUC_U - AUC_FREQ > 0.05 AND AUC_U - AUC_NULL > 0.05.
  ~FREQ -> "useful" means "common", the criterion is the count we already had.
  ~0.50 -> the fourth trait is closed FOR WANT OF A MEASURE, not for want of a mind. Worth
           knowing exactly as much: it says the ceiling is the exam, not the architecture.
  Verified on a hand-made tape: U counts a sharing place, ignores a non-sharing one, and
  returns 0 when only the question's own place is in the neighbourhood.

    python _audit363_useful.py
    python _audit363_useful.py --min-fillers 1 --pool 128

## 8-RESULT-12. 363 FIRST PASS: the uniform pool was the wrong question

    pool 64:  USEFUL 0.6206   FREQ 0.7051   NULL 0.5238
    pool 128: USEFUL 0.6216   FREQ 0.7044   NULL 0.5252

TWO SIGNALS, NOT ONE, AND THE POOL CONFLATED THEM. U beats its wrong-place null by +0.097 -
there IS a hole-specific residue. But FREQ gets 0.70 FOR FREE: a uniform draw from the
vocabulary is mostly hapaxes, so "commoner than noise" finds the truth without knowing anything
about the hole. The pool was measuring the easy channel on top of the interesting one.

THE FIX IS A CONTROL, NOT A NEW SCORE: `--match-freq` draws distractors from the truth's own
frequency band (bit_length bands, widened until the band holds enough). Frequency is deleted BY
CONSTRUCTION - FREQ must read ~0.5 or the matching failed and nothing may be read (the sanity
row is printed first). What survives is the part of U that is about THIS hole, the only part
that could ever score generation.

  GATE (matched)  AUC_U - 0.5 > 0.05 AND AUC_U - AUC_NULL > 0.05, with FREQ within 0.08 of 0.5.
  If nothing survives matching, the fourth trait stays closed for want of a measure - with the
  frequency excuse removed, which makes the closure final.

    python _audit363_useful.py --match-freq
    python _audit363_useful.py --match-freq --pool 128

MATCHED RUN: the gate passes as written, and the gate was not strict enough.

    pool  64:  USEFUL 0.5999  FREQ 0.5573  NULL 0.5069   over freq +0.0426  over null +0.0930
    pool 128:  USEFUL 0.6010  FREQ 0.5776  NULL 0.5136   over freq +0.0234  over null +0.0874

  - Gate as declared: AUC_U - 0.5 = +0.10, over null +0.09, FREQ inside 0.08 of 0.5. PASS.
  - BUT FREQ IS NOT DELETED: 0.557 and 0.578, and at pool 128 it is at the edge of the sanity
    band. bit_length bands are coarse (band 4 = frequency 8-15) and the truth is still commoner
    than its band-mates. U over FREQ is only +0.024..+0.043 - UNDER the original gate.
  - AND THE NULL MAY BE DEAD RATHER THAN BEATEN. If most pool values score zero, ties are
    counted as half and an AUC of 0.507 is what a DEAD null looks like, not a fair one.
  - AUC 0.60 in a pool of 64 is not a reward. A reward needs the truth AT THE TOP, not above
    60% of strangers.

ADDED, all three are diagnostics rather than new scores:
    PLACES   the same count over DISTINCT PLACES instead of positions. A value repeated nine
             times at one place connects to ONE place; connecting BROADLY is what "useful" was
             supposed to mean, and this deletes the frequency the coarse bands left.
    ALIVE?   truth-nonzero rate, whole-pool-zero rate, null-all-zero rate. Reads the null on an
             ABSOLUTE quantity, which section 2 has cost us before.
    USABLE   p@1 and p@5 against chance - the number a reward would actually have to live on.
  Verified on a hand-made tape: 4 positions across 2 places reads (4, 2).

  READ IN THIS ORDER: ALIVE? first (a dead null voids everything below it), then USABLE, then
  PLACES over its own null. If p@5 is at chance, there is a correlation and not a criterion.

DIAGNOSTICS BACK. THE NULL IS ALIVE, THE SIGNAL IS REAL, AND IT IS NOT A REWARD.

                     pool 64    pool 128
    PLACES AUC        0.6031     0.6037     over its own null +0.096 / +0.089
    null all-zero     0.363      0.301      NOT dead by ties - the comparison is fair
    truth nonzero     0.325      0.325
    p@1               0.0287     0.0184     chance 0.0154 / 0.0078   ~1.9x / 2.4x
    p@5               0.1341     0.0766     chance 0.0769 / 0.0388   ~1.7x / 2.0x

  - The null is NOT dead, so +0.09 over it is a real hole-specific signal. That much stands.
  - But the whole thing is ABOUT TWICE CHANCE. As a reward, the top-1 by U is wrong on 97-98%
    of exactly the holes generation would have to serve. A correlation, not a criterion.
  - AND THE CEILING IS STRUCTURAL: U(truth) IS ZERO ON 67.5% OF THESE HOLES. The truth has no
    position at any place sharing a filler with this one, so no connection-based criterion can
    ever see it. Two thirds of the target subset are VOID for this measure, not failed by it.

WHAT IS BEING ADDED, and it is the last cut before a verdict: the same numbers SPLIT BY
NEIGHBOURHOOD SIZE. Conditioning on U(truth) > 0 would let the label choose the subset - the
selection would be the answer - so the split is on |neigh|, which is known before any answer
exists. If the criterion is strong where the hole has many neighbours, that is usable: it says
WHEN to trust it. If it is flat across quartiles, the 2x is spread thin everywhere and there is
nothing to build a reward on.

THE SPLIT CAME BACK MONOTONE ON BOTH POOLS - NOT FLAT, WHICH WAS THE USABLE BRANCH:

    quartile   neighbours    AUC places   p@5 (pool 64, chance .0769)   truth nonzero
      q1          1-9          0.5169              0.0472                   0.056
      q2          9-41         0.5969              0.1445                   0.260
      q3         41-168        0.6333              0.1534                   0.428
      q4        170-1046       0.6651              0.1912                   0.556

q1 IS AT CHANCE and holds a quarter of the subset; q4 is 2.5x chance on p@5 and AUC 0.665. The
criterion is not uniform noise - IT KNOWS WHEN IT APPLIES, and |neigh| is known before any
answer exists, which is exactly what makes it usable as a reward's precondition.

AND THE DECOMPOSITION THE TREND DEMANDS. A mean AUC can rise two ways: the criterion
DISCRIMINATES better, or it is merely DEFINED more often - when the truth scores zero and so
does most of the pool, ties drag that question to ~0.5. Back-of-envelope on both quartiles
gives AUC-given-defined near 0.80 in q1 AND in q4, which would mean THE ENTIRE TREND IS
COVERAGE. That matters more than anything else in this audit: silence is fixable with density,
bluntness would not be.

So the audit now prints `AUC|defined` and `AUC|silent` per quartile, with the recombination
checked (P(def)*AUC|def + (1-P)*AUC|silent reproduces the mean exactly on a synthetic case).
IT IS LABEL-CONDITIONED AND MAY NEVER BE A GATE - selecting on U(truth) > 0 lets the answer
choose the subset. Diagnostic only, and it decides the next run rather than any claim.

## 8-NEXT-9. 364: does the criterion's COVERAGE scale with density?

If AUC|defined is flat and coverage is the whole story, then the criterion is already sharp and
merely starved - and neighbourhood size is a function of how much tape is in front of it.

    python _audit363_useful.py --match-freq --window-lines 1600
    python _audit363_useful.py --match-freq --window-lines 3200

  PASS  truth-nonzero rises with the window AND overall p@5/chance rises with it, while
        AUC|defined stays ~0.80. Then the measure is real, starved rather than blunt, density
        is its lever, and generation can be posed as a task on the covered part.
  FAIL  coverage flat, or AUC|defined falls as coverage grows (the new neighbours are noise).
        Then 2x chance is the ceiling of connection-counting and the fourth trait closes for
        want of a measure - with the frequency excuse and the density excuse both removed.
  NOTE the standing result this does NOT contradict: 335 closed CORPUS SIZE as a lever for the
  MIND's advantage. This asks whether a CRITERION becomes defined more often. Different
  quantity, and it has never been asked.

## 8-RESULT-13. 364: DENSITY IS THE LEVER. THE CRITERION WAS STARVED, NOT BLUNT

    window        400      1600     3200
    places        5340    24004    50458
    truth nonzero 0.325    0.561    0.621     COVERAGE NEARLY DOUBLES
    AUC places    0.603    0.674    0.673
    over FREQ    +0.043   +0.092   +0.091     clears the ORIGINAL gate from 1600 on
    over null    +0.096   +0.152   +0.161
    p@5           0.134    0.248    0.216     chance 0.077 -> 1.7x, 3.2x, 2.8x
    p@1           0.029    0.067    0.053     chance 0.015 -> 1.9x, 4.3x, 3.4x
    AUC|defined   0.904    0.870    0.840     LABEL-CONDITIONED DIAGNOSTIC

VERDICT: PASS from 400 to 1600, decisively - coverage 0.33 -> 0.56, p@5 1.7x -> 3.2x, and the
criterion pulls AWAY from frequency (+0.043 -> +0.092), which is the line that had been under
the original gate. SATURATION at 3200: coverage still climbs but p@5 and AUC do not.

AND THE ONE HONEST QUALIFICATION: AUC|defined FALLS, 0.904 -> 0.870 -> 0.840. My gate said
"stays ~0.80" for PASS and "falls as coverage grows" for FAIL, and it does both - it declines
while staying far above 0.80. The truthful reading is DILUTION, not noise: doubling coverage
admits harder holes, and the criterion stays very sharp where it speaks. It is not the clean
PASS I wrote, and it is not the FAIL either. Saying so is the point of pre-registering.

A SWEET SPOT APPEARED, AND IT IS KNOWN BEFORE THE ANSWER. The top quartile is no longer the
best once the tape is dense: at 1600 the best band is q3 (206-824 places, AUC 0.750) and at
3200 it is q2 (72-508, AUC 0.721), with q4 falling to 0.645 / 0.657. Too few neighbours and the
criterion is silent; too many and it is diluted. SUSPECT NAMED: `--cap 500` bounds positions
read per value, and at 50458 places a frequent value hits it - the q4 decline may be the cap
rather than the tape. Check with `--cap 2000` before believing the shape.

## 8-NEXT-10. U USED FORWARD, AS A THIRD CHANNEL (built into 363, no new file)

The criterion argument can stop here, because there is a version of this that needs NO reward
and NO criterion: walk the neighbourhood once, count for every value HOW MANY NEIGHBOURHOOD
PLACES HOLD IT, drop the place's own fillers and the walk's offer, and OFFER THE TOP EIGHT. The
truth is the label, exactly as everywhere else in the project.

    recall (home) . substitution (the walk) . CONNECTION (this)

`u_reach` = the share of the CURRENTLY UNREACHABLE holes that the connection count reaches at
top-8. Those holes are 50% of all questions at min_fillers 1 and every number the project has
printed for them is zero, so anything above zero here is new reach, not a re-description.
Verified: forward U equals the per-candidate count over distinct places, own fillers and offer
values are excluded, and the neighbourhood cap subsamples rather than truncates.

    python _audit363_useful.py --match-freq --window-lines 1600
    python _audit363_useful.py --match-freq --window-lines 3200 --cap 2000

## 8-RESULT-14. THE THIRD CHANNEL IS REAL AND MODEST. STRICT IS DEAD, THE WEIGHT WINS

                        w1600      w3200
    u_reach  plain      0.0933     0.0737     out of 2425 / 4985 values
    uw_reach WEIGHTED   0.0975     0.0758     best on both
    us_reach STRICT     0.0393     0.0351     out of 1133 / 2569 - WORSE PER CHANCE TOO
    AUC plain           0.6739     0.6844
    AUC weighted        0.6918     0.7029     +0.018 on both
    chance at top-8     0.0033     0.0016     so the weighted form is ~30x and ~47x

STRICT IS CLOSED. Requiring two shared fillers halves the candidate pool and MORE than halves
the reach: 5.6x chance against the plain form's 28x. The dilution seen in the quartiles is a
VOLUME effect, not weak neighbours being noise - single-filler neighbours carry the signal and
throwing them away throws away the channel. My reasoning for the threshold was wrong and the
run says so.

THE OVERLAP WEIGHT IS A SMALL CONSISTENT WIN and it is the form to build: +0.004 reach and
+0.018 AUC at both windows, parameter-free, same single pass.

WHAT IT IS WORTH, PLAINLY. Unreachable holes are 48% of all questions, so a channel reaching
0.0975 of them adds ABOUT +4.7 POINTS OF ABSOLUTE REACH on top of 0.626. Real, measured against
a null and against frequency, on holes where the project has only ever printed zero. Not a new
capability - a fourth thing the tape can be counted for.

AND THE HONEST STATE OF THE TWO HALVES, because it is easy to lose in the numbers:
  KNOWLEDGE keeps improving. min_fillers 1 bought +10 points of hit; this buys ~+5 of reach.
  THE MIND HAS NOT MOVED SINCE 353. No new decision has been earned: interleave beat choosing
  for two channels and will very likely beat it for three. Phi still routes, picks, calibrates
  and gates - the same four things, better fed.

## 8-NEXT-11. 365: THE THIRD CHANNEL IS IN THE STAGE. `--connect`, default OFF

    reach_connect(p, q, k)   neighbourhood = places sharing a filler with this one
                             overlap       = how many fillers each shares - the weight
                             score(v)      = sum of the overlaps of the places holding v
                             own place excluded, own values excluded (recall covers them),
                             retention honoured exactly as reach_places honours it

INTERLEAVED INTO `reach_candidates` BEFORE THE CAP, never appended. The offer stays at
REACH_CANDS, so this is a channel comparison and not a wider offer - `both` interleaves places
for the same reason, and 347 already measured what a wider offer costs. Weighted, not
thresholded: 363 closed the strict form (5.6x chance against the plain form's 28x).

`_check365_connect.py`, new, and it RUNS rather than only reading source: the ten static
properties, plus the ranking on a tape where the weighted and plain orders differ, plus the
interleave capped and deduped with the walk first. `365 OK`. _check301 30/30 and _check337 pass
unchanged.

    <standing arm> --connect --seed <s> --run-tag 365conn_s<s>       4 seeds
    standing arm = --reach-depth 2 --two-way --min-fillers 1
    CONTROL = 360fill, in hand: hit_rate 0.549, reachable 0.626.

PRE-REGISTERED GATE:
  PASS  hit_rate rises above 0.549 AND the mind still beats every count rival (PICK, ROUTER).
  FAIL  hit_rate flat or down. The channel reaches on paper and the offer cannot carry it -
        which is exactly 347's finding for thickness (support x2.2, offer x5.4, present@8 FELL)
        and would make it the second measurement of the same law.
  NOTE  `reachable` is DELIBERATELY unchanged - reach_reachable still measures the WALK's
        ceiling, so it stays comparable across every arm ever run. The connect channel shows up
        in hit_rate and in the pick, which is where it has to pay.
  WATCH the displacement: the offer is capped, so every connect candidate REPLACES a walk
        candidate. If hit_rate is flat while the mix changed, the two channels are redundant.

## 8-RESULT-15. 365 PASSED, SMALLER THAN PREDICTED, AND THE DISPLACEMENT WAS THE REASON

                        360fill    365conn
    hit_rate             0.549      0.567     +0.018   (I predicted +0.047)
    reachable            0.626      0.639
    ceiling              0.670      0.729
    hit_of_walk_only     0.793      0.867
    PICK pooled                  1183/0 of 1371   z +34.39
    ROUTER            1.13-1.32x  1.13-1.25x mind vs 0.78-1.09x counting - wins on all four
    GATE-WO pooled               1.0000 at EVERY fraction, 5% through 50%
    CONFIRM pooled     8/141 -10.90   2/229  z -14.94

GATE: PASS. hit_rate rose and the mind beats every counting rival by more than ever. But the
gain is +1.8 points against the +4.7 the audit predicted, and the named risk is exactly why:
THE OFFER IS CAPPED, SO EVERY CONNECT CANDIDATE REPLACES A WALK CANDIDATE. cand_places 4.00 ->
4.17, expand 10.77 -> 12.21. The channel pays, and displacement eats most of it - the second
measurement of 347's law, as pre-registered.

END TO END: coverage 0.5937 x 0.567 = 0.337 of every token position, from 0.25 four steps ago.

AND THE PATTERN IS NOW UNMISTAKABLE ACROSS 360, 362, 365: every improvement to the tape makes
the walk better and makes the mind abandon home harder. CONFIRM is 2/229. The mind has become a
pure walker. It is not wrong to walk - 0.567 against the own-count's 0.24-0.32 - but the
ceiling is 0.729 and the gap is 16 POINTS, nearly all of it home.

## 8-NEXT-12. 366: THE TWO BRANCHES WERE NEVER THE SAME SIZE

362 closed the summary question with "the next lever must change WHAT THE BRANCHES ARE". Here
is what they are, and it is an asymmetry nobody has looked at:

    stage one's own worlds   `world(v, [], 0)`      - the question place, NO imported rows
    the step under lookahead  max(l2), and every stage-two world is the question place PLUS
                              `budget` imported rows

SO "STAY" HAS ALWAYS BEEN COMPARED AGAINST SYSTEMATICALLY LARGER WORLDS - in a codebase whose
own stated rule, three functions above, is that silence must cost the same number of rows as
speaking, because Phi pools over rows and a row count competing with an answer is what undid
291. That rule was applied between REFUSE and the own worlds. It was NEVER applied between the
own worlds and the step.

`--own-import` (default OFF) builds each own world with the same import every candidate gets -
its homes elsewhere on the tape, `outside_mentions`, at the same `budget`. THE CLAIM IS CHECKED
RATHER THAN ASSERTED: `own_import_full` in the report is the share of own worlds that actually
reached the budget, and far below 1.0 means the branches are still unequal and the arm measured
something else. Static check: 8/8. _check301 and _check365 pass unchanged.

    <standing arm> --connect --own-import --seed <s> --run-tag 366own_s<s>     4 seeds
    CONTROL = 365conn, in hand: hit_rate 0.567, CONFIRM 2/229, ceiling 0.729.

  PASS  CONFIRM pooled z >= 0 AND hit_rate >= 0.567. Home reclaimed without paying for it -
        the thing 353 got half of and 362 could not get at all.
  FAIL  hit_rate falls -> equalising the branches costs more than the asymmetry did, and the
        stay/go decision is closed as unfixable by world construction. Then the honest end of
        the project is the number: 0.337 of wikitext from counts plus 5633 parameters.
  READ FIRST  `own_import_full`. Below ~0.8 the arm is VOID, not failed.
  WATCH  361 showed a way to win CONFIRM by degenerating into a home-answerer at half the
        accuracy. hit_rate is the guard against that and is read first.

## 8-RESULT-16. 366 FAILED. THE ASYMMETRY WAS LOAD-BEARING, AND STAY/GO IS NOW CLOSED

                        365conn    366own
    hit_rate             0.567      0.376     FAIL - the gate needed >= 0.567
    hit_of_own           0.656      0.820     home worked, exactly as designed
    hit_of_walk_only     0.867      0.280     and the walk fell apart
    step                 0.775      0.295     it stopped walking
    CONFIRM pooled     2/229 -14.94  15/74 -6.25   better, still negative
    PICK pooled                    391/2 of 1376  z +19.62 - still beats every count

THE LEVER DID WHAT IT WAS DESIGNED TO DO AND THAT IS WHY IT LOST. Equalising the world sizes
made "stay" competitive, home rose 0.66 -> 0.82, CONFIRM more than halved its deficit. But the
old imbalance was LOAD-BEARING: the mind needed the size bias to step at all, and without it
stepping fell 0.78 -> 0.30 and walk-only accuracy fell 0.87 -> 0.28. Net -19 points.

ONE NUMBER STILL OWED, AND IT IS THE VOID GUARD: `own_import_full` from any 366 report. Below
~0.8 the branches never actually became equal and this arm is VOID rather than failed. The knob
plainly moved its quantity (step 0.78 -> 0.30), so a DEAD KNOB is ruled out either way.

STAY/GO IS CLOSED, THREE WAYS. The summary (362, margin, closed). The branch construction (366,
sizes, closed). What remains is 353's two-way max, which took half of it and stands. And the
forbidden fourth option is TUNING THE IMPORT BUDGET between 0 and full - that is parameter
fitting, it is barred by the standing rule, and it is not proposed.

## 8-NEXT-13. 367: STOP CHOOSING. The project already measured that choosing is worth ~nothing

359 measured this exact question on the two channels and the answer has been sitting there
since: INTERLEAVING them lands within +0.026 / +0.032 of a PERFECT channel pick. The stage,
meanwhile, still spends its whole stage-one softmax on that choice - the one decision that has
failed three separate repairs.

`--own-in-offer` (default OFF) puts the home values into the SAME offer as the walk's, round
-robin across lanes, deduped, under the UNCHANGED cap. With `--stage2-always 1.0` the stay/go
decision disappears entirely and one softmax ranks every channel's values together. Home values
get the same import from the same source (`outside_mentions`) as every other candidate, so no
world is larger than another - the rule 366 established and then died proving.

    <standing arm> --connect --own-in-offer --stage2-always 1.0 --seed <s> --run-tag 367mix_s<s>
    4 seeds. CONTROL = 365conn: hit_rate 0.567, ceiling 0.729.

  PASS  hit_rate > 0.567. It approaches the 0.729 ceiling by NOT DECIDING, which is what 359
        predicted and what three failed repairs imply.
  FAIL  hit_rate <= 0.567 -> the choice was carrying something after all, and 0.337 end-to-end
        is the project's number.
  WHAT IT COSTS, SAID BEFORE THE RUN: the ROUTER result disappears BY CONSTRUCTION - a mind
  that never routes cannot beat a router. That is the project's oldest claim, and this arm
  trades it for accuracy. If it wins, the honest statement changes from "the mind routes better
  than every count" to "the mind RANKS better than every count, and routing was never the part
  that paid". PICK, GATE-WO, the transplant and the shuffle null all still ride along and still
  carry the separation claim.
  _check365_connect.py extended: 12 static rows and the three-lane round-robin, all pass.

## 8-RESULT-17. 367 FAILED HARD, AND IT EXPOSES THE LAW THE LAST FIVE STEPS HAVE BEEN OBEYING

                    365conn    367mix
    reachable        0.639      0.777     +0.14, the biggest offer this project has had
    ceiling          0.729      0.786
    hit_rate         0.567      0.257     COLLAPSE
    hit_of_walk_only 0.867      0.070     (0.251 / 0.020 / 0.002 / 0.007)
    hit_of_own       0.656      0.713     unchanged in substance
    PICK             0.94-0.97  0.27/0.03/0.005/0.013   THE PICKER DIED
    PICK vs count             103/0 pooled - it still never LOSES to a count

MY EXTRAPOLATION WAS WRONG AND I SHOULD NAME IT. 359 measured that INTERLEAVING beats CHOOSING
- but it measured it on REACH, "is the truth in the union". I carried it over to PICK. Reach did
exactly what 359 predicted (0.64 -> 0.78). Picking paid for all of it and more, because on a
WALK-ONLY question every home value in the offer is GUARANTEED WRONG: the round-robin spends
half the candidate budget on values that cannot be the answer. Interleaving is free when you
only ask whether the truth is present. It is expensive when something has to pick.

### THE LAW, VISIBLE ONLY NOW THAT THERE ARE FIVE POINTS

            reach   hit    hit/reach
    360fill 0.626  0.549     0.877
    365conn 0.639  0.567     0.887
    366own  ~0.64  0.376     0.59      policy broken
    367mix  0.777  0.257     0.33      policy broken

TWO KNOWLEDGE LEVERS WON (360 min-fillers, 365 connect - both were "count something more").
THREE POLICY LEVERS LOST (362 summary, 366 branch sizes, 367 no-choice - all were "rearrange
the decision"). Not one rearrangement of the decision has ever paid. And on the arms where the
policy survives, hit/reach sits at ~0.88 and will not move: THE PICKER IS AT ITS CEILING, and
every candidate added beyond it costs more than it brings.

### WHAT WE HAVE BEEN DOING WRONG

Chasing the decision. Five steps of it. The evidence says the decision is not where the slack
is - the picker is saturated and the tape is not. The one thing that has never been re-asked
under a rich offer is THE PICKER'S OWN SIZE.

342a closed capacity: d=32 against d=64, flat, invariant intact. IT WAS MEASURED AT reach 0.127
WITH 8 CANDIDATES OF WHICH ~1 WAS EVER RIGHT. The offer is now six times richer and the picker
is provably the binding constraint. That is the same shape of reason that reopened depth (352,
against 322) and it is a named new reason, not a re-litigation.

## 8-NEXT-14. 368: RE-ASK CAPACITY ON A RICH OFFER. No new code

    <standing arm> --connect --dim 64 --seed <s> --run-tag 368wide_s<s>      4 seeds
    standing arm = --reach-depth 2 --two-way --min-fillers 1
    CONTROL = 365conn, in hand: reach 0.639, hit 0.567, hit/reach 0.887.

  PASS  hit/reach rises above 0.887 AND the invariant holds (transplant, native control,
        shuffle null all ride along as always). Then the picker was the constraint, capacity is
        the lever ON A RICH OFFER, and 342a's closure was scoped to a starved one.
  FAIL  hit/reach flat -> the 0.88 is the SUBSTRATE's, not the decider's: 12% of the truths in
        the offer are indistinguishable from their distractors by any reading of the tape.
        Then the project is finished at 0.337 end-to-end, and that is a result, not a defeat.
  READ  hit/reach FIRST. hit alone moves with the tape; the ratio is the picker.
  RISK NAMED: d=64 is 21505 parameters. The invariant is a TEST, never a parameter count
        (section 1), and 342a already showed it holds at that width - but it is re-checked here
        rather than assumed, because the offer it is holding over is new.

## 8-RESULT-18. 368: CAPACITY IS NOT THE LEVER ON A RICH OFFER EITHER. THE 0.88 IS THE TAPE'S

                     365conn (5633)   368wide (21505)
    reachable            0.639            0.670
    hit_rate             0.567            0.578      +0.011 for 3.8x the parameters
    HIT/REACH            0.886            0.863      FELL. The gate needed > 0.887
    per seed                              0.882 0.844 0.834 0.893
    CONFIRM pooled     2/229 -14.94     23/82 -5.76   best since 360
    PICK pooled        1183/0 +34.39    1060/1 +32.51
    GATE-WO pooled     1.0000 all       1.0000 all, 740/740 at 50%

FAIL, on the number the gate named. 3.8x the parameters bought +1.1 points of hit and NOTHING
in picking efficiency - the small hit gain came from reach moving (0.639 -> 0.670), not from
better discrimination. 342a said capacity is not the lever; it said so on a starved offer, the
objection was fair, and the answer is the same on a rich one.

SO THE 0.88 BELONGS TO THE SUBSTRATE. About 12% of the truths that ARE in the offer cannot be
told from their distractors by any reading of this tape - and that has now survived five
policy rearrangements and a 3.8x capacity test. It is not the decider's limit. It is what
substitution-plus-recall-plus-connection can distinguish.

END TO END: coverage 0.5937 x hit 0.578 = 0.343 of every token position in wikitext.

## 9-FINAL. WHAT THE PROJECT HAS ESTABLISHED, in the form the evidence supports

1. KNOWLEDGE AND DECISION CAN BE SEPARATED. Four independent controls: transplant to two
   foreign corpora, fine-tuning buys nothing, a native mind is indistinguishable from the
   transplant (73/71 of 8000, z +0.17), and the shuffle-tape null erases everything.
2. THE DECIDER IS SMALL AND IT IS NOT A STORE. 5633 parameters, `not_loaded` ink, and every
   improvement to accuracy this session came from COUNTING MORE, never from the decider.
3. WHAT THE DECIDER IS ACTUALLY GOOD AT, and it is not what we claimed at step 0: it RANKS
   (PICK 1060/1 pooled against every counting rival) and it KNOWS WHEN IT IS RIGHT (GATE-WO
   1.0000 at every fraction from 5% to 50%). ROUTING was the claim; five attempts to improve
   the routing all failed and one of them (367) showed the routing can be removed and the
   ranking survives. The honest claim is RANKING AND CALIBRATION, not routing.
4. THE LIMITS ARE THE TAPE'S, and they are now numbered:
       coverage    0.59 of all token positions are askable at all
       hit/reach   0.88 of the truths that reach the offer can be distinguished
       product     0.343 of every token position, from counts plus 5633 parameters
5. OF STEP 0'S FOUR TRAITS: three tested and closed (compose 310, remember 354/355, revise
   361-366 as the stay/go decision). The fourth - PRODUCE WHAT WAS NEVER OFFERED - is closed
   FOR WANT OF A MEASURE, not for want of a mind: 363 found a real label-free criterion at
   about twice chance, sharp where it speaks (AUC|defined ~0.84) and silent on two thirds of
   the holes that need it.

## 9-NEXT. THE ONE LEVER LEFT WITH EVIDENCE BEHIND IT, and it is small

Everything that paid this session was "count more" (360: +10 points, 365: +2). Everything that
rearranged the decision lost (362, 366, 367) and capacity lost twice (342a, 368). The product
is coverage x hit, and COVERAGE HAS NEVER BEEN PUSHED IN THE STAGE - every arm runs a 400-line
region with --addresses 1500. 364 showed the connect criterion's coverage nearly doubles from
400 to 1600 lines.

    <standing arm> --connect --addresses 4000 --seed <s> --run-tag 369dense_s<s>
    CONTROL = 365conn. READ hit/reach: if it holds ~0.88 while coverage rises, the product
    rises with it. If hit/reach falls, density buys reach and loses picking, which is 347's law
    for the third time and the end of the tape side too.
  335 IS NOT CONTRADICTED: it closed corpus size as a lever for the MIND'S ADVANTAGE over
  counts. This asks whether the PRODUCT rises. Different quantity, the same distinction 364
  made and confirmed.

## 8-RESULT-19. 369 FAILED, AND I MIS-SPECIFIED IT

                     365conn    369dense (--addresses 4000)
    reachable         0.639       0.579
    hit_rate          0.567       0.459
    hit/reach         0.886       0.793
    questions/seed     ~840       ~4400
    PICK pooled     1183/0 +34.4  3391/25  z +57.59
    GATE-WO pooled   1.0000       0.996 / 0.990 / 0.991 / 0.969

MY ERROR, NAMED: `--addresses` is not a coverage lever. 356's coverage - the share of token
positions that are askable at all - is a property of the WRITE PATH (frame_max, min_fillers)
and is already maximal at min_fillers 1. `--addresses` only changes HOW MANY PLACES A RUN
SAMPLES. So 369 did not test what its gate was written for. It tested more places at an
unchanged walk budget, and that loses.

WHAT IT DID SHOW, and it is the part that matters for scaling: on 5x the questions the MIND'S
ADVANTAGE OVER COUNTS GREW - 3391/25 pooled, z +57.59, the largest margin this project has
recorded - and the gate stayed at 0.97-0.996. THE SEPARATION RESULT STRENGTHENS WITH SCALE.
Accuracy fell because the question distribution moved to thinner places, not because the mind
got worse at deciding.

THE HONEST SPLIT FOR ANY FUTURE SCALING:
    scales UP with data   the mind-vs-count margin, the gate, the invariant
    does NOT scale        hit/reach - it is set by the RELATION SET on the tape, not by size

## 10-NEXT. LETTING GO OF THE HAND-WRITTEN RELATIONS. Three steps, the second decides

THE DIAGNOSIS. The tape has three relations and a human wrote all three. Phi chooses among
their OUTPUTS and has never chosen WHAT TO COUNT - which is why more corpus gives a better
index and not a different mind. But all three are the SAME SHAPE: score OTHER PLACES given THIS
place, then read off their fillers. That shape is enumerable, so the space can be measured
before anything is made learnable.

  370  is there anything to choose among the THREE? The 359 design on three channels. Skipped
       for now: 359 already answered it for two (choosing beats interleaving by +0.03) and the
       answer does not block anything.

  371  IS THE SPACE BIGGER THAN THREE? BUILT: `_audit371_family.py`, torch-free.
       Twelve members, every one a COUNT with no fitted constant, all at the SAME top-m so no
       relation can win on budget:
         own · share_1/2/3 · share_w · rare_w · common_w · mention_w · cos_k (Jaccard, not a
         cosine - no embedding) · two_hop · same_line · len_match
       `rare_w` and `common_w` are opposite weightings of the same sharing and DISAGREE where
       share_1 is blind - verified on a hand-made tape, so the family is not one relation in
       twelve costumes by construction. `common_w` is in precisely because a family containing
       only the members I find plausible is a family I chose. `len_match` is the family's own
       null: same shape, knows nothing about content. `same_line` is in even though 304 closed
       the line CHANNEL - as one member of a family it costs nothing to ask.
       GATE  oracle over the family - best single member > 0.05.
       Also printed: the SOLE-winner rate, because a family where one member is nearly always
       the only one that reaches is a family with one useful relation and a tail.

         python _audit371_family.py
         python _audit371_family.py --window-lines 1600

  372  ONLY IF 371 PASSES. The relation arrives as EVIDENCE ON THE CANDIDATE - which relation
       offered it, with what count - and Phi ranks candidates as it always has. NOT a second
       head, which is the 4x law (321, 341, 352), and NOT another lane in the offer, which is
       what 367 measured and lost. One head, one price, and the invariant holds because a
       relation is corpus-independent: the transplant must still carry.

## 10-RESULT. 371 FIRST PASS: THE GATE PASSES AND THE GATE IS NOT SAFE TO READ YET

    window 400                       window 1600
    own        0.4060                own        0.4087
    mention_w  0.0507                mention_w  0.0513
    rare_w     0.0487                common_w   0.0487
    share_1    0.0480                share_1    0.0480
    share_w    0.0480                share_w    0.0480
    common_w   0.0480                rare_w     0.0433
    two_hop    0.0473                cos_k      0.0413
    cos_k      0.0400                two_hop    0.0393
    share_2    0.0253                share_2    0.0367
    same_line  0.0127                share_3    0.0220
    share_3    0.0120                same_line  0.0127
    len_match  0.0080  null          len_match  0.0107  null
    ORACLE 0.4973 over best +0.0913  ORACLE 0.5107 over best +0.1020

TWO THINGS BEFORE THIS IS BELIEVED, AND BOTH ARE MINE.

1. A BUG, CAUGHT BY MY OWN RULE. `share_1` and `share_w` print 0.0480 at four decimals on both
   windows. Identical digits are a collided tag (section 2) - and they were: I wrote both as
   `len(vs)`. LITERALLY THE SAME MEMBER TWICE. A duplicate cannot add to a union so the oracle
   is unaffected, but the family was eleven relations wearing twelve names. Fixed: share_w is
   now `len(vs)/len(fillers[j])` - how much OF THAT PLACE the sharing covers, which prefers a
   small focused neighbour over a large loose one and breaks ties share_1 cannot see. Verified.

2. THE GATE IS UNREADABLE WITHOUT A FAMILY-SIZED NULL, and this is the same mistake I have
   caught three times elsewhere. A PERFECT CHOOSER OVER TWELVE RELATIONS GETS TWELVE SHOTS AT
   TOP-8. "The oracle beats the best single member" is partly just shots on goal. The eleven
   place-relations sit at 0.04-0.05 each and the null member at 0.008-0.011, so eleven random
   place-sets would ALSO lift a union above own's 0.406 by simple accumulation.
   ADDED: `oracle_null` - own plus ELEVEN RANDOM PLACE-SETS, same places_k, same top-m, same
   number of shots. GATE IS NOW BOTH: oracle - best > 0.05 AND oracle - oracle_null > 0.05.

WHAT THE FIRST PASS DOES ESTABLISH REGARDLESS: `own` is 0.406 and every place relation is
0.04-0.05. SOLE-winner is 0.406 for own and at most 0.014 for anything else. WHATEVER THE NULL
SAYS, THE PLACE RELATIONS ARE ALL WEAK AND HEAVILY OVERLAPPING - the space, if it exists, is
not twelve different ways of reaching, it is one strong relation (recall) and a crowd of faint
ones that occasionally differ.

    python _audit371_family.py
    python _audit371_family.py --window-lines 1600

## 10-RESULT-2. THE FAMILY NULL FIRED, AND THEN THE POPULATION TURNED OUT TO BE DILUTED

    window 400                        window 1600
    own          0.4060               own          0.4087
    ORACLE       0.4993               ORACLE       0.5107
    NULL FAMILY  0.4653               NULL FAMILY  0.4620
    over null    +0.0340  UNDER GATE  over null    +0.0487  UNDER GATE, and it GREW

The null I added did its job: most of the +0.09 headroom was TWELVE SHOTS AT TOP-8, not twelve
relations. Read on the whole population, the gate fails.

BUT THE WHOLE POPULATION IS THE WRONG POPULATION, and this is arithmetic, not an excuse. `own`
alone answers 41% of questions. On those, NOTHING a place relation does can show - the answer
is already home. Re-reading both oracles on the 59% where OWN FAILS, which is the only place a
relation could ever matter:

                       relations   random family   difference
    window 400           0.1571       0.0998        +0.0572
    window 1600          0.1725       0.0901        +0.0824

ABOVE THE GATE ON BOTH, AND IT GROWS WITH THE WINDOW. The subset is defined by a THIRD arm's
outcome (recall's), identically for both arms, so the comparison between them is fair - the
absolute numbers are conditional and are labelled as such, the same discipline 363's
AUC|defined got.

AND THE PER-MEMBER EVIDENCE SAID THIS ALL ALONG: each place relation reaches 0.04-0.05 alone
while the null member `len_match` reaches 0.008-0.011. A single relation is FOUR TO SIX TIMES a
single random set. It was only the UNION that saturated, because eleven weak overlapping
relations and eleven random sets both fill a 96-slot union.

SO: NOT BAD LOGIC AND NOT BAD DATA - A DILUTED METRIC. The instrument was measuring a
population that is 41% already-answered. The audit now reports the own-fails cut as well, and
that is the number 372 rests on.

    python _audit371_family.py
    python _audit371_family.py --window-lines 1600

## 10-NEXT-2. 372 BUILT: THE COMPASS IS A RELATION, AND THERE ARE NOW SEVEN

Until now the stage could steer by exactly two hand-written rules: `cos` (the fingerprint) and
`share` (which IS 371's mention_w). `--reach-compass` now also takes five more of 371's family,
each a count with no fitted constant:

    share1    how many DISTINCT values are shared
    rare      weighted by 1 / the shared value's corpus mentions - a shared rare word says more
    common    the opposite weighting, in as its own control
    cover     how much OF THE NEIGHBOUR the sharing covers - prefers small focused places
    jaccard   the two filler sets against their union

NO HEAD, NO LANE, NO BUDGET CHANGE. The walk returns the same k places, chosen by a different
count. That is deliberately NOT "Phi picks the relation" yet: 371 measured reach CEILINGS, and
before handing Phi a choice among eleven we need to know what the stage actually DOES under
each one - with the rivals, the shuffle null and the transplant riding along, none of which a
torch-free ceiling can give.

`_check372_compass.py`, new and it RUNS: six static properties (each member reachable, the new
branch taken before the old one, retention honoured so two compasses are not two tapes, the own
place excluded, `both` still the only interleaver, argparse updated) plus the five members
ordering one hand-made tape THREE different ways.

AND THE CHECK CAUGHT SOMETHING WORTH KEEPING. My first test tape gave one neighbour BOTH shared
values and it won under four members of five. Not a bug - the honest shape of the family:
summing over shared values makes "shares MORE values" dominate every weighting, which is
exactly why 371 found all eleven place relations at 0.04-0.05 and heavily overlapping. The
weightings only separate when the neighbours share the SAME NUMBER of values.

    <standing arm> --connect --reach-compass <m> --seed <s> --run-tag 372<m>_s<s>
    m in (share1, rare, cover, jaccard) - `common` only if one of those wins, as its control.
    CONTROL = 365conn (compass cos): hit_rate 0.567, reach 0.639, hit/reach 0.886.

  PASS  any member's hit_rate > 0.567 with the mind still beating every count rival. Then the
        compass was a hand-written choice that cost us, and 373 - Phi choosing among branches,
        one per relation, under 353's equal-counts rule - is worth the cardinality risk.
  FAIL  all of them at or below cos. Then the relation set is a real space (371) that the STAGE
        cannot exploit, the hand-written compass was already the right one, and the honest end
        is 0.343 end-to-end.
  READ  hit_rate first, hit/reach second. And watch `reachable`: a compass that reaches more
        while hitting less is 347's law again, which has now fired three times.

## 10-RESULT-3. 372a FAILED, AND KOSTYA CAUGHT THE DRIFT: I BUILT THE WRONG 372

    pooled     hit     reach   hit/reach   cand_places
    cos       0.566    0.639     0.886        4.35
    jaccard   0.500    0.555     0.901        1.72
    cover     0.480    0.531     0.904        1.78
    share1    0.445    0.521     0.854        1.50
    rare      0.413    0.505     0.817        1.55

FAIL, nothing beats cos, and it is NOT 347's law: reach fell TOGETHER with hit, and
cand_places collapsed 4.35 -> ~1.6. The sharing graph is simply THINNER than the fingerprint -
a worse route, not a different-but-equal one. Phi still picks well on whatever arrives
(hit/reach 0.90 on two of them). THE HAND-WRITTEN COMPASS WAS THE RIGHT WALK.

AND THE CORRECTION IS KOSTYA'S, NOT MINE: 371 measured RELATION AS EVIDENCE ON A CANDIDATE. I
built RELATION INSTEAD OF WALKING. 371 never said to throw substitution away - it said that on
the questions recall cannot answer, choosing among relations beats random places.

## 10-NEXT-3. 372b: THE THING 371 ACTUALLY MEASURED. `--reach-import relation`

The walk stays `cos`. The offer does not change. What changes is WHICH MENTION a candidate
brings as evidence: not the first rows the walk passed, and not its homes anywhere on the tape,
but ITS MENTIONS AT PLACES RELATED TO THIS ONE, best overlap first. Same candidate list, same
budget, one head. The relation becomes a property of the EVIDENCE rather than a second
decision - which is the only shape that dodges the 4x law (321/341/352) and 367's lane problem
at once.

A candidate the relation cannot witness falls back to the walk's own rows rather than to
nothing: an empty world would make "unrelated" read as "unsupported", a different claim, and a
size marker besides (291's undoing).

Static check 7/7; _check301, _check365 and _check372 all still pass.

    <standing arm> --connect --reach-import relation --seed <s> --run-tag 372rel_s<s>
    CONTROL = 365conn (import=walk): hit 0.566, reach 0.639, hit/reach 0.886.

  PASS  hit/reach > 0.886 with reach unchanged. Reach CANNOT move - the offer is identical by
        construction - so this is a clean test of the picker alone, the first one we have had.
  FAIL  hit/reach flat -> which mention a candidate brings does not change what Phi can tell
        apart, the 0.88 is the substrate's for a third time, and the relation set is closed as
        a degree of freedom the stage cannot use.
  IF REACH MOVES AT ALL, something is wired wrong - read that before anything else.

## 10-RESULT-4. 372b FAILED, AND MY "REACH CANNOT MOVE" CLAIM WAS WRONG

                    365conn    372rel
    hit               0.5668    0.5706    +0.004, noise
    reach             0.6385    0.6493    IT MOVED: -0.007 / +0.008 / +0.043 / 0.000
    HIT/REACH         0.8877    0.8788    FELL. Gate needed > 0.886
    per-seed h/r    .880 .951 .930 .783   ->  .925 .899 .885 .809   two up, two down
    CONFIRM        2/229 -14.94           7/116  z -9.83
    PICK pooled    1183/0 +34.39          1081/1 z +32.83
    GATE-WO        1.0000 everywhere      1.0000 / 1.0000 / 0.9915 / 0.9900

FIRST, MY ERROR. I pre-registered "reach CANNOT move - the offer is identical by construction -
so this is a clean test of the picker alone; if reach moves at all, something is wired wrong."
IT MOVED, AND NOTHING IS WIRED WRONG. The arm runs at DEPTH 2, and the second hop's root is the
place that offered THE MIND'S OWN PICK. Changing the evidence changes the budget, changes the
pick, changes the root, changes what the deep walk reaches. 352 knew this exactly - it is why
`--deep-root first` exists as that result's artefact check - and I forgot it one step later.
s8642 is bit-identical (0.6280) because that seed's picks happened not to move the root, which
is the shape a genuine wiring bug would NOT have.

SECOND, THE RESULT. FAIL. hit +0.004 is noise and hit/reach FELL 0.888 -> 0.879. WHICH MENTION
A CANDIDATE BRINGS DOES NOT CHANGE WHAT PHI CAN TELL APART.

## 11. THE RELATION BRANCH IS CLOSED, AND WITH IT THE LAST STRUCTURAL QUESTION

    371   the space of relations is REAL - on the 59% of questions recall cannot answer, a
          perfect chooser over the family reaches 0.1725 against 0.0901 for the same number of
          random place-sets. That result stands.
    372a  relation INSTEAD OF the walk: every member loses to cos, and not to 347's law -
          reach fell WITH hit and cand_places collapsed 4.35 -> 1.6. The sharing graph is
          thinner than the fingerprint.
    372b  relation AS EVIDENCE on the candidate, which is what 371 actually measured: flat.

SO THE SPACE IS REAL AND THE STAGE CANNOT REACH INTO IT. A perfect chooser over relations would
gain; neither of the two shapes that keep one head and one price can deliver that gain, and the
shapes that could (a second head, more lanes) are already closed by the 4x law and by 367.

THE 0.88 IS THE TAPE'S, NOW ON THE THIRD INDEPENDENT TEST: five policy rearrangements, a 3.8x
capacity test, and both forms of the relation set. About 12% of the truths that reach the offer
cannot be told from their distractors by any reading of this tape.

FINAL: coverage 0.5937 x hit 0.5706 = 0.339 of every token position in wikitext, from counts
plus 5633 parameters that hold no facts. The standing arm is
`--reach-depth 2 --two-way --min-fillers 1 --connect`.

## 12-NEXT. 373: THE ATOM, THE ONE PART OF THE CONSTRUCTION NEVER VARIED

A value has been a WHOLE WHITESPACE TOKEN since the first frame commit. Every closure - the
decision (five rearrangements), capacity (two tests), the relation set (371, 372a, 372b) - was
measured with the answer drawn from a finite BAG of whole tokens. While the answer is CHOSEN
FROM A BAG, "produce what was never offered" is not hard, it is ARITHMETICALLY IMPOSSIBLE.

`_audit373_pieces.py`, torch-free, on exactly 363's subset - truth not among the place's own
fillers and not in the walk's offer, where every number this project prints is zero:

    material   what is actually in front of the mind there: own fillers plus the top-m offer
    COVER      is every k-gram of the truth present among the material's k-grams
    ASSEMBLE   the strict, interpretable one: can the truth be SPELLED by concatenating whole
               available values - "wikitext" out of "wiki" and "text". Bounded at three parts,
               and the bound is reported: a four-part spelling is a coincidence, not a compound
    NULL       both, for a word from the TRUTH'S OWN FREQUENCY BAND. Pieces of common words are
               everywhere and an unmatched null would hand COVER a win for free - 363 learned
               that when a uniform pool gave raw frequency an AUC of 0.70
    job        how many available values an assembler would weigh, as 351 reported `paths`

  GATE  cover - cover_null > 0.05 AND assemble - assemble_null > 0.05.
  COVERED BUT NOT SPELLABLE -> the pieces are present the way the letters of any word are
        present: real material, no compositional route from it to the answer.
  NEITHER -> the atom was never the obstacle, and making values smaller does not make
        production possible. That closes the last untouched piece of the design.
  `--ngram` is SWEPT (3, 4, 5), not tuned - a conclusion that holds at one size only is a
  conclusion about that size.

  Verified before shipping: assembly is exact and respects its bound (wikitext yes,
  unbelievablewiki no at three parts, wikitex no), and the k-grams mark word boundaries so a
  prefix and an infix are different material (`^ca` is in `cat` and not in `scat`).

    python _audit373_pieces.py
    python _audit373_pieces.py --ngram 4
    python _audit373_pieces.py --ngram 5 --window-lines 1600

## 12-RESULT. 373: THE ATOM WAS NOT THE OBSTACLE. Held on all three sizes

                        COVER vs null        ASSEMBLE vs null
    k=3, window  400   0.0040 / 0.0020        0.0080 / 0.0000
    k=4, window  400   0.0007 / 0.0000        0.0080 / 0.0000
    k=5, window 1600   0.0000 / 0.0000        0.0107 / 0.0007
    both gates wanted +0.05. Material at each hole: 22-39 values.

TWO THINGS TO SAY PRECISELY, BECAUSE "ZERO" WOULD BE THE WRONG WORD.

1. COVER IS NEAR-DEGENERATE AND SHOULD NOT BE READ. "Every k-gram of the truth present" almost
   never happens, for the truth OR the null, at any size. A comparison between two numbers that
   are both ~0.001 measures nothing - the VOID rule, one last time.
2. ASSEMBLE IS REAL AND TINY. The truth is spellable out of what is in front of the mind on
   0.8-1.1% of these holes against a frequency-matched null of 0.0-0.07%. The RATIO is large
   and the null is genuinely near zero, so composition is NOT an artefact - there is just very
   little of it. 1% of the 59% that nothing reaches is 0.6% of all questions, below any
   threshold that could carry a design.
   And SHARE (the fraction of pieces present) is +0.068 at k=3 and MELTS as k grows: letters
   overlap between any two English words, compounds do not.

SO THE FOURTH TRAIT IS CLOSED, AND FOR THE THIRD REASON IN A ROW - NOT THE ONE I EXPECTED EACH
TIME. Not "no architecture" (352-368), not "no measure" (363 found one at 2x chance), but NO
MATERIAL: the answers this tape cannot reach are not built out of what it can.

ONE CONDITION UNDER WHICH THIS WOULD READ DIFFERENTLY, and it is specific rather than a
consolation: English wikitext is morphologically thin. The same audit on German, Finnish,
Turkish, or on CODE IDENTIFIERS (getUserName, max_retry_count) - where a large share of unseen
tokens ARE compounds of seen ones - could put ASSEMBLE an order of magnitude higher. `--corpus`
already takes any file, so it is one command, not a project.

## 13. THE END STATE, AND IT IS AN ANSWER

WHAT IS PROVEN. Knowledge and decision separate. Four independent controls: transplant to two
foreign corpora, fine-tuning buys nothing, a native mind is indistinguishable from the
transplant (73/71 of 8000, z +0.17), the shuffle-tape null erases everything. The decider is
5633 parameters and holds no facts.

WHAT THE DECIDER IS GOOD AT, corrected from what we claimed at step 0: it RANKS (PICK 1081/1
pooled, never once beaten by a count) and it KNOWS WHEN IT IS RIGHT (GATE-WO 1.0000 at every
fraction). Routing was the original claim; five attempts to improve it failed and 367 showed it
can be removed while the ranking survives.

THE NUMBERS THAT BOUND IT, each measured and each independently confirmed:
    coverage    0.59   the share of token positions that are askable at all
    hit/reach   0.88   the share of reached truths that can be told from their distractors -
                       survived five policy rearrangements, a 3.8x capacity test, and both
                       forms of the relation set
    product     0.339  of every token position in wikitext, from counts plus 5633 parameters

STEP 0'S FOUR TRAITS: compose - closed (310). Remember - closed (354/355, oracle included).
Revise, as the stay/go decision - closed three ways (362, 366, 372). Produce - closed for want
of MATERIAL (373), having first been closed for want of a measure (363).

WHAT WOULD MOVE, IF ANYTHING: the separation margin GROWS with scale (369: z +57.59 on 5x the
questions, gate 0.97-0.996). Accuracy does not. Any future work should be on a corpus where an
answer is CHECKABLE BY ANOTHER HOLE - code, tables, records - because that is the only setting
where the fourth trait has both a measure and material.

## 12-RESULT-2. THE MORPHOLOGY CONDITION WAS TESTED AND IT DOES NOT HOLD EITHER

At matched material (job = available values per hole):

    corpus              job     ASSEMBLE vs null    SHARE
    en wiki   w400     21.6      0.0080 / 0.0000    +0.036
    de news   w8000    21.6      0.0000 / 0.0000    +0.046
    fi wiki   w8000    10.2      0.0000 / 0.0000    +0.049
    code idents w8000  11.2      0.0000 / 0.0000    +0.164
    code src  w8000    12.3      0.0107 / 0.0027    +0.061

I NAMED THIS CONDITION MYSELF AND IT IS REFUTED. German, Finnish and code identifiers -
the three cases where unseen tokens most obviously ARE compounds of seen ones - assemble at
EXACTLY ZERO. Only English wiki and code source show anything, and code source's 0.0107 sits
against a null of 0.0027, the weakest margin in the table.

AND THE ONE ROW THAT LOOKS LIKE HOPE IS THE CLEAREST REFUTATION. `code idents` has SHARE +0.164
- by far the highest piece overlap anywhere, exactly as morphology predicts, camelCase and
snake_case sharing `get`, `set`, `name`, `count` everywhere - AND ASSEMBLE 0.0000. THE PIECES
ARE SHARED AND THE ANSWER STILL CANNOT BE SPELLED FROM WHAT IS REACHABLE. That separates the
two claims completely: morphological overlap is real and is NOT compositional reach.

SHARE rises with morphology (.036 en -> .046 de -> .049 fi -> .164 idents) and ASSEMBLE does
not follow it at all. Shared material is not available material: the pieces of the answer sit
in OTHER words of the corpus, not in the ~20 values standing at the hole.

SO THE CLOSURE IS NOT ABOUT ENGLISH. Producing what was never offered fails for want of
material on five corpora across three language families and two kinds of code, with the
condition I predicted would rescue it tested head-on and refuted by its own best case.

## 8-WAS-4. THE SKETCH BEFORE IT WAS BUILT

The concrete step is small: neighbourhood values, overlap-weighted, appended to the candidate
list the walk already produces, at the same REACH_CANDS budget. Every rival and null rides
along as always. Expected +4-5 points of reach, and the honest risk is that a bigger offer
makes PICK harder and gives it all back - which is exactly what 347 measured for thickness.

    if u_reach is materially above zero, the tape has a THIRD channel and 359's whole apparatus
  applies to it: rivals, a null, and the question of whether choosing between three channels is
  worth a head (interleave beat choosing for two - it may well beat it for three).

## 8-WAS-3. THE BUILD SKETCHED BEFORE THE NO-PICK RIVAL RAN (kept: it is what was refuted)

recall becomes a SECOND BRANCH in the SAME stage-one argmax,
compared to substitution by 353's rule - each branch worth its best world, over equal counts.
Not a second objective, so the 4x law does not apply; it is exactly the two-way mechanism that
already worked, applied to channels instead of to stay-vs-go. And Phi's known scale-freeness
(329/337/340: only the gap carries) is precisely what an incommensurable pair of channels
needs.

THEN, IN ORDER:
  - 358 RE-RUN WITH RECALL ADMITTED. What a document writes about itself is CONSTANT, so
    ingestion and recall are one story and that is why 358 read flat.
  - every closure in section 5 was measured on the substitution channel alone. They are not
    void, but they are now CONDITIONAL: "on the substitution channel". Memory (354/355),
    resolution (357) and the second relation (349/350) are the ones worth re-asking.

## 8-WAS-2. THE LOOP (351). 353 delivered the chain ACROSS HOPS; this is ACROSS QUESTIONS

Depth 2 is a two-hop chain inside ONE question. Nothing yet touches state BETWEEN questions.
First lever, torch-free and measured before anything is built: `_audit354_session.py`.

    a SITUATION = `--session` questions drawn from one run of `--session-lines` lines, answered
    in order. Each answer joins the lens set W for the questions after it.
      reach_own   the offer from the question's own rows           (today)
      reach_W     the offer with the session's previous ANSWERS added as lenses
      reach_R     the same with WRONG answers of the same shape    THE NULL, load-bearing
    Matched budget: the same top-m at the end, so a session cannot win by being offered more.

  GATE  gain_over_null = reach_W - reach_R > 0.05. Read against the NULL, never against
        reach_own: adding any lens widens the offer, and a gain a wrong answer also buys is
        offer size, not memory.
  CONTROL BUILT IN: `--session-lines 0` draws from the whole tape, which IS 324's
  independent-question setting. It should reproduce ~0, and if it does not, the audit is wrong
  before the result is interesting.

    python _audit354_session.py --session 6 --session-lines 40
    python _audit354_session.py --session 6 --session-lines 0     # 324's setting, expect ~0

## 8-WAS. THE STANDING DIAGNOSIS

The pipeline is assembled FOR ONE SHOT: question k+1 knows nothing of question k; the answer is
scored and discarded; the reward is terminal; WE pick the hole at random. Composition,
generation and revision are all things a mind does ACROSS steps, and all three were closed as
single moves. (Kostya's metaphor: the parts of a car laid on the floor, pressing an accelerator
that lies next to the engine.)

Two closures DO NOT TRANSFER and are re-openable:
- 324 closed MEMORY on a perfect write-back's marginal RETRIEVAL gain over INDEPENDENT
  questions. It never measured a dependent chain.
- 322 closed DEPTH, and 352 has now overturned it.

AND AN ORDERING ERROR IN THE LADDER (344): generation was put first and revision third. Both are
chain properties, and generation may be IMPOSSIBLE single-shot while NATURAL in a chain.

## 9. FILES

    _stage289_derivation.py   the single stage (~7500 lines). Flags added recently:
                              --constrain --cons-resolve {count,share} --cons-lenses
                              --retain --retain-by --rival-mind --speak-batch --speak-weight
                              --two-way --two-way-by {max,margin} --reach-depth --deep-root
    _check301_wiring.py       row/column width assertions (reach 30, pair 17, cons 15)
    _check309_pair.py         the pair verb's leak checks
    _check337_rank.py         one argmax; retention honoured everywhere; the guards exist;
                              rank_auc/prec_at/gate_top RUN against hand-computed answers;
                              two-way equal counts and the deep-attach ORDER
    _check345_cons.py         the constraint's leak check, RUN on a hand-made tape
    _read299.py               the main reader: void, claim, CONFIRM, ROUTER, PICK, DEPTH,
                              MARGIN, RANK, GATE, GATE-WO, OTHER, pooled
    _read309.py _read338_retention.py _read342_capacity.py _read345_cons.py
    _audit323..._audit351_chain.py   the torch-free ceilings, each measured BEFORE building
    _sweep335 _sweep342 _sweep347 _queue348 _queue352

## 10. WHAT THE PROJECT HAS EARNED, STATED PLAINLY

KNOWLEDGE AND DECISION CAN BE SEPARATED. Demonstrated four independent ways, with a decider of
5633 parameters that routes, picks, calibrates and gates better than every counting rival and
holds no facts. The limits measured are the TAPE's - substitution only, one relation - and
saying which limit belongs to which half is most of what this project set out to do.

What is NOT yet earned: a mind that produces what was never offered, combines two facts into a
third, abandons a wrong path, or chooses its own question. 352 is the first evidence that the
route to those is DEPTH INSIDE ONE OBJECTIVE, not a new relation and not a second term.

## 14. THE COPY CHANNEL, 374-378: WHAT WAS MEASURED AND WHAT IT COST

**374 (form).** 373 refuted CONCATENATION of whole available values, not inflection - its own
cross-corpus table said so, SHARE rising strictly with morphology (.036 en, .046 de, .049 fi,
.164 code idents) while ASSEMBLE stayed 0.0000 everywhere, because an agglutinative ending never
stands at a hole as a whole value. 374 measured the operation a language actually performs -
available stem plus frame-predicted ending. On german w8000:

    pred_local .746   pred_func .521   pred_global .269      shape_gain +0.251
    produce_oracle .020   oracle_null .009   G1 +0.011       produce_func .007   G2 +0.005

THE FORM SIGNAL IS REAL AND PRODUCTION HAS NO MATERIAL. The function words of the address DO
carry the ending, +0.25 over the corpus null - the largest clean signal in many steps. And even
knowing the WHOLE ending inventory, an unreachable truth is an available stem plus an ending
0.020 of the time against 0.009 for its frequency twin. The stems are not at the hole either.

**375 (the address atom).** So the signal's place is the ADDRESS, not the value. Addresses are
exact-token tuples - singleton_bucket 1.0, every frame unique - which is why places are thin.
kappa(t) = t for the F most frequent tokens, else its counted ending class.

    en w400 f32    unreach .497 -> .547   (-10%, WORSE)   hit -0.021   cov +0.57
    en w1600 f64   unreach .475 -> .522   (-10%)          hit +0.019   cov +0.50
    de w8000 f32   unreach .684 -> .672   (+1.9%)         hit -0.009   cov +0.77

COVERAGE ROSE BY HALF AND THE UNREACHABLE SHARE DID NOT MOVE. By the audit's own verdict the
wall is not the alphabet: the missing truths barely REPEAT in the window, and a count cannot
reach what was counted once.

**376 (copy, the audit).** "Counted once at a place" is not "absent from the page". On exactly
373's population - truth in neither own nor offer:

    en w400   D=4    copy .538  null .186  +0.352
    en w1600  D=16   copy .679  null .293  +0.386
    de w8000  D=4    copy .120  null .051  +0.069

THE TRUTH IS STANDING NEXT DOOR. The german row is the channel's own limit written down: news
does not repeat its subject the way an encyclopedia article does. This pays on DOCUMENTS.

**377 (--copy, interleaved).** Paired against 365conn, same four seeds, same tape:

    reach   .5771->.6356  .6404->.7139  .7085->.7607  .6280->.6848    +0.060  4/4
    hit     .5078->.5556  .6089->.6693  .6588->.6209  .4917->.5059    +0.021  3/4
    hit/reach                                                          -0.044  4/4
    cand_places 4.17->4.08  4.28->4.10  4.46->4.23  4.47->4.10        down    4/4

Gate as declared FAILED (hit on reachable fell 0.044 against a 0.02 bar). Recorded as a failure.
THE GATE WAS ALSO BADLY CHOSEN AND THAT IS MY ERROR: hit/reach is composition-dependent, and a
lever whose purpose is moving questions from unreachable to reachable must drop it unless the
new questions are as easy as the old. Same mistake as 369's addresses gate. What did NOT move,
so neither is the lever's doing: CONFIRM z -14.94 -> -15.23, and GATE-WO pooled 1.0000 at
5/10/25% while the random baseline fell 0.93 -> 0.73 - calibration held on a HARDER population.

**378 (--copy-backfill).** Hypothesis: the eviction is the cost, so let copy take only the slots
the walk left empty. Gates declared: reach not below 377 on 4/4, hit at or above 365conn on 4/4,
cand_places not below 365conn.

    reach        +0.006 (and BELOW 377 on 4/4 by .04-.07)      FAIL
    hit          -0.069 (below 365conn on 3/4)                 FAIL
    cand_places  +0.34 on 4/4 - the eviction really is gone    PASS

THE HYPOTHESIS IS REFUTED AND THE REFUTATION IS THE RESULT. Remove the eviction and the supply
collapses from +0.060 to +0.006. The copy candidates are therefore WORTH MORE than the walked
ones they displaced, and free slots barely exist - the walk normally fills the cap. Eviction was
not the cost of the channel, it WAS the channel.

s4711 collapsed outright: step .8400 -> .1505, ROUTER stepped 709 -> 127, arrive .9820 -> .1848,
AUC on `right` 0.4969 (chance), and hit_of_own 0.8643 == own_rival_of_own 0.8643 to the digit -
the mind became the counting rival because it stopped stepping. The mechanism follows from
backfill itself: copy candidates arrive ONLY where the walk under-filled, i.e. on the thinnest
questions, so the channel is fed exclusively into the hopeless ones and teaches the router not
to step. Interleave spreads them everywhere. Backfill is worse in supply AND in what it teaches.

**STANDING ARM AFTER 378:** `--min-fillers 1 --reach-depth 2 --two-way --connect --copy
--copy-d 4`. Neither 377 nor 378 passed its gate; 377 is strictly better than both 378 and
365conn on reach (4/4) and hit (3/4). `--copy-backfill` is retired.

**WHAT 377-vs-378 LEAVES OPEN, and it is a decision, not material.** The cap is the binding
resource and the merge rule is a CONSTANT, not a decision: round-robin and backfill are both
fixed allocations. Phi never learns which allocation this hole wants, because IT CANNOT SEE
WHICH CHANNEL A CANDIDATE CAME FROM - `from_place` records -1 connect, -2 home, -3 copy, and
feeds only `n_places`. That is 379.

## 15. 379-380: THE FEATURE WAS FINE, THE SECOND READ WAS NOT

**379 (--reach-channel) ran and its gate does not close - but the gate was never tested.**
Kostya found the reason in the code, not in the numbers, and it is a bug four steps old.

    seed   hit d    reach d   hit_of_deep
    1337   +0.047   +0.011    .954 -> .947      root still a walk place - RANKING WORKED
    4711   +0.027   -0.033    .980 -> .830
    8642   -0.013   -0.040    .949 -> .793
    2890   -0.049   -0.051    .919 -> .796      PICK 1238/1480 vs 377's 1258/1572

THE INDICATORS ARE BUILT CORRECTLY - off is off, only the answered row carries them, the hidden
value is not an input, cand_places matches 377 to fractions and reachable_wide / reachable_random
are bit-identical, so neither the tape nor the offer moved. What moved is the SECOND READ.

`reach_deep` rebuilt its own value->place map by scanning only the WALKED places, so a candidate
offered by any other channel was absent from it and the root fell back to `places[0][0]` - the
walk's first place, unrelated to where that value stands. `reachable_rate` counts the deeper
candidates (`ansble = truth in set(cands) | set(_dc_all)`), so the offer can be identical and
reach still moves. Not a leak, not a different tape: a broken hop-2.

**THE BUG IS FROM 365, NOT FROM 379.** Connect candidates have hit that fallback since
`--connect` existed, copy candidates since 377. 379 only made Phi pick non-walk candidates more
often, so it fired more often. Therefore 365conn, 377copy, 378bf and 379 are ALL partly
confounded, in proportion to how often the mind picked a non-walk candidate - and 378's s4711
collapse now has a better explanation than the one recorded in section 14: backfill feeds copy
candidates ONLY onto thin questions, the mind picks them, the second read roots at an unrelated
walk place, depth returns garbage, and the router learns to stay. The mechanism named there was
right; the reason was wrong.

**380 fixes it with one rule for every channel.** A walked candidate keeps its walked place,
which `from_place` already records. Any other candidate is rooted through its OWN FIRST ROW: a
row is a slot, a slot has an address, an address is a place. For connect that recovers exactly
the neighbour place - its rows come from that place's fills, best overlap first - so no
signature changes. Rows at the question's own address are skipped, the same exclusion the walk
and connect already apply. `rc["rows_of"]` is read and not `ev`, so the root cannot move with
--reach-import. `_check380_deeproot.py` covers six properties torch-free; both the fallback and
the `ev` dependency were verified by reintroducing them.

**Also corrected:** 379's docstring said the stage-one flag is constant "because every world is
a home world". The true reason is narrower - connect and copy drop own values by construction,
so a home value cannot carry a negative from_place. Under --own-in-offer it can (-2) and stage
one is no longer constant. Narrowed in place.

**379 MUST BE RERUN after 380.** Its numbers are an honest record of what that arm does with the
bug in it, and they settle nothing about provenance: the lever's effect and the bug's damage are
entangled in one number, and the one seed whose root stayed a walk place is the one where hit
rose. The 379 queue as it stands is not to be re-read as a verdict.

## 16. 380 RERUN: THE FIX REVERSES THE COPY VERDICT, AND A SECOND BUG

All three arms rerun on 380, same four seeds, `reachable_wide` identical per seed - same tape.

    arm                    hit     reach   hit/reach  PICK pooled
    365r  connect          .5987   .6525   .918       1132/1  (.799)
    379r  copy + channel   .5394   .6362   .848        983/1  (.718)
    377r  copy             .4750   .5726   .830        673/1  (.583)

AGAINST THE SAME ARMS BEFORE THE FIX: 365conn hit .5668 -> .5987, and 377copy hit .588 -> .475.
The fix RAISED connect and SANK copy, and the reason is legible. The broken root sent the second
read to `places[0][0]`, the most similar walked place - a good neighbourhood. The correct root
for a copy candidate is wherever that token happens to live on the tape, and a copy candidate is
related to the question only by SITTING NEAR IT ON THE PAGE, so its tape home is unrelated. The
bug was accidentally the better policy for copy and the worse one for connect, whose neighbours
are related by a shared filler and whose true root therefore helps.

**--copy IS RETIRED FROM THE STANDING ARM.** 379 does work - it lifts copy on every axis
(hit .475->.539, reach .573->.636, PICK 673->983, GATE-WO at 50% .853->.968, and s4711 out of
collapse .3235->.5782) - but it is damage control for a channel that costs more than it brings.
Connect-only beats it on hit on 3 seeds of 4. 347's law, measured a fourth time.

**STANDING ARM: `--min-fillers 1 --reach-depth 2 --two-way --connect`.**

**WHAT NEVER MOVED, ACROSS FOUR ARMS AND A FOUR-STEP-OLD BUG.** The PICK rival scored 1, 0, 1, 1
against thousands of questions, z +25.9 to +34.4, and GATE-WO held 1.0000 at 5/10/25% in every
arm that did not collapse. The bug was an unplanned natural experiment: it moved reach and hit
by .05-.12 and broke whole seeds, and it did not move the mind's ranking advantage off zero.
That invariance is stronger evidence for the separation than any single arm's numbers.

**SECTION 13 IS WITHDRAWN.** "The end state" was a claim about my own imagination, not about the
system. One question about word endings produced 374-380 and uncovered a bug four steps old.

**381, THE SECOND BUG, found while re-reading rather than by a number.** `reach_candidates`
computed `n_places` as `len({from_place[c] for c in cands})`, and from_place stores -1/-2/-3 as
CHANNEL MARKERS for non-walk candidates. So every connect candidate collapsed into one
pseudo-place however many neighbours they came from, and every copy candidate into another:
with the channels on, `cand_places` was not measuring what its name says. It also undermines an
argument recorded in section 14 - "copy evicts walked candidates from distinct places", drawn
from cand_places falling 4.17->4.08 on 4/4 - which was partly reading that collapse. Fixed by
resolving each candidate's real place through its rows (`place_of_rows`, now shared with
`deep_root_of`); a candidate with no row outside its own place has no place and is not counted.

## 17. CODE AUDIT AFTER 381: ONE MORE SILENT BUG, AND TWO READING HAZARDS

Run as a read of the code rather than of the numbers, because both bugs found so far were found
that way and both had been live for four steps.

**382, AND IT TOUCHES THE HEADLINE.** `reach_count_rival` iterated PLACES and filtered by
membership:

    for j, _it, _sim in places:            # walked, plus deep since the COMP_ONLY fix
        for v, _rows, c in ix["fills"][j]:
            if v not in seen: continue

A value contributed by connect or copy stands at NO WALKED PLACE - if it did, the walk lane
would have offered it first and the interleave would have deduped it away. So the rival could
never score exactly the candidates the channels add. On a question whose truth arrived by
connect, the rival was not losing, IT WAS MUTE BY CONSTRUCTION, and the mind's margin there was
a definition. This is the COMP_ONLY trap that function's own docstring already names twice as
having been walked into; it was closed for depth and left open for the channels.

`--connect` has been in the standing arm since 365, so PICK is inflated for that whole era -
365conn, 365r, 377, 378, 379 alike. The fix scores every unscored candidate at the place 381
resolves for it, by the same exact share, with the walk's order still breaking ties and the
walked pass still running first. THE PICK NUMBERS OF THE CONNECT ERA HAVE TO BE RE-MEASURED,
including 365r's 1132/1.

**VERIFIED CLEAN, so it is not re-examined next time.** The `own` exclusion used by connect and
copy is `q["slots"][:q["query_row"]]`, and the hidden slot is appended LAST when the question is
built (`"slots": rows + [hidden_slot]`, `"query_row": len(rows)`), so that slice is exactly the
evidence rows and nothing at the question's place escapes the ban.

**READING HAZARD 1: `reachable_wide` and `reachable_random` are WALK-ONLY.** `reachable_rate`
counts channels and depth; the two controls do not. That is what makes them a clean control of
the TAPE - they were identical per seed across 365r/377r/379r, which is how we knew the tape had
not moved - but the gap between `reachable` and `reachable_wide` can no longer be read as "our K
was too small" once a channel is on, because the channels are inside one number and outside the
other.

**READING HAZARD 2: the import budget is a MINIMUM over candidates**, so adding a channel
changes world sizes as well as the offer. `expand` moved 12.21 -> 13.02 between 365conn and
377copy on s1337. The project has twice been undone by a row-count marker (291, 296), so hit
compared across arms must be read with `expand` beside it. Visible in the report, not hidden -
but it means two arms differing in a channel differ in more than the channel.

## 18. 383: THE COUNTING RIVAL'S RULE WAS SATURATED, AND I HAD BEEN OVERSELLING IT

382 landed and changed nothing: `PICK vs COUNT` was bit-identical across 365r and 365r2 on all
four seeds (count-rival 0/1/0/1, paired 287/262/306/276, z +16.94/+16.19/+17.49/+16.61), and
`count_rival_hit_rate` over ALL questions was unchanged too. `cand_places` rose 4/4 (4.17->4.61,
4.28->4.69, 4.46->5.02, 4.47->5.07), so 381 resolves places and the channel pass is alive.

THE REASON WAS ALREADY PRINTED IN EVERY DUMP AND I HAD NEVER READ IT. `top_share` is exactly the
share `reach_count_rival` returns, and it reads 0.999 to 1.000 on every seed of every arm. The
share rule SATURATES: its argmax is essentially always a value that OWNS its place, and with
--min-fillers 1 that means a single-filler frame, where the share is 1.0 whether the value
stands there nine times or twice. No candidate can beat 1.0, and ties went to the walk's order -
so adding candidates could not change the answer in principle. 382 is correct and was inert.

**A CORRECTION I OWE, not a code bug.** I have called `reach_count_rival` "the strongest
counting rival the walk allows" many times. In fact it reduced to "name the filler of the first
walked place that has only one filler", and its near-zero score was substantially a property of
THE RULE rather than a limit of counting. This weakens one line of evidence - "the mind beats
the best count" - and does not touch the separation, which rests on four independent controls
(transplant, fine-tune, native-vs-transplant, shuffle null).

**383 breaks the tie by the RAW COUNT.** Lexicographic (share, count): a value standing nine
times at a place it owns is stronger evidence than one standing twice. Still exact, still one
rule, nothing fitted, and strictly stronger than what it replaces - which is the standard that
function's own docstring sets. The walk's order remains the last tie-break, so nothing became
arbitrary. `cr_ties` is a new column at the END of REACH_COLS (indices unmoved, _check301_wiring
passes) reporting how many candidates shared the winning share, printed by _read299 next to
`top_share`: 1.00 means the old rule was determinate but blunt, above 1.00 means it was
arbitrary and the walk's order was deciding. That distinction cannot be inferred from the share
and had to be counted.

Gate, declared: count-rival rises on at least one seed. If it does not move at ties 1.00, the
rule was determinate all along and the next rival must be a different rule, not a better
tie-break - and I will propose one only after seeing the number.

## 19. 383 READ, AND 384: THE CONSTRAINT'S RESOLUTION HAS NEVER BEEN A SELECTION

**383 answered, and against my own prediction.** I expected `ties` near 1.00 - the rule
determinate but blunt. It reads **6.14, 6.41, 6.60, 6.14**: of eight candidates, SIX shared the
winning share. With --min-fillers 1 most places hold one filler, so almost every candidate owns
its hole and the share carries no information at all; the walk's order was deciding. The old
rival was ARBITRARY, not blunt. With the count tie-break it comes alive - count-rival
0/1/0/1 -> 0/2/1/2, two hits becoming five - and is still 5 against the mind's 1132 of 1417,
now while seeing channel candidates (382) and choosing by evidence rather than by order (383).
That is the honest version of this comparison and it is closed.

WHAT THAT MEANS FOR THE HEADLINE, stated plainly: "the mind beats the best count" has been
measuring, in large part, that COUNTING HAS NO RULE HERE. The separation is untouched - it rests
on four independent controls - but this particular line of evidence is weaker than I have been
writing it.

**THE DIAGNOSIS KOSTYA PUT, and the numbers agree.** What exists is a very good chooser plus a
very good calibrator over a short offered list. The mind's ONLY search action in the whole
system is the root of hop 2 - one root, one step. Where to look is otherwise fixed policy, and
four attempts to widen the list have each cost more than they brought (347, 365, 369, 377).
ROUTER 1.13-1.27x against counting's 0.78-1.09x is the weakest number we print, and it is
exactly the searching one.

**384. The constraint interface exists and is CLOSED TWICE** - 345 L1 by raw count, 345 L2 by
share, plus 346 (two lenses intersected). The verdict recorded for the family is "the third time
SUMMING LENSES has lost", concluding that what could work is "a SELECTION of one lens, never an
accumulation". But the accumulation was never in the mind's half: the mind already selects one
of its own rows. It was in the TAPE's half - `cons_cooc` adds the fills of EVERY place holding
the lens, and share only divides that sum afterwards. A constraint whose RESOLUTION is also a
selection has never been run.

    CONS_RESOLVE = "place"    the tape answers from ONE place - where the lens stands most
                              often, ties to where it owns the largest share of its hole, then
                              tape order. The question's own place is excluded OUTRIGHT rather
                              than subtracted, because when one place is being chosen there is
                              nothing left to subtract from.

The ingredient did not exist in 345: 380 had to answer "where does this value actually live" to
root the second read, and this asks the same question of a lens. `cons_rows_for` follows the
resolution, so the evidence comes from the place the answer came from - otherwise the arm would
differ from its baseline in two things at once.

GATES ARE 345's, UNCHANGED AND UNSOFTENED:
  (a) choosing the lens is a decision, not a count: against `rare`, `frequent`, `decisive`,
      paired, on questions where the truth is NOT among the question's own rows. z >= +1.645
      pooled AND ahead on >= 3 of 4 seeds.
  (b) it reaches more than the enumeration it replaces: `answerable` against `walk_answerable`
      on the same question, ahead on >= 3 of 4 seeds. One read against the walk's eight.

THE RISK, NAMED FIRST: this is the neighbouring form of a thing that has lost three times. If it
does not take those gates, the seeker does not build on this tape, and the honest description of
what we have is Kostya's - a strong classifier over a short list.

`_check345_cons.py` gains the place rule: it must not return the question's own place, must not
reproduce the sum's answer, must not invent a place for a lens standing nowhere else. Both
failures were verified by reintroducing them.

## 20. 384 FAILED EMPTY, AND THE FAILURE IS SEMANTIC - PLUS A CORRECTION OF SECTION 19

    seed   answerable   walk    present@topm   beyond_own hit   constrain_rate
    1337     0.0084    0.2007      0.0263          0.0000           0.2963
    2890     0.0000    0.2756      0.0026          0.0000           0.2388
    4711     0.0000    0.2642      0.0036          0.0000           0.2666
    8642     0.0024    0.2429      0.0071          0.0000           0.1967
    POOLED   GATE a  mind 0 / decisive 4 of 2262  z -2.00  ahead 0/4   FAIL
             GATE b  reaches more than the walk on 0/4, matched offer 0/4  FAIL

NOT "BEHIND" - EMPTY. The truth is in the lens's top-m essentially never (0.003-0.026), and the
mind scores zero on beyond_own on every seed. That is worse than 345's sum, which is the
opposite of what selection was supposed to do.

**THE AXIS I NAMED WAS THE WRONG ONE.** 384 was built on "the resolution has never been a
selection", and the data says selection-versus-accumulation was irrelevant. What loses is the
SEMANTICS, and it has now lost four times with three different arithmetics. A place is a HOLE
and its fillers are SUBSTITUTES for one another, so "what stands beside my lens elsewhere"
returns substitutes OF THE LENS - which has almost nothing to do with what fills the question's
hole. Summing many places at least produced a large noisy set; choosing one place removed the
only thing keeping the channel non-empty, which is exactly why it fell to zero instead of
merely behind.

**AND A MISTAKE OF MINE, ONE LEVEL UP.** Kostya's formulation was "look at the neighbouring
lines", "step from here" - constraints about WHERE. `cons_*` is a lens over a VALUE. I mapped
his idea onto the machinery that already existed because it existed. So 345 and 384 have never
tested what he proposed:

    cons_* (345/384)   Phi emits one of its own VALUES; the tape returns what stands beside it
    the proposal       Phi emits one of a few MOVES; the tape EXECUTES it and returns a world

The moves are few and countable: stay; step by fingerprint; step to a place sharing filler X;
read the neighbouring lines. Each builds a world, the same Phi scores worlds, one head, one
objective. The output space is MOVES, not values, so a fact still cannot be encoded in it and
the invariant holds. Nothing like this has been run: the four channels have always been merged
by a FIXED RULE into one offer, with Phi choosing a name. Its only search action today is the
root of hop 2.

The one permitted retry is NOT spent here. At present@topm 0.003 the fault is not the
place-selection rule, and trying another one would be fitting.

**SECTION 19'S LAST LINE IS CORRECTED.** I wrote that if 384 failed, "a strong classifier over a
short list" becomes the final description. That is right about THIS CONSTRUCTION and wrong as a
verdict on the project - the construction is ours and is being rebuilt. Kostya's objection, and
he is right: the seeds are instruments, not the goal, and four channel failures are evidence
about four channels.

## 21. 385: THE MIND EMITS A MOVE AND THE TAPE EXECUTES IT

The first arm in which Phi decides WHERE TO LOOK rather than WHICH NAME TO SAY.

    output space   step | share | lines     three names, so a fact still cannot be encoded
    step           the fingerprint walk, unchanged
    share          365's connect - places sharing a filler
    lines          376's copy - the neighbouring lines
    each offered at the UNCHANGED cap, so a move is a DIFFERENT offer, not a thinner one

THE CHOICE IS MADE BEFORE ANY CANDIDATE WORLD IS SCORED, and that is the whole difference.
Enumerating a lane is cheap; scoring worlds is not. Each move is judged on ONE PROBE WORLD - its
first candidate carried by a single row - and then the chosen move alone is executed at full
cap. Under lookahead the mind would score every lane in full and take the best, which is
today's argmax wearing a different name and decides nothing. That variant is deliberately not
built.

WHY THIS IS NOT ANOTHER CHANNEL. All four channels have always been merged by a FIXED RULE, with
Phi choosing among the names that rule produced; its only search action in the whole system was
the root of hop 2. Four attempts to tune the merge rule each cost more than they brought (347,
365, 369, 377), and 384 closed the value-lens reading of "a constraint the tape resolves". This
changes the OUTPUT SPACE, not the offer's width: one head, one softmax, one objective - 321, 341
and 352 each measured a second objective at about 4x.

THE HONEST LIMIT, stated before the run: 359 measured an ORACLE over channel choice at only
+0.026 over interleaving them. That is evidence against, and it is not the same measurement -
there every channel was executed and merged, here one is executed and the budget goes to it
entirely - but the resemblance is close enough that the gate must separate them.

GATE, declared: against the STANDING ARM (interleave at equal cap), not against a single
channel. hit >= the interleave on 3 of 4 seeds AND the move split is not degenerate - if one
move is taken on nearly every question the arm is a constant with extra steps and fails whatever
hit does. `move_share` is reported for exactly that, and `move_id` is the last reach column.

SAFETY, because this arm has new ways to become its own baseline silently: an offer requested
before the move is chosen RAISES rather than defaulting to the merged one; a lane enumerated
while choosing is not cached, so the last move probed cannot become the question's offer; every
probe is one row, since two probes of different size would make the choice a row-count contest -
the tell that undid 291 and 296. `_check385_moves.py` covers eight properties, three of them
verified by reintroducing the failure.

## 22. 385 READ, AND 386: THE MECHANISM HAS LIFE, THE BALLOT WAS WRONG

385 failed its gate - hit beat the interleave on 2 seeds of 4, not 3 - and the SPLIT says why.

    seed  hit 377r -> 385     step/share/lines        reach 377r -> 385
    1337  .4815 -> .5806  +.099   .910 / .010 / .080   .5257 -> .6153
    4711  .3235 -> .6339  +.310   .867 / .027 / .106   .5201 -> .6896
    2890  .5984 -> .3740  -.224   .336 / .033 / .631   .6260 -> .2113
    8642  .4964 -> .2974  -.199   .521 / .199 / .280   .6185 -> .4159

THE CORRELATION IS PERFECT ACROSS ALL FOUR. Where the mind stayed with `step` it BEAT the
interleave, by +0.099 and +0.310 - s4711 from .324 to .634, the largest single-arm gain this
project has recorded. Where it went to `lines` it collapsed. The split is not degenerate (63%
`lines` on s2890), so the decision is alive; what it learned is a BAD POLICY, because `lines` is
copy - the channel already retired from the standing arm on independent evidence (377r hit .475
against connect-only .599) and re-enabled in 385 only to give that move a lane. Handing the mind
a bad option WITH THE RIGHT TO COMMIT ENTIRELY to it is worse than the same option diluted.

AND A SECOND READING OF 347's LAW. reach: connect-only .653 > connect+copy interleaved .573 >
moves .483, monotone with how much copy participates. At the SAME cap, eight candidates from ONE
source hold the truth far less often than eight from four sources. THE CHANNELS ARE
COMPLEMENTARY, NOT REDUNDANT - diversity of source is worth more than depth in one - and that is
exactly what committing to a single move spends.

What survived: PICK pooled 708/2 of 1061, z +26.50. CONFIRM z -7.61 -> -5.75, its best ever.
GATE-WO fell (1.0000 at 5%, .9242 at 25%, .8450 at 50%) on the collapsed seeds. On s8642,
`hit_of_own` == `own_rival_of_own` == 0.8907 with step 0.1043 - the same signature as 378bf, the
mind collapsing into the counting rival because it stopped stepping - and there, for the first
time, 383's rival scored: count-rival 11 (0.3235).

**386 makes the ballot a flag** (`--move-set step,share`) rather than a constant, validates it,
and WRITES IT INTO THE REPORT, because `move_id` is an index into it and means nothing without
it. It also splits hit BY MOVE (`move_hit`), which is the number 385 lacked: that arm could only
be read by correlating the split against hit across four seeds, and within a seed the same thing
is a measurement rather than a story.

Dropping `lines` is not fitting to 385: the decision to retire copy was taken earlier and on
other data. Gate unchanged - hit >= the interleave on 3 of 4 AND a non-degenerate split. And the
split is itself the second reading: if without `lines` the mind takes `step` nearly always, then
what was alive in 385 was not the ability to choose but the absence of a bad option.

## 23. 386 READ, AND 387: MEASURE THE CEILING OF CHOOSING BEFORE BUILDING A CHOOSER

386 failed its gate on 1 seed of 4, and `move_hit` - added for exactly this - named the fault.

    seed  hit 365r3 -> 386ss   step share/hit         share share/hit    reach 365r3 -> 386ss
    1337  .5448 -> .5054       .793 / .518            .207 / .457        .5878 -> .5102
    2890  .6522 -> .6745       .908 / .689            .092 / .529        .6549 -> .6562
    4711  .6232 -> .6197       .969 / .636            .031 / .115        .7133 -> .7085
    8642  .5746 -> .4384       .677 / .413            .324 / .491        .6540 -> .5024

The 385 pattern repeats with the bad option removed: the more the mind uses a move other than
`step`, the worse the arm. And on three seeds of four the hit OF `share` is below the hit of
`step` - so the mind picks the second move precisely where that move is worse. One probe row is
not enough to tell which lane will pay.

**THE LOSS IS NOT THE PRICE OF COMMITTING, AND THAT IS PROVABLE.** The merged offer is the two
lanes interleaved and cut at eight, so merged is a SUBSET of step_top8 | share_top8. A perfect
chooser therefore reaches at least as much as the interleave, by construction. Measured reach
FELL on three seeds of four. Every point of that gap is chooser error.

Survived: PICK pooled 956/1 of 1363, z +30.87. GATE-WO recovered to 1.0000 at 5/10/25% and
.9794 at 50%. CONFIRM z -4.14, the best this project has recorded - and the trend across the
move arms is monotone: -14.94 (365conn, buggy) -> -7.61 (365r) -> -5.75 (385) -> -4.14 (386).

**387 measures the ceiling before anything is built.** reach through each lane, through the
interleave, and through EITHER lane - a perfect move. Plus `only_share`, where the second lane
is the only route, and the mean Jaccard of the two eights.

THE SUSPECT IS DECLARED BEFORE THE RUN, and it is structural: the arm runs `fp=fillers`, so a
place's fingerprint IS its bag of fillers, and the cosine between two filler profiles is ZERO
unless they share a filler. `connect` walks places that share a filler. SO BOTH LANES DRAW FROM
THE SAME NEIGHBOUR SET and differ only in how they rank it - cosine of the profile against count
of the overlap. If that holds, no chooser can pay, and what is needed is a move that reaches
where the walk cannot rather than a better way of picking between two views of one
neighbourhood.

GATE: oracle - merged > 0.05 AND only_share > 0.05.

The verdict has four branches, and the fourth was added after the smoke test produced a case my
first three read wrongly: lanes that DO differ (only_share .111) while the interleave already
takes both (headroom .000). That reads "choosing cannot add what pooling already has", not "the
lanes are redundant", and printing the second for the first would have been the same class of
misreading this file exists to prevent. VOID below 300 questions.

## 24. 387 READ: THE MOVE IS CLOSED, AND MY STRUCTURAL SUSPECT WAS WRONG

Three settings, one answer.

    setting        step   share  merged  oracle  headroom  only_share  Jaccard  random
    default        .4590  .5157  .5353   .5580   +0.0227   .0990       .1776    .0170
    w1600          .4507  .5377  .5473   .5673   +0.0200   .1167       .1177    .0200
    places 16      .4697  .5157  .5353   .5610   +0.0257   .0913       .1939    .0180

GATE FAILS: headroom +0.020 to +0.026 against a 0.05 bar, stable everywhere.

**THE SUSPECT I DECLARED IS REFUTED.** I predicted `only_share` would be small because both
lanes draw from the same neighbour set - the cosine of two filler profiles is zero unless they
share a filler. The neighbour set IS the same and the prediction was still wrong: only_share is
.09-.12 and the Jaccard of the two eights is .12-.19, so the two orderings produce almost
DISJOINT offers at a cap of eight. Same set, different order, different offer.

**AND `share` REACHES MORE THAN `step`** - .516 against .459 on every setting - while 386's mind
took `step` on 68-97% of questions. It preferred the weaker lane. (Caveat that must travel with
that sentence: the audit's `lane_step` is a count reconstruction of the cosine walk, not the
stage's walk, so the two numbers are not directly comparable to the arm's.)

**WHAT KOSTYA ASKED - combine instead of choosing - IS WHAT THE STANDING ARM ALREADY DOES, and
that is why it is still the best thing here.** merged .535 beats BOTH lanes alone. And the
cleverer version, allocating the eight slots between lanes per question rather than choosing one
lane, is closed by the same number: any allocation's offer is a subset of the union of the two
eights, so its ceiling is the same oracle .558, the same +0.02. Nine allocations, no headroom.

THE MOVE AS AN OUTPUT SPACE IS CLOSED BY MEASUREMENT: pooling dominates choosing, and the
ceiling above pooling is below gate.

**387 then grew a depth block**, because one number was left unmeasured: with merged .535 and a
random floor at .017, 44% of these questions have the truth in NEITHER eight, and that is either
a CUT (present in the lane, below rank eight) or MATERIAL (absent at any depth). Those point at
opposite next steps. The block reports each lane's full size, reach at cutoffs 8/16/32/64/128,
the mean rank of the truth when present, and the union read at the better of the two ranks. It
also makes an asymmetry explicit that had never been written down: `step` sees the fillers of K
places while `share` scores EVERY place sharing a filler, so the two moves were never the same
size of read.

  cut_cost = union@128 - merged.  Above 0.05 the question becomes how many candidates the mind
  can weigh - which 368 tested on a merged offer and never on the depth of one lane. At or below
  it, the misses are not in the lane at any depth, and that is the wall 373, 375 and 376 each
  reached from a different side.

## 25. 387 DEPTH READ: THE CUT BINDS, AND ONE LANE IS BADLY ORDERED

    default (5340 places)        step                      share
      lane length                24.5                      444.6
      present                    .5063                     .7760
      mean rank when present     5.0                       67.6
      @8 -> @128                 .4590 -> .5033            .5157 -> .6750
    w1600 (24004 places)         35.5 / .4893 / rank 6.3   2466.5 / .8797 / rank 315.2
    cut_cost = union@128 - merged = +0.148 and +0.157

`step` IS EXHAUSTED. Its whole lane is 24 values, the truth sits at rank 5, and the eight takes
nearly everything it has: +0.044 from cap 8 to cap 128. Widening the offer cannot help it.

`share` HOLDS THE TRUTH .78-.88 OF THE TIME AND PUTS IT AT RANK 67 (315 at w1600). We offer 8 of
444 (of 2466).

The obvious reading - raise the cap - is the one 347 has already refused three times, and the
numbers agree: 8->16 buys +.045, 16->32 +.043, 32->64 +.033. Reaching .70 from .535 needs
SIXTEEN TIMES the candidates, which is exactly the trade 347 measured as a loss.

**THE REAL FINDING IS THAT THE LANE IS BADLY ORDERED, AND THE DEFECT HAS A NAME HERE - TWICE.**
365's rule is `score[v] += o`, the SUM of the overlaps of every neighbouring place holding v, so
a value accumulates score for being COMMON. On this population the truth is RARE by construction:
the question is here precisely because its answer is not among its own values. 317 found this in
`cons_resolve` and measured the raw-count rival at 2/69 = 0.029 against a one-place rule's 0.222,
seven times worse, and fixed it by dividing. 383 found it again in the count rival, where the
saturated share rule was picking by walk order. Third appearance, same shape.

387 now ranks the connect lane a second way - the MEAN overlap of the places a value stands at,
`score(v) / |places holding v|`, two exact counts and nothing fitted - and reports both orders
at every cutoff.

AND THE CONTROL THAT MAKES IT READABLE: dividing by frequency PREFERS RARE VALUES, and the truth
here is rare. A frequency-matched decoy, drawn from 373's bands, is ranked both ways alongside
it. `rerank_net = truth gain - decoy gain` is the number that decides:

    net > 0.05                 the ordering was the fault. Same eight, same cap, same channel,
                               free reach - and it goes straight into the standing arm.
    truth gain > 0.05, net ~0  rarity, not relatedness. The gain belongs to the null.
    truth gain ~ 0             frequency is not why the truth sits deep; ask what the top of the
                               lane is actually full of.

## 26. 387 RERANK FAILED THE OTHER WAY, AND 388: IS THERE A LEVEL ABOVE A PLACE

**THE RERANK IS REFUTED, AND IT IS REFUTED BACKWARDS.** Ranking connect by the MEAN overlap
instead of the SUM did not lift the truth, it buried it: share@8 .516 -> .166 (default) and
.538 -> .109 (w1600), the truth's rank going 67.6 -> 123.5 and 315 -> 531. The frequency decoy
also fell (-.030, -.025), so this is not a rarity artifact - net -.320 and -.404.

WHAT THAT ACTUALLY SAYS, and it is worth more than the failed gate: connect's signal is
ACCUMULATED WEAK EVIDENCE. Many loosely related places agreeing on a value beats one strongly
related place holding it. The 317/383 precedent - "divide, do not sum" - DOES NOT TRANSFER, and
I transferred it on the strength of two precedents without checking whether the quantity was the
same kind. 365 was right to sum.

**KOSTYA'S DIAGNOSIS, and it names the hole better than any number I have produced.** Asked for
e=mc2 a person does not scan facts: something like science -> physics -> relativity happens
first, with a logic linking each narrowing to the question. THIS PROJECT HAS NO LEVEL ABOVE A
PLACE. Every operation ever run is place -> place - the walk, connect, copy, the deep read, the
moves. There is no "physics" on the tape, only individual holes.

**388 measures whether such a level exists BEFORE anything is built on it**, in two independent
blocks, because the first can degenerate and the second cannot:

  THE LEVEL, by label propagation over the bipartite place/value tape. Linear in the incidences,
  deterministic ties, no threshold. Values come out labelled, so every candidate has a region.
  Measured against a FREQUENCY-MATCHED twin (big regions match by size alone) and against labels
  permuted between values with the size distribution preserved.

  TRANSITIVITY, locally, with nothing global to collapse. A place two hops away shares NOTHING
  with the question - it is related through a bridge - and that relation has never existed here:
  connect is one hop and --reach-depth 2 is a second sequential READ rooted at a candidate.
  Scored by a count: over the bridges, min(overlap(q,b), overlap(b,j)), summed - summed, because
  387 just measured that summing is what this channel wants.

  GATE  in_label@8 - share@8 > 0.05 AND same_label - decoy_label > 0.05, plus hop2_only as the
  separate reading of whether two hops reach anywhere one hop does not.

THREE THINGS THE SMOKE TEST FORCED, all of them guards against reading my own instrument as a
fact about the tape:
  - THE PROPAGATION COLLAPSES. Hub values bridge everything - function words stand in holes too
    - and one region swallowed 82-91% of the values at every weight setting tried. The verdict
    now REFUSES to say "there is no level" when the largest region holds more than half; it says
    the construction collapsed, which is a different claim. `--weight plain|inv` is swept.
  - THE STRONG VERDICT WAS OVERSTATED. "There is no level above a place on this tape" became
    "this construction finds no usable level", because label propagation is one instrument.
  - THE COST. The bridge-to-hop-2 overlap was being recomputed once per shared value; cached, it
    is one pass.

CLOSED AND NOT TO BE PROPOSED AGAIN, recorded here so the next step does not re-run them: strict
two-filler connect (365), intersecting two lenses (346), value-lenses at all (384), the move as
an output space (387), a wider offer (347, four times), mean-instead-of-sum ranking (387).

## 27. A LEAK IN MY OWN AUDITS. EVERYTHING 387 AND 388 PRINTED SO FAR IS INVALID

Found while designing the region-as-source measurement, by re-reading rather than by a number.

**THE HIDDEN TOKEN WAS A SEARCH KEY.** Both audits built the question's neighbourhood from
`prof[pid]` - the place's full filler profile, WHICH STILL CONTAINS THE ANSWER. So the lane
walked to the places holding the truth and then accumulated score for it out of exactly those
places. The stage has never done this: `reach_connect` builds its lens from
`slots[:query_row]`, hidden row excluded. The audits were measuring a channel the arm does not
have.

The size of it, on a fixed synthetic tape, before and after the fix:

    387   step  .628 -> .134     share .611 -> .180

THE LEAK WAS DOING MOST OF THE WORK. Every number read off 387 and 388 in sections 24-26 is
therefore void: `share holds the truth .78-.88 of the time`, `mean rank 67 / 315`, `cut_cost
+0.148 / +0.157`, the rerank comparison, the two-hop block, the region numbers. They have to be
re-run. What survives untouched is anything measured by the STAGE, since the stage never had
this leak - 385, 386 and every arm before them stand.

**AND A SECOND LEAK, IN THE LABELS.** The propagation decides a value's region from the places
it stands at and a place's region from the values standing in it. The hidden token stands at the
question's place, so the truth voted for its place's region and the place voted for the truth's
region - a mutual reinforcement no frequency twin can have. `same_label .5187 against a twin's
.0017` was measuring that. Fixed by choosing the questions FIRST and stripping every hidden row
before a single label is computed. It over-removes - each question's tape also lacks the other
questions' hidden tokens - and that direction is conservative: it can only cost the truth region
mass, never grant it.

**AND THE MEASUREMENT I HAD NOT MADE.** 388 only ever used the region to FILTER the connect
lane, and then reported that a filter keeping 3% of its input loses reach - which says nothing
about the region. The region is a CANDIDATE SOURCE in its own right, and the leaked numbers
already hinted at what that could be worth: `decoy_label .0017` over 4644 values means the
question's region held about eight values, with the truth in it half the time. 388 now reports
`region_size`, `region_reach`, `region_only` (the truth in the region and NOT in the lane's
eight - the route the tape has never had), `region+lane@8` and the candidate COST, so a giant
region cannot look like a win.

Gate, declared: after the leak fix, `region_reach - decoy > 0.05` AND `region+lane@8 - lane@8
> 0.05` at a cost no higher than 16 candidates. If `region_reach` falls to the twin, the whole
thing was the leak and it gets recorded as such.

## 28. THE GOAL WAS SUBSTITUTED. A CORRECTION, WITH THE ASYMMETRY NAMED

Kostya, reading section 27: "ты подменил главный вопрос проекта... чтоб подтянуть цифры мы
делаем 1000 замеров и пробуем 1000 вариантов, но чтоб решить хотя бы 'понимает' протестили
непонятно что и закрыли вопрос." He is right, and this section exists so the substitution is on
the record next to the numbers it distorted.

**THE GOAL, AS STATED BY HIM, REPEATEDLY.** Knowledge separate; A MIND SEPARATE - a full one,
understanding, able to speak; a tape of ANY size; and THE MIND GETS SMARTER as the tape grows or
as it is fine-tuned on other tapes.

**WHAT I REPLACED IT WITH.** "Can knowledge and decision be separated." That is requirement one
of four. I proved it, wrote "the project has earned this", and then spent roughly twenty-five
steps raising `hit` on wikitext - a number that appears in none of his requirements.

**THE ASYMMETRY, IN COUNTS.** Every lever touching the offer got four seeds, a paired baseline,
several settings and often a torch-free ceiling audit first: 347, 365, 369, 377, 378, 379, 385,
386, plus 387 and 388 as ceilings. Against that:

    produces what was never offered   373 (wrong operation, admitted) and 374 - ONE corpus, ONE
                                      window, no seeds. G1 = 0.011. Closed on that.
    combines two facts into a third   NEVER TESTED. 346 intersected two lenses, which is the
                                      nearest thing, and it lives in the constraint family that
                                      384 showed asks the tape the wrong question.
    abandons a wrong path             NEVER TESTED.
    chooses its own question          NEVER TESTED.
    understands / speaks              NEVER OPERATIONALISED. The exam ranks eight names in a
                                      hole. There is no question in this project that asks for
                                      understanding or for speech, so nothing about them has
                                      been measured, let alone closed.
    smarter as the tape grows         NEVER MEASURED CLEANLY. 369 scaled the QUESTIONS 5x, not
                                      the tape, and reported that the separation margin grows
                                      while accuracy does not. Tape size varied (w400, w1600,
                                      w8000) only alongside other changes.

**THE DOUBLE ACCOUNTING, WHICH IS THE WORST OF IT.** "Fine-tuning on a second tape buys
nothing" was recorded as one of the FOUR CONTROLS PROVING SEPARATION. It is simultaneously a
direct failure of his requirement four - a mind that does not get smarter from another tape. I
counted that number as a win for my version of the goal and never once counted it as a loss for
his. One measurement, entered on one side of the ledger only.

**WHAT THIS SECTION DOES NOT DO.** It proposes no lever. Every trait above is untested or
tested once, and the honest next move is to decide WHICH of his four requirements to
operationalise properly - with the same four seeds, paired baselines and declared gates that the
offer levers were given - rather than to add a fifth way of reordering eight candidates.

## 29. THE PLAN TO THE ACTUAL GOAL: SPEAKS / UNDERSTANDS / GROWS WITH THE TAPE

Written after section 28 named the substitution. This is the route from what exists to the goal as
the user states it: knowledge apart, a full mind apart (understanding, able to speak), a tape of any
size, and the mind getting smarter as the tape grows or as it is fine-tuned on other tapes.

### 29.1 What exists, and which of it is a brick

  * THE TAPE. Mentions counted out of raw wikitext; places `(w, left, right)`; fillers. It is
    counted, never trained, so its size is free. BRICK for "any size" - and the only requirement
    already met without argument.
  * PHI. One scalar, 5633 params at d=32, mean+max pooling. Holds no facts; carries across to a
    second tape without retraining (the four controls of the separation result). BRICK for "mind
    apart".
  * THE READ MACHINERY. Four channels (step / connect / lines / home), the second read with an
    honest deep root (380/381), the move (385/386) that lets the mind pick WHERE to look before it
    sees an offer. BRICK, raw, for "a seeker".
  * WHAT IS ACTUALLY PROVEN. Given eight names in a hole, Phi picks the right one: PICK rival 0-2
    per thousand, z +25.9..+34.4, GATE-WO 1.0000 at 5/10/25%. That is RANKING INSIDE A GIVEN
    OFFER. It is not speech, it is not understanding, and it was never a measurement of tape size.

### 29.2 The three buildings that do not exist yet

**A. SPEAKS.** Today the output space is eight candidates handed over by the proposer. Speech has no
candidate list. What is needed: an autoregressive loop where at each step the offer is the fillers of
the CURRENT place (or its neighbourhood) with no gold inserted, Phi ranks, the top is emitted, the
position steps. Measure: continue N tokens of a held-out line; exact-match rate, and the share of
emitted tokens that came from outside the question's own row. Gate declared before the run.
THE WALL: 347's law - a wider offer costs more than it brings, measured four times. Speech is the
widest possible offer (the whole vocabulary of a place), so the wall stands exactly across the door.
WHAT WAS TRIED AGAINST IT: 376 copy, 378 copy-backfill, 385/386 moves, 387 rerank. Every one of them
REORDERS an offer that is already eight long. Not one of them widens the output space. The door has
never been pushed on directly.

**B. UNDERSTANDS.** Never operationalised - the exam only ranks eight names in a hole, so the project
contains no question about understanding at all. It needs a definition that is a COUNT. Three
candidates, each its own run:
  * B1 produces what was never offered. Tested ONCE (373), one corpus, one window, no seeds - and
    with the wrong operation: concatenation of whole values, when inflection is what German asked
    for. Its own SHARE-vs-ASSEMBLE table was evidence against my reading of it.
  * B2 combines two facts into a third. NEVER TESTED.
  * B3 abandons a wrong path. NEVER TESTED.
THE WALL: Phi carries no state between steps. B3 needs a refusal-move, B2 needs a second row folded
into one answer. Both are structure, not a parameter.

**C. GROWS WITH THE TAPE.** Never measured cleanly: 369 scaled the QUESTIONS, not the tape. What is
needed: the same Phi, the same question set, the tape cut to 1x/2x/4x/8x - accuracy must rise
monotonically, and that monotonicity IS the gate. Then the cross-tape leg: fine-tune on tape A, test
on tape B.
THE WALL, AND THE DOUBLE ACCOUNTING: "fine-tuning on a second tape buys nothing" is on record as one
of the four controls proving separation. It is simultaneously a direct failure of requirement four,
and it was never written down as one.

### 29.3 Order, and why

  1. **C first.** Cheapest: the code exists, only the tape has to be cut. The gate is monotonicity,
     which needs no new metric and cannot be fitted. It also answers a question that is currently
     answered twice in opposite directions.
  2. **A second.** Expensive and it is the real thing - "speaks". It must be pushed on directly
     rather than approached through a fifth reordering of eight candidates.
  3. **B last**, because B is defined THROUGH A: without an output space there is nothing for
     "produced what was never offered" to mean.

### 29.4 Debt that must be cleared regardless

  * 387 and 388 must be re-run after the 27 leak fix. Every number they printed is void.
  * No number from a leaked audit may be cited in support of any step above.

## 30. 389: THE GAUGE. WHY THREE REFUSAL ATTEMPTS COULD NOT HAVE WORKED, AND THE ONE TERM THAT CAN

### 30.1 The correction that had to come first

I proposed a "null world" - a ninth option, trained on questions whose gold is absent - and it is
CLOSED. It exists already: REFUSE_LABEL is a world in both stages, on by default, and
`reach_reward` pays 1.0 for refusing an unanswerable hole. speak_term's own docstring records the
outcome three times over (299_hash, 311's first pair, 339): on a tape where 87% of holes are
unanswerable, "always refuse" is the arithmetic optimum of the reward, so the base rate answers
before the mind does. Re-proposing it would have been a fourth run of a closed lever.

### 30.2 What is actually broken, stated exactly

Every gradient Phi has ever received arrives through a softmax over the worlds OF ONE QUESTION, and

    softmax(l + c) = softmax(l)

so a constant added to every world of a question is INVISIBLE to the loss. Each question carries
its own free offset. Phi's raw value is therefore comparable only within one question, and that is
not a shortage of training - it is a SYMMETRY OF THE OBJECTIVE. It is also the same fact 329, 337
and 340 each reported in different words: "no absolute scale, only the gap carries".

This explains the three refusal failures at one stroke, and explains them structurally rather than
by the base rate alone: REFUSE lives inside the same per-question softmax, so it moves with the
offset like everything else and what it contributes is again a gap. speak_term (341) fixed the
SCOPE of the decision - one unit of speech spent across a batch - but it compares MARGINS, and a
margin is gauge invariant too. The gauge has never been touched by anything in this project.

### 30.3 The number that was on screen for eighty steps and never read

`rankblock` has always computed the AUC of `mind_score` - the RAW Phi of the settled world - beside
`mind_margin`, and `_read299` has always printed it, in a parenthesis: `AUC mind X (score Y)`. It
appears in every dump the run has produced. It is never mentioned once in sections 1-29 of this
file. It is the gauge measurement, and it is free.

A free per-question offset predicts `mind_score` AUC ~= 0.50 no matter how good the mind is, while
`mind_margin` was 0.866 at 337 and 0.969 at 352. That prediction is a THEOREM about the objective,
not a guess, which is what makes it a usable baseline. 389 therefore has a control that costs
nothing: read the GAUGE line off any existing dump before running anything.

### 30.4 The term

`--calib-batch B` (with `--calib-weight`, declared 1.0, never swept). B questions are scored
together; each contributes the raw Phi of the world it settled on; ONE softmax ACROSS QUESTIONS,
against the uniform distribution over the ones the tape can actually answer.

  * Nothing new is asked of Phi. It is still a comparison - the only thing it has ever done - but
    the things compared belong to DIFFERENT questions, which is the only way B-1 of the B free
    offsets can be removed. One global offset survives, and one is all a single threshold needs.
  * IT CANNOT BE WON BY A CONSTANT. The term is shift invariant, so pushing every score down
    changes nothing. There is no "always refuse" here to meet - the trap that took the three
    earlier attempts is structurally absent rather than argued away.
  * THE TEACHER IS THE TAPE'S. The label is `answerable` (truth among the candidates), a property
    of the tape and the walk. Not `right`: training on the mind's own current correctness is a
    moving target, and it would consume the one target the result has to be read on.
  * No new parameter. It is a SECOND TERM, and 321 and 341 each priced a second objective at ~4x
    the route on these 5633 parameters - so it is off by default, reports the route, and is never
    pooled with its control.

Shares the batch machinery with `--speak-batch` and nothing else: B is the larger of the two, each
term is added only if its own flag asked for it, and both accumulators are cleared in a `finally`.

### 30.5 THE GATE, DECLARED BEFORE THE RUN

Arm `--calib-batch 8`, control the identical arm with the flag off, four seeds, matched question
budget (`--train-steps` divided by B).

  1. PRIMARY. `mind_score` AUC on target `answerable` must beat the control on at least 3 of 4
     seeds, and must beat BOTH counting rivals (`count_n_own`, `count_top_share`) on at least 3 of
     4 - 337's pre-registered rule, unchanged, applied to the raw score instead of the margin.
  2. TRANSFER, and the one that would mean something. The same must hold on target `right`, which
     the term never saw.
  3. GUARD. PICK rival must not degrade pooled, and GATE-WO must stay 1.0000 at 5/10/25%. A scale
     bought by wrecking the ranking is not a scale.
  4. VOID CHECK, read first. If the control's own `mind_score` AUC is already well above 0.50, the
     gauge was never as free as the algebra says and the whole premise of this step is wrong -
     read that before reading anything else.

### 30.6 What this is NOT

It is not "understands", and it is not a step toward reordering eight candidates better. It is the
precondition named in 29: a mind cannot carry a result between reads until it can tell that a read
FAILED, cannot abandon a path without "not here", and cannot stop generating without a threshold.
All three need one scale. This step buys the scale and nothing else, and if it fails the honest
reading is that Phi's single scalar cannot hold one.

## 31. 389 IS VOID. THE GAUGE WAS NEVER FREE, AND THE VOID CHECK COST NOTHING

Gate 4 of 30.5 - "read this before reading anything else" - fired on the control, before the arm
was ever run. 389 is closed without a training run.

### 31.1 The numbers (365r3, held, four seeds, target `answerable`)

    seed    raw-score AUC   vs coin   margin   rivals own/share
    1337        0.7191       +0.219    0.780     0.396 / 0.501
    8642        0.7250       +0.225    0.757     0.470 / 0.497
    2890        0.6385       +0.139    0.740     0.373 / 0.500
    4711        0.6643       +0.164    0.750     0.477 / 0.500

Target `right`, which the term would never have seen: 0.723 / 0.713 / 0.616 / 0.738.

On all four seeds the RAW score already beats both counting rivals. The thing 389's gate was
built to buy, a standing world already holds.

### 31.2 The exact error, which is mine

The algebra is right: `softmax(l + c) = softmax(l)`, so the loss does not constrain a
per-question offset. THE INFERENCE IS WRONG. Phi is a function of the world's CONTENT with shared
weights - the offset is not a free parameter attached to each question, it is whatever the shared
function emits for that question's material. Unconstrained by the loss is not the same as
arbitrary. A symmetry of the objective does not become a symmetry of the learned function, and I
turned one into the other in a single step and called the result a theorem.

It is the same shape of mistake as 387's rerank (a precedent transferred without checking the
quantity was the same kind) and as 377/369's gates (a gate specified on a composition-dependent
quantity). Three times now the fault has been an argument that looked like arithmetic.

### 31.3 It was refutable from dumps already in hand

353margin s1337 prints `AUC mind 0.860 (score 0.872)` - the parenthesis was never a coin. Of every
s1337 dump carrying `question_rank`, exactly three read FREE: 366own 0.500, 341speak 0.476,
362marg 0.530 - and none of them is the standing world. The prediction was already contradicted by
the same dumps I quoted in 30.3 when I said the number had never been read. It had never been read
by ME; it was on screen, and it disagreed.

### 31.4 What survives, and what does not

  * DOES NOT: `--calib-batch` as a step. The premise is false on the control, and running it would
    pay a second term at ~4x the route (321, 341) for a scale that is not empty. The flag stays in
    the code, off, with this section named at its docstring so it is not re-proposed a fourth
    time - the refusal lever's own history is the reason that matters.
  * DOES: the reading of why the three refusal attempts failed (30.2) is unaffected - REFUSE
    inside a per-question softmax contributes a gap, and "always refuse" meets the base rate.
    That was never the gauge argument.
  * DOES: the GAUGE line in `_read299`. It reports a real quantity that had gone unread for eighty
    steps, and its answer is the opposite of what I expected, which is exactly what it is for.
  * DOES: 30.6's dependency claim, now standing on its own without the gauge story. A mind still
    cannot carry a result between reads, abandon a path, or stop generating. Those need a
    MECHANISM, and 389 shows they will not fall out of a better-scaled score.

### 31.5 The open direction, in the user's words

"The next move is not another term on Phi, but the thing a scale does not give either: THE MIND
STILL DOES NOT CHOOSE WHERE TO LOOK."

`--moves` (385/386) is the only thing built in that direction and it chooses among three fixed
channels. Nothing in the project lets the mind name a place it was not offered. That is the same
wall as A (speaks) in section 29 approached from the other side - an output space - and it is
where the next lever has to go, not into another read-side reordering.
