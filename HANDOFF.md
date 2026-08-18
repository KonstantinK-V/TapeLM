### WHAT 299 ACTUALLY SETTLED (a–f, four seeds, hash ink, frame tape)

**Void first.** 299a was void as a payoff measurement: reachable 10%, so silence pays 0.975 and
the mind matched it to four decimals — found_rate 0.0. Correct play, no information. `--reach-no-refuse`
removes the bet and scores hits instead.

**Replicated, all four seeds.**
- The filler-bag fingerprint (`--frame-fp fillers`) gives the walk direction: reachable
  17.95/18.85/20.80/19.45 against reachable_random 1.85/1.65/2.10/2.05. Nine to ten fold.
  The address fingerprint hashed six characters like `the|of` and carried nothing.
- The walk saturates at eight places — `reachable_wide` tracks `reachable` to a tenth of a point,
  so the bound is the fingerprint, not K.
- `--reach-lookahead` (step logit = max(l2), no blob world) woke the step: 0 → ~1%.
  `step_vs_size_r` −0.03..−0.07, so row count is not what moves it.
- Steps are selective: `steps_on_walk_only` 0.64/0.40/0.46 against a 5% walk-only base.

**Refuted.** The walk-only advantage. Seed 1337 read 12/6; the next three read 5/7, 6/6, 9/9.
Pooled 32 against 28 on 401 questions, z 0.52. A lucky throw, found in 24 minutes of extra seeds.

**Unchanged across everything.** The mind ties counting wherever it is measured: hit_rate ≈
own_rival_hit_rate, hit_of_own ≈ own_rival_of_own. Live channels are the reason — `rare_nonzero_rate`
0.0009 is dead, and `cos_mean` near 1 is NOT evidence either way (most row pairs in a world share
an address, so they share a fingerprint exactly; I once read that number as a channel test and was
wrong — `reachable_rate` is the test).

**The route beats counting, on four seeds twice over.** At an equal step budget the mind puts
38 of 77 steps (299f) and 40 of 85 (299h) on walk-only questions; a counting router given the
same budget puts 8 and 10. Enrichment ~10x against ~2x over a 5% base, and the mind wins on
every seed of both arms. This is the first replicated win over counting in the line, and it is
about WHEN to look, not what to say. `--reach-import homes` destroyed it (enrichment fell to
~1.0, below the counting router) by letting worlds grow with arbitrary imported rows - the
row-count marker took the step over. Standing arm: `--frame-fp fillers --reach-lookahead
--reach-import walk`.

**CONFIRMED, on four fresh seeds registered in advance: `--reach-home-cos` beats counting at
ANSWERING.** On the CONFIRM subset - the answer already stands at the place - pooled 156 against
74 of 1630, z +5.41, every seed positive (+2.32/+2.88/+3.16/+2.63), +5.0 points of accuracy. It
is not a blanket preference for local values: every candidate there stands at this place, and
what separates them is where else they stand.

**The same feature makes walk-only significantly WORSE** - pooled 13/37, z -3.39, a loss and not
a tie. The mechanism is the feature working as designed against the wrong question: home_cos asks
whether a value usually stands in places like this one, which is true of a CONFIRM answer by
construction and false of a walk-only answer by definition. It penalises exactly the answers the
walk exists to reach.

**And the route is now CONFOUNDED.** With home_cos on fresh seeds, enrichment is
0.00/1.13/4.74/6.88 against a counting router at 3.02/1.13/2.37/1.97 - two of four, not four of
four. But the routing win was only ever measured on seeds 1337/8642/5200/2718, so seeds and
feature moved together. The missing cell is the standing arm WITHOUT home_cos on the four fresh
seeds; until it is run, the route claim above is not settled.

**So: routing works, picking does not.** The open question is whether the routing is a decision
or a count in a costume — `router` in the report gives a counting router the same step budget and
prints both enrichments.
