# Stage 266 instruct trunk ladder

**WORDS_FORMULATE_QUERY** · remap_261=`QUERY_MUST_BE_WORDS` · bank=4352 exam=352

## Fit vs eval (mixer diagnostic)

| trunk | FIT top1_sem | EVAL top1_sem | FIT 20way_sem | EVAL 20way_sem | EVAL fp | overfit |
|-------|-------------:|--------------:|--------------:|---------------:|--------:|:-------:|
| qwen05_base | **0.284** | 0.034 | 0.699 | 0.148 | 0.222 | False |
| qwen05_instruct | **0.085** | 0.062 | 0.290 | 0.227 | 0.227 | False |
| qwen15_instruct | **0.085** | 0.062 | 0.290 | 0.227 | 0.227 | False |

## Matched pair

| trunk | metric |
|-------|--------|
| 0.5B base | eval sem **0.148** / fp 0.222 / top1 0.034 / a=0.72 / FIT top1_sem **0.284** |
| 0.5B-Instruct + chat template | eval sem **0.227** / fp 0.227 / top1 0.062 / a=0.00 / FIT top1_sem **0.085** |
| Δ sem 20-way | 0.07954545454545453 |
| Δ fp_only 20-way | 0.005681818181818177

## Ladder

| id | result |
|----|--------|
| qwen05_base | eval sem **0.148** / fp 0.222 / top1 0.034 / a=0.72 / FIT top1_sem **0.284** |
| qwen05_instruct | eval sem **0.227** / fp 0.227 / top1 0.062 / a=0.00 / FIT top1_sem **0.085** |
| qwen15_instruct | eval sem **0.227** / fp 0.227 / top1 0.062 / a=0.00 / FIT top1_sem **0.085** |
| qwen3_instruct | n/a |

## Word-vote arms (0 train)

| arm | top1 | median | 20-way | silence | low-ov silence | low-ov\|vote |
|-----|-----:|-------:|-------:|--------:|---------------:|-------------:|
| surface words | 0.199 | 70.5 | 0.426 | 0.449 | 0.816 | 0.062 |
| prompted keywords | 0.159 | 46.0 | 0.398 | 0.540 | 0.885 | 0.100 |
| surface ∪ keywords | 0.165 | 86.5 | 0.432 | 0.449 | 0.816 | 0.062 |
| paraphrase novel | 0.023 | 407.5 | 0.062 | 0.920 | 0.931 | 0.333 |
| surface ∪ paraphrase | 0.188 | 88.0 | 0.455 | 0.426 | 0.770 | 0.000 |
| trained W (matched Instruct) | 0.062 | — | 0.227 | — | — | — |

Paraphrase bridge: novel_on_tape=0.756, queries_with_bridge=0.545, woken_silent=0.023

## Gates

- G_instruct_beats_base_matched: **False** (alpha_collapsed_compare_fp_only)
- G_ladder_monotone: **None** (alphas_collapsed_trunk_out)
- G_prompted_query: **True**
- G_prompted_beats_surface: **False** (headline: top1/20-way)
- G_prompted_median_better: **True**
- G_union_beats_surface: **False**
- G_mind_refines: **False**
- G_paraphrase_breaks_silence: **False**
- G_paraphrase_novel_on_tape: **True**
- G_paraphrase_useless: **True**
- G_words_crush_learned: **True**
- G_any_words_suffice: **True**
- G_mixer_overfit: **False**
- G_harvest_helped: **True**

## vote_rank fix audit

vote_rank fix (empty sc→n_slots): 261f and 264 votes top1 unchanged at 0.245562 (empty_sc_n=0/338 both; delta=0). Artifact was 266 paraphrase-only.
