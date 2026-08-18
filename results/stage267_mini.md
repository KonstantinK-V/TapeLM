# Stage 267 read-refine

**READ_REFINE_INVALID** · model=Qwen/Qwen2.5-0.5B-Instruct · bank=451 · eval=51 · trained params **0**

| arm | top1 | median | 20-way | silence |
|---|---:|---:|---:|---:|
| A hop0 surface | 0.353 | 4.0 | 0.549 | 0.373 |
| B refine grounded | 0.314 | 6.0 | 0.549 | 0.373 |
| C refine RANDOM passages | 0.235 | 10.0 | 0.549 | 0.373 |
| D refine selective | 0.353 | 5.0 | 0.569 | 0.373 |
| E refine blind | n/a | n/a | n/a | n/a |

- woken (grounded): **0** queries, top1 among them nan; random control woke 0
- copy rate from shown passages: **0.636**
- hop0 uncertain: 26/51

## Gates

- G_hop0_reproduces_264: **False**
- G_refine_beats_hop0: **False**
- G_grounding_causal: **True**
- G_silence_reduced: **False**
- G_woken_useful: **False**
- G_selective_beats_always: **True**
- G_reads_passages: **True**
- best_arm: **D_selective**
