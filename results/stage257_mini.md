# Stage 257 fp composition (two-hop)

**FP_COMPOSE_OK** trunk=stage253_joint_l02.pt slots=1296 eval_chains=16

- EM 2-hop: head_only **0.000** -> one-hop-only **0.000** -> hop loop **1.000** (value in first 3 tokens: 1.000)
- retrieval@cue 2-hop: chain **1.000**, answer reached 1.000, halt correct 1.000 (shuffled chain 0.000)
- unseen relation pair: **1.000**
- causal: shuffled 0.000, empty 0.000, no-edge1 bank 0.000
- delete middle edge: 2-hop 1.00 -> 0.00, its B->C 1.00 -> 1.00, others 1.00
- expected hops: 1-hop q **1.00** vs 2-hop q **2.00**
- prose CE glue off 3.985 -> on 3.987
