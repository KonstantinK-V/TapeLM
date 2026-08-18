# Stage 256 slot-bias glue

**SLOT_BIAS_GLUE_NO** trunk=stage253_joint_l02.pt slots=1248 eval_facts=24

- EM free-form: head_only **0.000** -> glue **0.875**
- causal: shuffled **0.875**, empty **0.000**
- slot delete: target 0.88 -> 0.00, retained 0.88
- gate: fact **0.959** vs prose **0.000**
- prose CE glue off 4.612 -> on 4.612 (hold CE 3.882)
