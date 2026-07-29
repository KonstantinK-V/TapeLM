# Stage202b — decisive B: encoder fine-tune on PAWS

**Overall:** `SEM_B_CAP_NO`

- curve: PAWS **0.705** | 179 para 0.806 / hard 0.941 (**inversion=False**) | next_tok(copy) 0.55
- gpt:   PAWS 0.701 | para 0.547 / hard 0.746 (inversion=False)

gates: {'g_paws': False, 'g_inversion': False, 'g_parity': True}

Encoder fine-tuned end-to-end (copy); product P1 frozen. Tests whether the curve substrate CAN reach semantic invariance given a meaning signal + trainable encoder, at parity with GPT.