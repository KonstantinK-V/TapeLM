# Stage 258 semantic query

**SEM_QUERY_OK** trunk=stage253_joint_l02.pt slots=1712 chance=0.12

- unseen paraphrase: fp-only **0.000** -> +sem **0.646**
- seen relation:   fp-only 0.057 -> +sem 1.000
- anchored (must not regress): 0.758 -> 0.762
- blend a: paraphrase 0.472 vs anchored 0.347 (blend now sees fp top1-top2 margin)
- unseen_para reading: **gpt_parity**; predicted relations {"lead": {"work": 27, "lead": 5}, "birth": {"marriage": 2, "birth": 30}, "death": {"marriage": 28, "study": 1, "death": 2, "work": 1}, "marriage": {"marriage": 26, "work": 6}, "work": {"work": 30, "marriage": 2}, "prison": {"prison": 31, "lead": 1}}
- bank-wide top1 0.646, shuffled keys 0.125
- matched GPT-2 unseen paraphrase: 0.276
