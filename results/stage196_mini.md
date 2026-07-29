# Stage196 — TapeLM assembled stack + anti-clone gate

**Overall:** `TAPELM_PARTIAL`

- **parity (entry ticket):** curve 0.867 vs gpt 0.843 (Δ+0.023) — hold=True
- **recall (win):** curve_fp 0.947 vs gpt_param 0.300 vs **gpt+rag 0.980** (chance 0.25)
- **calibration (win):** curve_lexAUC 0.982 vs gpt_bpeAUC 0.380
- **one-shot edit (win):** curve 0.417 vs gpt 0.283 (n=60, chance 0.25)

gates: {'parity_hold': True, 'recall_win': True, 'recall_beats_rag': False, 'calib_win': True, 'edit_win': False}

One frozen curve encoder serves generation + fact memory + lexical calibration + one-shot edit from a single shared fp-space. Win counted only where BPE-GPT is structurally weak (recall/calib/edit), with GPT+RAG as the nearest rival control.