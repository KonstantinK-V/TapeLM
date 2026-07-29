# Weights-as-bank (side note — not a live gate)

**Status:** open reflection / future side branch.  
**Not** SQL. **Not** LM chat. **Not** “listen to API like a server.”  
**Not** blocking 150–158 or BDLM-after-LM-ceiling.

Related: `results/sote_bdlm_product_layer.md` (fact-RAG / hops product). This note is narrower: **storage medium**.

---

## One-line idea

Use **model weights (or in-model memory slots)** as an **exact data bank** — without a conventional external index (B-tree / chunk store).  
“Find” ≈ forward / lookup in parameters, not scan a table structure.

Index does not disappear: it is **baked into weights** (learned or written map key → value).

---

## Why it is interesting

- Contrast with classic DB: structure + indexes + search/scan.  
- Contrast with soft LLM-KB: here the bar is **exact** on a **frozen** closed set.  
- Possible win vs **expensive** search (full-text / semantic / raw scan) — **not** automatically vs `PRIMARY KEY` hash/B-tree (those are often cheaper than huge matmuls).

---

## Already exists (other names)

Embedding/codebook rows, memory nets / Hopfield-like, memorizing transformers, overfit maps on fixed banks, LAMA-style facts-in-weights (usually **not** 100% exact).

Industrial “replace SQL Server with only continuous Transformer weights + ACID” — basically no.  
Practice: discrete slots/tables for facts; weights for encode/retrieve/speak.

---

## SOTE angle — text vs numbers (different roles)

Do **not** jam everything into one softmax:

| Role | Carrier | Job |
|------|---------|-----|
| Text / meaning / morph | fp, word/morph codebook, compose | who/what in language |
| Exact numbers / ids / quantities | separate digit/value channel or slots (F85 digit path spirit) | 100% get, no semantic blur |
| Relations | discrete hops | who–rel→whom |
| Fast exact get | frozen emb rows / SoftPhrase / value memory | map without text index |

Text = how to name and speak; numbers = exact magnitude; hops = edges.

---

## Triple constraint (keep honest)

1. **100% exact** → closed frozen bank (or explicit slot write).  
2. **No distribution shift promise** outside that bank.  
3. **Minimal “thinking”** → good for get(key); arbitrary analytics still need algorithm or query layer outside.

---

## Program placement

1. **Now:** SOTE-as-LM queue (150→158).  
2. **After LM ceiling:** BDLM / neuro-symbolic fact extract (hops API).  
3. **Side fantasy / later dig:** weights-as-exact-bank + text≠number roles (this file).

Do not mix into STORY gates or claim “faster than SQL on primary key” without a measured bakeoff.
