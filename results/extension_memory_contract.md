# TapeLM memory contract — three layers + family W

> **Shipping trunk:** **221 → 227 → 228c → 230 → 226c**

**Status:** **product contract** for TapeLM variant A (same encoder, same repo as 192–205).  
**API grain:** `W` is **per domain-family**, not per corpus name. See also [`docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md).

---

## Shared map + domain bundle

```text
SHARED:  arc_enc (frozen) → one fp geometry in R^d
PRODUCT: canonical slots + W_family@read + decode + resolve (+ optional head_family)
```

W is a **lens** (change-of-basis), not a second map. Heads specialize generation without moving the map (213/225). Read prose slots with `W_prose`; cross-family read costs drop (224 ~0.12 for code).

**Stage 225:** `_stage225_family_fork.py` — (A) legal reuse of `W_prose` vs fork; (B) `head_prose` / `head_code` with frozen `arc_enc`.

**225 full:** `DOMAIN_BUNDLE_OK` — legal **no fork** (reuse 0.92 ≥ matched 0.90); heads specialize (cross_drop stories **0.33**, code **0.70**); fp drift **0**.

**227 full:** `CANONICAL_STORAGE_OK` — write = canonical fp; read = **qmap** (domain→old) preferred over keylift; cross-code recall **0.95**, drop vs same **0.05**. Unifies to **one bank + disposable W**.

**Architecture after 227:** not `{W, head, slots}_family × N`, but **`slots_canonical` + `W_family`@read + `head_family`**. One memory, many lenses.

**226:** `JOINT_GEN_MEM_NO` — retrieval ≠ utilization (gold_inject ≈ no_inject). Path A = honest boundary; 226b probes seed reconcile + code-native inject (C), not joint SFT (B).

**226b full:** `RETRIEVAL_OK_UTIL_BOUNDARY` — H1: qmap recall **1.0** under 227-matched protocol (226’s 0.60 = seed/n). H2: code_comment **0.47** vs none **0.43** (negligible); assignment **hurts**. Accept **A**; C not enough at this scale; B = open branch.

**228b:** `FP_GUIDED_DECODE_NO` — global argmax retrieve. **228c:** `FP_DECODE_FIX_YES` — 4-way retrieve (227) → fp-scorer **1.0** vs head **0.48**. Contract: *decode-time scorer works when retrieval matches the 4-way slot protocol*; raw `cos(fp(c), query)` is the wrong object (ctx mix).

**Official API:** [`docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md) · `_tapelm_ext.fp_decode_pick_retrieved_4way` · persisted W via `artifact/scripts/export_w_registry.py`.

**229:** `CONTRADICTION_RAW_MEMORY_OK` — multi-hit candidates; resolution not in slot layer.

**230:** `RESOLUTION_POLICY_OK` — `query_cue` / `composite` macro **1.0** vs raw argmax **~0.47** on mixed neutral+cued queries; policy layer closes 229 gap.

**226c:** `JOINT_FP_DECODE_OK` — cross-domain **recall_4way ~0.88**, **fp_retrieved_4way ~0.88** vs head **~0.45** (228c wired at code return); old 226 global retrieve **~0.60** `JOINT_GEN_MEM_NO`.

---

## Three layers

| Layer | Name | Contract | Evidence |
|-------|------|----------|----------|
| **L1** | **Freeze** | Default: `arc_enc` frozen → zero-train fp API (lexicon / slots / hop / edit). Absolute **fp** stability under that freeze. | 213 |
| **L2** | **W-remap** | If encoder *did* shift: keep slot bank; apply **`W_family`** (change-of-basis / restorer). Prefer migration over full reindex under B-shifted encoder. | 221, 222 M3≫M7 (intra-run), 224 |
| **L3** | **Stream decay** | Long-running: age / refresh / unlearn so the bank stays bounded. | 219 |

L2 does **not** replace L1. L3 does **not** replace L1/L2. Product story: *Stable fp · Migratable slots · Bounded stream*.

---

## L2 detail: family registry (not per-domain dict)

224 suggests a **hierarchy**, not binary “one W vs infinite W”:

| Family | Members (examples) | cos(old,new) @224 | Cross-drop vs best wrong |
|--------|--------------------|-------------------|---------------------------|
| **prose** | wiki, stories, med, (news, legal?) | 0.67–0.92 | stories ≤0.05; med weak shift |
| **code** | Python, (JS, SQL…) | ~0.59 | ~0.12 vs prose W |
| **outlier** | math / music / chem formulas | ? | open |

```text
W_registry = {
  "prose": W_prose,   # shared across narrative/wiki-like
  "code":  W_code,    # programming surface forms
  # "math": W_math,   # fork when measured
}
```

**Reuse rule:** new corpus (e.g. legal) → try `W_prose`; if cross-drop vs matched **&lt; 0.05** on fixed exam → **reuse**; else **fork** new family.

**Caveat:** med @224 had mild drift (cos **0.92**); do not overclaim “all prose forever.” Family membership is **empirical**, validated by drop, not by label alone.

---

## Runtime policy (cos trigger)

After any arc_enc shift, measure `mean_cos = mean_i cos(fp_old(w_i), fp_new(w_i))` on a **fixed core vocab** (~800).

| Band (working defaults) | Action |
|-------------------------|--------|
| **cos &gt; 0.85** | `W_active = I` (geometry ~intact) |
| **0.65 ≤ cos ≤ 0.85** | `W_active = registry[family]` if present, else learn once |
| **cos &lt; 0.65** | need **family W** (learn or load); code-class shifts |

```python
# Pseudocode — see WFamilyPolicy in _tapelm_ext.py
if mean_cos > 0.85:
    W_active = Identity
elif family in registry:
    W_active = registry[family]
else:
    W_active = learn_W(core_old, core_new)  # ~800 words, one-shot
```

Thresholds are **policy knobs** seeded by 224, not physics laws. Re-tune on fixed exam (222b).

---

## 224 → “restorer”, not crutch

Code: cos **~0.59**, matched W recall **~0.88**. Strong deformation still **structurally recoverable** by linear W on the core. That strengthens L2 claim: **migration without reindex** is for **hard** shifts, not only mild prose drift.

W matrices between families are **not** the same (cos_flat **~0.43–0.53**) → family registry is justified even when some prose pairs look interchangeable.

---

## What this adds (value)

1. **Honest continual-learning alternative:** weights may drift; **slots stay**; W restores coordinates.  
2. **Bounded ops cost:** O(d²) per family, not reindex O(slots).  
3. **Scalable product API:** few families + fork-on-drop, not one W per customer corpus.  
4. **Clear negatives:** partial FF (216), recency-as-domain (214), **210–212 `THESIS_NO_AT_SCALE`**, 207 falsified — stay off the trunk.

---

## Next engineering (order)

1. ~~Persist `checkpoints/w_registry/` (`W_*_bwd.pt`, manifest)~~ — **done:** `export_w_registry.py` + [`docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md).  
2. **~~229+ resolution policy~~** — `_stage230_slot_resolution.py` + `_tapelm_ext.resolve_slot_contradiction`.  
3. **~~226 e2e~~** — `_stage226c_joint_fp_decode.py` (228c at return token).  
4. ~~Wire L3 decay with `W_active` version id on slots~~ — **232** `STREAM_W_VERSION_OK`.
5. ~~Temporal W / tool bind / compositional W~~ — **231 / 233 / 234** OK (ops, not trunk).
6. **Open:** full multi-domain **L1 pretrain** (235 = bounded probe only).

Code: `_tapelm_ext` (`WFamilyPolicy`, decode API). Narrative: this file + [`extension_thesis_W_remap.md`](extension_thesis_W_remap.md).
