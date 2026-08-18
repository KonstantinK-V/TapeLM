# Thought as a marked slot

Status: **design** (not a stage, not measured).  
Keeps the SOTE invariant. Drops the fantasy of 100% certainty.  
Related: handoff §6b, stages 279 / 288 / probe 289b.

---

## Contrast

**GPT.** Point in a continuous space → decoder → text.  
Mixing happens inside one forward pass and vanishes. The output carries no trace of what was mixed. Interpolation and hallucination are the same channel.

**Proposal.** Point → **write a slot on the tape** → output only **reads** the slot.  
Between thought and speech stands a record.

The difference is not cosmetic. A thought becomes an **object**, not an **event**. An object can be quoted, disputed, deleted with its premises, recomputed when new evidence arrives, and — under the depth rule — used as evidence later, but only after the world confirms it.

---

## The side effect that matters more

Hallucination stops being a failure mode of generation and becomes a **labeled state**.

GPT cannot say which of its tokens were interpolation.  
Here, interpolation cannot appear except as a **marked slot** — there is nowhere else to write.

Error is allowed. What is not allowed is not knowing that it was a guess.

---

## Who writes

The mind does not write. **`Tape.decide`** writes (WRITE / CONFIRM / DISPUTE / DERIVE).  
The mind proposes; the tape commits under rules. That keeps facts and guesses out of weights.

---

## What already exists (no new store)

| Piece | Role |
|---|---|
| 279 WRITE / CONFIRM / DISPUTE + support counts | write path and reliability |
| 288 repair against \(T^*\) | break / restore; forgery detection |
| leak test | conclusions die when premises die |
| slot index answers | content is not sampled from weights |

A derived slot is an ordinary slot with **depth > 0** and provenance  
`(verb ∈ closed set, premises = slot ids)`.

Nothing new is needed as a memory engine. The tape is enough.

---

## Hard rules (else it is philosophy)

### 1. Guess that repeats an existing value → CONFIRM, not a new slot

Otherwise the tape inflates with retellings of itself, and support counts — free reliability — stop meaning anything.

### 2. Depth must reach the exit, not only the index

An answer served from a slot with depth > 0 is marked as a **guess to the asker**.  
A mark that lives only inside is bookkeeping, not honesty.

### 3. Refutation must cost the mechanism

The world refutes a derived slot — deleting the slot is not enough.  
The **pattern that produced it** (verb + premise shape / pathway) loses support.  
Otherwise it will derive the same error forever. That loss is the selection loop; without it — flicker, not evolution.

### 4. Two gates, or it stays a story

**(a) Predictive value.** A tape *with* guesses must predict incoming text better than the same tape *without* them — selection against what the mind cannot edit (\(T^*\) / world stream).

**(b) No self-inflation.** The share of guesses must not grow without bound as the tape grows — a direct collapse / retelling check.

---

## Depth / evidence rule

1. **Only depth-0 slots count as evidence** for further derivation.  
2. **Only confirmation by incoming text** (the world) resets depth to 0.  
3. A thought becomes knowledge by passing through the world, never through itself.

---

## How a step looks

```text
mind proposes a candidate
        ↓
Tape.decide:
  same value exists? → CONFIRM (bump support)
  else             → DERIVE slot {value, verb, premises, depth>0}
        ↓
answer reads slot; if depth>0 → marked guess to the asker
        ↓
world CONFIRMs → depth:=0, pattern support↑
world refutes  → slot gone + pattern support↓
```

Execute the verb over slot indices (deterministic).  
Do not decode free text from a continuous blend.

Optional later: a smooth **mind workspace** (289b) may *suggest* candidates. It must not *speak*. Suggestion → `Tape.decide` → read.

---

## What this deliberately is not

- Not chaotic interpolation of hidden states as the answer channel.  
- Not an evolving lexicon of new logic types from custom thought-vectors.  
- Not “risk of drift = 0” or accuracy 1.00.  
- Not generate-and-filter against the tape alone (that saturates deductive closure).  
- Not mind-authored writes that bypass CONFIRM when the value already exists.

Verbs are **installed** (closed set). Content can be **new** as a derived slot.  
Selection hits **patterns**, not only rows.

---

## Why this is “almost already built”

279 can WRITE/CONFIRM/DISPUTE. 288 can stress and repair. Leak already treats dependency.  
Missing pieces are mostly **policy + gates**: DERIVE vs CONFIRM, depth on the answer wire, pattern-level support loss, gates (a)(b).

289b only asks: is there a smooth situation space worth interpolating *before* `Tape.decide`?

---

## Invariant (three lines)

> The mind holds no facts.  
> Whatever is thought must be written (by `Tape.decide`).  
> A written thought is not evidence until the world confirms — and until then the asker must see it as a guess.
