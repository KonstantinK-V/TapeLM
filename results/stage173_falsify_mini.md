# Stage173 — orthography vs language falsify

**Reading:** `LETTER_SEQUENCE_SENSITIVE`

- reading `LETTER_SEQUENCE_SENSITIVE`
- Real letter order helps beyond skeleton — still may be orthographic n-grams, not semantics.
- natural k1=0.848 | random_skel k1=0.666 | flat_skel k1=0.556
- shuffle_all k1=0.637 | destroy_spaces k1=0.642
- gaps={gap_shuffle_all=+0.210, gap_random_letters_skel=+0.182, gap_flat_skel=+0.291, gap_destroy_spaces=+0.206, gap_shuffle_letters=+0.189}
- LETTER_SEQUENCE_SENSITIVE ≠ language understanding. It only means char-order of real text affects Δ predictability under this pen.
