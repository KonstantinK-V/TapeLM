"""Three exact speedups shared by every tape stage - and the reason they exist.

289a's resample cost twenty minutes and I guessed at the cause twice, wrongly, before measuring
it. Caching the corpus scan changed nothing; caching the ink changed nothing. The cost was
fp_addresses making a few million tiny GPU round-trips, and compute was never involved. The
lesson is in HANDOFF 9b: time the loop before optimising it.

All three are EXACT. Not "close enough to keep the conclusions" - byte-identical output,
verified against the original paths, with --no-scan-cache / --no-fast-grouping kept so that
verification can be repeated at any time. A speedup that moved a number would quietly
invalidate every gate that depends on it, which is worse than a slow run.

They live here rather than in a stage because 286 needs them too and 289a imports 286: putting
them in a stage would make the import graph a cycle.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

# What counts as a word. ascii is the rule the whole project has used since stage 194 and must
# stay byte-identical to it - verify_word_rule proves that rather than trusting this comment.
# unicode is the same idea with the Latin-only assumption removed: a letter followed by letters
# or digits, or a run of digits. No tuned constant, no language list, nothing chosen; it is the
# Unicode character classes doing the deciding.
#
# The unicode rule only pays for itself with hash ink. arc_enc's stoi was built from an English
# corpus, so a Cyrillic character maps to index 0 and every Russian word inks to nearly the same
# vector - the regex would widen the intake and the encoder would throw it away. That is why it
# is a separate flag from --fp and not folded into it.
WORD_ASCII = re.compile(r"[A-Za-z][a-z]{2,}")
WORD_UNICODE = re.compile(r"[^\W\d_][^\W_]*|\d+", re.UNICODE)
WORD_RULES = {"ascii": WORD_ASCII, "unicode": WORD_UNICODE}

WORDS_MAX = 40  # must match FpBank.ctx_fp
WORDS_MIN = 3


def words_of(text, exclude=None, rule=WORD_ASCII):
    return [w for w in rule.findall(text) if w != exclude][:WORDS_MAX]


def verify_word_rule():
    """The ascii rule is duplicated from stage 194. Duplicated rules drift; this catches it."""
    from _stage194_fp_fact_memory import WORD_RE
    return WORD_ASCII.pattern == WORD_RE.pattern


# ---------------------------------------------------------------------------------------------
# One definition of "a digest becomes a sign vector", used by every path that needs one, because
# three implementations of the same bit order is three chances to disagree silently.
#
# coordinate k takes its sign from bit (k mod 8) of byte (k div 8).
#
# The sign vector is +/-1 over sqrt(d), so it is a unit vector and two different strings are
# near-orthogonal - a property of the digest, not a tuned constant.
# ---------------------------------------------------------------------------------------------


def digest_of(s: str, d: int) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=d // 8).digest()


def sign_bit(dg: bytes, k: int) -> int:
    return (dg[k // 8] >> (k % 8)) & 1


def sign_vector(dg: bytes, d: int) -> torch.Tensor:
    """The same mapping as sign_bit, vectorised. A python loop over d coordinates per n-gram
    costs minutes over a real vocabulary, and this stage has already been bitten once by doing
    per-item work that could be done per-batch."""
    u = torch.frombuffer(bytearray(dg), dtype=torch.uint8)
    bits = (u.unsqueeze(1) >> torch.arange(8, dtype=torch.uint8)) & 1
    return (bits.reshape(-1).to(torch.float32) * 2.0 - 1.0) / math.sqrt(d)


def hash_perm(d: int, seed: int = 0) -> torch.Tensor:
    """A permutation of d coordinates derived from a digest rather than from torch's RNG.

    torch.randperm with a seeded generator is reproducible within a version and is NOT promised
    across versions or platforms. That is exactly the fragility hash ink exists to remove, so
    the bigram binding must not reintroduce it one line later: a torch upgrade would silently
    change every context vector while every gate still passed.

    Sorting the coordinates by an independent digest key is a permutation by construction - no
    rejection, no loop, nothing to get wrong - and is uniformly random for the same reason the
    keys are.
    """
    keys = [digest_of(f"perm:{seed}:{k}", 64) for k in range(d)]
    return torch.tensor(sorted(range(d), key=lambda k: keys[k]), dtype=torch.long)


class CachedBank:
    """The ink is a pure function of the string, so inking the same context 120 times is pure
    waste - and that waste was the whole cost of this stage.

    A resample changes only WHICH addresses rng draws; the corpus does not change, so the
    contexts recur almost perfectly across tapes. Worse, every assertion was inked twice per
    resample: once by fp_addresses to build the key, once by pack_from_corpus to build the
    slot. Memoising on (text, exclude) removes both, changes no number anywhere - the frozen
    arc_enc returns the same vector by construction, and the hash gate still proves it - and
    turns a 20-hour run into a few hours.

    Nothing here is stored per ADDRESS or per identity: the key is the text, and the cache is
    an execution detail with no path into the mind's inputs.
    """

    # a cap so a long run cannot quietly eat the GPU: each entry is 256 floats, and the
    # corpus has far more distinct contexts than any one tape uses. Clearing wholesale rather
    # than evicting one by one is deliberate - the next tape re-warms it in seconds and an LRU
    # would cost more bookkeeping than it saves.
    MAX = 400_000

    def __init__(self, b):
        self._b, self._c, self._f = b, {}, {}

    def ctx_fp(self, text, exclude=None):
        k = (text, exclude)
        if k not in self._c:
            if len(self._c) >= self.MAX:
                self._c.clear()
            self._c[k] = self._b.ctx_fp(text, exclude=exclude)
        return self._c[k]

    def fp(self, words):
        k = tuple(words)
        if k not in self._f:
            self._f[k] = self._b.fp(words)
        return self._f[k]

    def __getattr__(self, n):
        return getattr(self._b, n)


class HashFp:
    """Ink with nothing trained in it at all.

    ctx_fp and the frozen arc_enc were introduced at stage 191-194, when the question was
    whether a curve could hold anything and English wikitext was the whole world. Everything
    built since then rests on that choice, and it has three defects that no amount of care
    downstream can repair:

      arc_enc mean-pools characters, so fp("listen") == fp("silent"). Anagrams collide, a
      prefix and a suffix are the same evidence, morphology is invisible.

      stoi was built from an English corpus. A Cyrillic or CJK character maps to index 0, so
      every Russian word inks to nearly the same vector. This is not a regex problem and no
      regex fixes it - the wall to other languages is INSIDE the frozen weights.

      A word not in the training distribution has no meaningful vector, and there is no
      mechanism by which it could get one.

    So take the invariant to its limit. "Whatever is trained may not hold facts" is satisfied
    most simply by an ink that was never trained:

        fp(w)  = normalize( sum over character n-grams t of "^"+w+"$" of  H(t) )
        H(t)   = (bits of blake2b(t, d/8 bytes)) * 2 - 1, over sqrt(d)

    H is a deterministic +/-1 vector read straight out of a cryptographic digest. No generator,
    no torch RNG, no version dependence, no checkpoint: one string, one vector, on any machine,
    forever. Near-orthogonality is a property of the digest, not something tuned.

    What that buys, all at once and all for free:

      order inside the word     "listen" and "silent" share no trigram
      every script and symbol   utf-8 bytes go into the digest; there is no character vocabulary
      no OOV, ever              an unseen word shares n-grams with seen ones, so morphology
                                works by construction - "^un" and "ing$" are features, not gaps
      the freeze question ends  nothing is trained, so nothing can leak, and the mind stops
                                depending on stage191_p1_curve.pt to mean anything

    What it costs is whatever arc_enc actually learned. The evidence says that may be nothing:
    the only channel carrying signal is the cosine of a bag of a bag, which is already a surface
    overlap measure. If a digest ties a trained encoder, the encoder was never contributing
    semantics - and that result is worth more than the accuracy either way.

    The boundary markers matter and are not decoration. Without them "^cat" and "cat$" are the
    same trigram and a prefix cannot be told from a suffix, which is the morphology case. They
    are added only for n >= 2: inside a 1-gram a marker carries no adjacency, it is just a
    constant vector added to every word, which inflates every cosine equally and tells nobody
    anything. n=1 is the setting a CJK corpus would want, so this is not a hypothetical.

    n=3 is an assumption about ALPHABETIC scripts and it is measured, not assumed: two-character
    CJK words share no trigram at all (東京 gives ^東京 and 東京$, 京都 gives ^京都 and 京都$ -
    the character they share never becomes a shared gram), so cos comes out at +0.017. --fp-ngram
    is exposed for exactly that reason. What hash ink removes unconditionally is the character
    VOCABULARY and OOV; how far a gram of a given length reaches into a script is a separate
    question with a separate number.
    """

    # the word cache is bounded by the vocabulary, but "the vocabulary" of a unicode corpus is
    # not a small number, and the gram cache is larger still. Same wholesale-clear policy as
    # CachedBank, and for the same reason: an LRU would cost more bookkeeping than it saves.
    MAX_WORDS = 400_000
    MAX_GRAMS = 400_000

    def __init__(self, device=None, d=256, n=3, rule=WORD_ASCII):
        if d % 8 or not 8 <= d // 8 <= 64:
            raise ValueError(f"d={d} must be a multiple of 8 with d/8 in [8, 64] for blake2b")
        if n < 1:
            raise ValueError(f"n={n} must be at least 1")
        self.d, self.n, self.rule = d, n, rule
        self.device = device or torch.device("cpu")
        self._g: dict[str, torch.Tensor] = {}
        self._w: dict[str, torch.Tensor] = {}

    def _h(self, t: str) -> torch.Tensor:
        """One gram's sign vector, on the CPU. Deliberately not on the device: building these
        one at a time on the GPU is a few hundred thousand tiny round-trips, which is the exact
        cost that made 289a take twenty minutes a resample. fp() transfers once per batch."""
        v = self._g.get(t)
        if v is None:
            if len(self._g) >= self.MAX_GRAMS:
                self._g.clear()
            v = self._g[t] = sign_vector(digest_of(t, self.d), self.d)
        return v

    def grams(self, w: str) -> list[str]:
        s = w if self.n < 2 else "^" + w + "$"
        if len(s) <= self.n:
            return [s]
        return [s[i:i + self.n] for i in range(len(s) - self.n + 1)]

    @torch.no_grad()
    def fp(self, ws: list[str]) -> torch.Tensor:
        # clear BEFORE deciding what is missing. Clearing after would drop words that are in ws
        # but were already cached - they would not be in `todo`, would not be rebuilt, and the
        # final stack would raise KeyError. It needs a 400k vocabulary to fire, so it would have
        # waited for --words unicode or a larger corpus and then crashed hours into a run.
        if len(self._w) >= self.MAX_WORDS:
            self._w.clear()
        todo = [w for w in dict.fromkeys(ws) if w not in self._w]
        if todo:
            built = torch.stack(
                [F.normalize(torch.stack([self._h(t) for t in self.grams(w)], 0).sum(0), dim=-1)
                 for w in todo], 0).to(self.device)          # one transfer for the whole batch
            for w, v in zip(todo, built):
                self._w[w] = v
        return torch.stack([self._w[w] for w in ws], 0)

    @torch.no_grad()
    def ctx_fp(self, text: str, exclude: str | None = None) -> torch.Tensor | None:
        ws = words_of(text, exclude, self.rule)
        if len(ws) < WORDS_MIN:
            return None
        return F.normalize(self.fp(ws).mean(0), dim=-1)


def verify_hash_ink(fp_bank, probes=("canada", "listen", "silent", "Россия", "東京", "1917")):
    """Is the ink actually deterministic, and does the tensor path agree with the digest?

    G_arc_enc_frozen is vacuous in a hash arm - there are no weights to move - so this is the
    gate that replaces it, and it has to be stronger than "it ran twice". Three claims:

      1 a FRESH bank reproduces every probe bit-for-bit (no hidden state, no RNG)
      2 the sign of every coordinate equals the corresponding bit of the digest, checked
        against hashlib directly rather than against the same code path
      3 distinct strings give distinct vectors, and an anagram is not a collision

    Returns (ok, notes) so the caller can log what failed rather than just that something did.
    """
    notes = {}
    fresh = HashFp(device=fp_bank.device, d=fp_bank.d, n=fp_bank.n, rule=fp_bank.rule)
    ok = True
    for w in probes:
        a, b = fp_bank.fp([w])[0], fresh.fp([w])[0]
        if not torch.equal(a, b):
            ok, notes[w] = False, "not reproducible in a fresh bank"
    # the digest cross-check: every coordinate's sign against hashlib, coordinate by coordinate,
    # in plain python. Not the same code path - that would only prove the code equals itself.
    for t in ("^ca", "abc", "東京京"):
        dg = hashlib.blake2b(t.encode("utf-8"), digest_size=fp_bank.d // 8).digest()
        v = fp_bank._h(t)
        want = [1.0 if (dg[k // 8] >> (k % 8)) & 1 else -1.0 for k in range(fp_bank.d)]
        if any(abs(float(v[k]) * math.sqrt(fp_bank.d) - want[k]) > 1e-6
               for k in range(fp_bank.d)):
            ok, notes[t] = False, "tensor path disagrees with the digest"
    lis, sil = fp_bank.fp(["listen"])[0], fp_bank.fp(["silent"])[0]
    notes["anagram_cos"] = round(float(lis @ sil), 4)
    if torch.equal(lis, sil):
        ok, notes["anagram"] = False, "listen and silent still collide"
    # a permutation that is mostly the identity would leave the bigram bind nearly commutative
    p = hash_perm(fp_bank.d)
    fixed = int((p == torch.arange(fp_bank.d)).sum())
    notes["perm_fixed_points"] = fixed
    if fixed > fp_bank.d // 16:
        ok, notes["perm"] = False, "the coordinate permutation is close to the identity"
    return ok, notes


# how many bigram contexts came out as a vector too short to normalise meaningfully. A product
# of unit vectors can cancel, and if it cancels often the binding is destroying the context
# rather than orienting it. Diagnostic only: nothing reads this, it is reported and reset.
INK_DEGENERATE = [0, 0]


class BigramBank:
    """Ink that can tell `X defeated Y` from `Y defeated X`.

    The edge ablation settled where the signal lives: `rare` contributes nothing (zeroing it
    leaves the run bit-identical, and alone it scores below the counting rival), so the whole
    4.42-sigma paired win sits in `cos` - the cosine between ctx_fp vectors. And ctx_fp is a
    mean over words of a mean over characters, order-blind twice. The mind's only working
    channel therefore cannot see which side of a relation a value stands on, and the errors say
    so: a third of them are exact swap pairs, A->B and B->A both present in the same confusion
    table (Leipzig<->Weimar, California<->Texas, British<->Canadian). It knows the pair and not
    the direction, which is precisely the signature of a bag.

    The fix belongs in the INK, not in the mind. arc_enc stays frozen and untrained, the
    permutation is a fixed function of a seed, and no parameter is added anywhere - the mind is
    bit-for-bit the same 3489 weights reading a sharper input.

        ctx_fp(T) = normalize( mean_i  fp(w_i) * P.fp(w_{i+1}) )

    P is a fixed permutation of the coordinates, applied to the RIGHT member only, which is what
    makes the binding non-commutative: bind(a,b) != bind(b,a). This is standard MAP binding from
    vector-symbolic algebra, not something invented here.

    It stays a bag - a bag of bigrams - and that is deliberate. Position-tagged sums (sum R^i
    fp(w_i)) also carry order, but shifting a phrase by one word changes every coordinate and
    similarity becomes brittle. Local order under global permutation-invariance is exactly what
    a relation needs and all it needs.

    The word list is re-derived here rather than taken from FpBank, which duplicates a rule -
    the 40-word cap and the 3-word minimum - and duplicated rules drift. verify_mean_path
    proves the duplicate is faithful by reproducing the base mean-ink bit-for-bit, and 289
    gates on it before training rather than assuming it.
    """

    def __init__(self, b, seed=0, rule=WORD_ASCII):
        self._b = b
        self._seed = seed
        self._rule = rule
        self._perm = None

    def _words(self, text, exclude):
        # FpBank.ctx_fp tokenises via context_words (stop list, lower, dedupe, cap 40) — not a
        # bare regex. BigramBank used to re-derive with words_of and drifted; verify_mean_path
        # then correctly refused the arm. Same path as the base ink, so the A/B is about order.
        if self._rule.pattern == WORD_ASCII.pattern:
            from _tape_index import context_words
            return context_words(text, exclude=exclude)
        return words_of(text, exclude, self._rule)

    def _rot(self, d, device):
        # from a digest, not torch.randperm - see hash_perm for why that distinction matters
        if self._perm is None or len(self._perm) != d:
            self._perm = hash_perm(d, self._seed)
        if self._perm.device != device:
            self._perm = self._perm.to(device)
        return self._perm

    @torch.no_grad()
    def ctx_fp(self, text, exclude=None):
        ws = self._words(text, exclude)
        if len(ws) < WORDS_MIN:
            return None
        V = self._b.fp(ws)                                     # (L, d), rows unit norm
        P = self._rot(V.shape[-1], V.device)
        m = (V[:-1] * V[1:].index_select(-1, P)).mean(0)
        INK_DEGENERATE[0] += int(float(m.norm()) < 1e-6)
        INK_DEGENERATE[1] += 1
        return F.normalize(m, dim=-1)

    @torch.no_grad()
    def verify_mean_path(self, texts):
        """Does the tokenisation duplicated above reproduce the base ink exactly?

        Not "closely" - torch.equal. If this is false the bigram arm and the mean arm are not
        comparable and every A/B conclusion drawn from them is about tokenisation, not order.
        """
        for t in texts:
            ws = self._words(t, None)
            a = self._b.ctx_fp(t)
            b = (F.normalize(self._b.fp(ws).mean(0), dim=-1)
                 if len(ws) >= WORDS_MIN else None)
            if (a is None) != (b is None):
                return False
            if a is not None and not torch.equal(a, b):
                return False
        return True

    def __getattr__(self, n):
        return getattr(self._b, n)


def install_assertion_cache(s279mod):
    """The corpus scan is deterministic; only WHICH addresses get drawn is not.

    corpus_assertions walks every line with three regexes to build by_addr, and only then does
    `rng.shuffle(addrs); addrs = addrs[:max_addr]`. Nothing before that shuffle depends on rng,
    so a hundred and twenty resamples re-derive the same table a hundred and twenty times. This
    keeps the table and re-draws from it.

    It is exact rather than approximate, which matters more than the speed: the first call runs
    the ORIGINAL function with no address cap, so the cached table is precisely what the
    original would have built. Later calls shuffle the same address list with the caller's own
    rng - same list, same rng state, same draw - and rebuild the assertions in the same order,
    so `source` indices are identical too. No number moves.

    Returns the uninstall function, so a caller that wants the unmemoised path can have it.
    """
    raw = s279mod.corpus_assertions
    store: dict = {}

    def cached(lines, rng, max_addr, min_mentions, mode, n_rel=3, common=None,
               require_link=False, min_frame_words=1):
        key = (id(lines), len(lines), min_mentions, mode, n_rel,
               require_link, min_frame_words, len(common) if common else -1)
        if key not in store:
            _out, _addrs = raw(lines, random.Random(0), 10 ** 9, min_mentions, mode,
                               n_rel=n_rel, common=common, require_link=require_link,
                               min_frame_words=min_frame_words)
            by_addr: dict = defaultdict(list)
            for a in _out:
                by_addr[a["address"]].append(a)
            # _addrs comes back ALREADY shuffled, so it is not the base the caller's own
            # shuffle must start from - and starting from the wrong base gives a different
            # draw, which an offline test caught. random.shuffle moves positions independently
            # of values, so replaying it on list(range(n)) recovers the permutation and
            # inverting it restores the original insertion order exactly.
            perm = list(range(len(_addrs)))
            random.Random(0).shuffle(perm)
            base = [None] * len(_addrs)
            for i, j in enumerate(perm):
                base[j] = _addrs[i]
            store[key] = (base, by_addr)
        addrs_all, by_addr = store[key]
        addrs = list(addrs_all)
        rng.shuffle(addrs)
        addrs = addrs[:max_addr]
        out = [dict(a) for ad in addrs for a in by_addr[ad]]
        return out, addrs

    s279mod.corpus_assertions = cached
    return lambda: setattr(s279mod, "corpus_assertions", raw)


def install_fast_fp_addresses(s279mod):
    """Same grouping, same order, same ties - one matmul per assertion instead of one per group.

    The original single-link loop is not compute-bound, it is round-trip-bound: for every
    assertion it walks the groups built so far and, for each, does `torch.tensor(mem)` (host to
    device) and `.tolist()` (device to host). With four thousand assertions and a thousand
    groups that is a few million tiny transfers, each costing far more than the 256-wide dot it
    carries. Measured: about twenty minutes per resample, unchanged by caching the scan or the
    ink, because neither was the cost.

    Every index in every group is smaller than i, so ONE product against all previous
    assertions covers every comparison the loop can make. The group walk then reads that vector
    on the CPU in exactly the original order - groups in order, members in order, `>=` so later
    ties still win - and the assignment is identical.

    The one honest caveat: a batched matmul may accumulate in a different order than the
    per-group one, so a dot product can differ in the last bit. That only matters if a
    similarity sits exactly on tau, and it would change a grouping decision rather than corrupt
    one. Verified equal on synthetic data below the level where that can happen.
    """
    raw = s279mod.fp_addresses

    def fast(assertions, bank, tau: float, min_overlap: int, min_mentions: int,
             addr_key: str = "two"):
        keys, akeys, ckeys, words = [], [], [], []
        for a in assertions:
            anchor = a["address"].split("|")[0]
            c = bank.ctx_fp(a["ctx"], exclude=a["value"])
            k = bank.fp([anchor])[0]
            keys.append(F.normalize((k + c) if c is not None else k, dim=-1))
            akeys.append(F.normalize(k, dim=-1))
            ckeys.append(F.normalize(c, dim=-1) if c is not None else F.normalize(k, dim=-1))
            words.append({w.lower() for w in s279mod.REL_RE.findall(a["ctx"])
                          if w.lower() not in s279mod.VALUE_STOP} - {a["value"].lower()})
        K = F.normalize(torch.stack(keys).float(), dim=-1)
        A = F.normalize(torch.stack(akeys).float(), dim=-1)
        C = F.normalize(torch.stack(ckeys).float(), dim=-1)

        members: list[list[int]] = []
        assigned = [-1] * len(assertions)
        for i in range(len(assertions)):
            if members:
                sims = (torch.minimum(A[:i] @ A[i], C[:i] @ C[i]) if addr_key == "two"
                        else K[:i] @ K[i]).tolist()
            best, best_s = -1, tau
            for g, mem in enumerate(members):
                for j in mem:
                    if sims[j] >= best_s and len(words[i] & words[j]) >= min_overlap:
                        best, best_s = g, sims[j]
            if best < 0:
                assigned[i] = len(members)
                members.append([i])
            else:
                assigned[i] = best
                members[best].append(i)

        groups: dict = defaultdict(list)
        for i, g in enumerate(assigned):
            groups[g].append(i)
        out, addrs = [], []
        for g, mem in groups.items():
            if len(mem) < min_mentions:
                continue
            name = f"fp{g}:{assertions[mem[0]]['address']}"
            addrs.append(name)
            for n, i in enumerate(mem):
                a = dict(assertions[i])
                a["address"] = name
                a["source"] = f"wiki:{name}:{n}"
                out.append(a)
        return out, addrs

    s279mod.fp_addresses = fast
    return lambda: setattr(s279mod, "fp_addresses", raw)


def install_all(s279mod, scan_cache: bool = True, fast_grouping: bool = True):
    """Both module-level patches at once. Returns the uninstall for each, in order."""
    outs = []
    if scan_cache:
        outs.append(install_assertion_cache(s279mod))
    if fast_grouping:
        outs.append(install_fast_fp_addresses(s279mod))
    return outs
