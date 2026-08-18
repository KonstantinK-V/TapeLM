"""
Stage 279 — Writing is a decision, and the disagreements it finds are the product.

Every stage so far read a tape somebody else built. The write side was named as the open half in
GOAL.md and never implemented, and it is not an append:

    WRITE     the address is empty                     -> a new slot
    CONFIRM   same address, same value                  -> support += 1, nothing is stored
    DISPUTE   same address, a different value           -> a second slot, both kept

Three things follow that no separate mechanism has to provide. The tape stops growing on
repetition, because a repeat is a counter and not a slot. Witnesses stop being a template: a
value said by four sources and contradicted by two IS 4-2, arrived at from the data, which is
exactly the structure 271-278 had to manufacture with WITNESS_TMPL. And a support count is a
reliability estimate that costs nothing to compute.

That last one is the reason this stage is worth running before the harder open items. A tape
built this way is a contradiction detector with provenance attached, and it audits two things
that are otherwise awkward to audit:

    --source corpus   every address where the text disagrees with itself, and who said what
    --source model    the same subject sampled N times from an LM at temperature. The share of
                      samples that agree is the model's uncertainty measured BEHAVIOURALLY, from
                      what it says rather than from its logits, and the corpus majority is
                      available as a check on whether that number means anything.

The calibration gate is the one that matters. If self-consistency predicts agreement with the
corpus, then a tape written this way estimates reliability with no labels at inference time. If
it does not, the support count is decoration and this stage says so.

Matching is exact by default and optionally soft. Soft is where the ink does the one job 277
left it: an address is a coordinate it created at write time, so a value written as "Paris" and
a value written as "paris ." should land together. The soft arm is guarded by a replay control -
writing a corpus twice must produce zero disputes, whatever the matcher - because a matcher that
invents disagreements is worse than one that misses them.

  python _stage279_write_decision.py [--smoke]
  python _stage279_write_decision.py --soft-match 0.9
  python _stage279_write_decision.py --source model --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 279
LOG_PATH = RES / "_stage279_log.txt"
CUE = "The {S} was"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


# --------------------------------------------------------------------------- the write decision

WRITE, CONFIRM, DISPUTE = "WRITE", "CONFIRM", "DISPUTE"


class Tape:
    """Addresses to slots, where a slot is a value with a support count and its sources.

    The whole write side is `decide`. Everything downstream - deduplication, the witness
    structure, the confidence estimate - is a consequence of it rather than a separate feature.
    """

    def __init__(self, bank: FpBank | None = None, soft_tau: float = 0.0):
        self.slots: dict[str, dict[str, dict]] = defaultdict(dict)
        self.bank = bank
        self.soft_tau = soft_tau
        self.counts = Counter()
        self.journal: list[tuple[str, str, str, str]] = []   # address, value, action, matched

    def _match(self, address: str, value: str) -> str | None:
        """The value already on the tape that this one is the same as, or None."""
        have = self.slots[address]
        if value in have:
            return value
        if self.soft_tau <= 0.0 or self.bank is None or not have:
            return None
        # the ink's surviving job: a coordinate reproducible from a different surface form. It
        # PROPOSES a match; the address it proposes within was still fixed at write time, so a
        # mistake here can only merge two values of one subject, never reach another subject.
        keys = list(have)
        V = F.normalize(self.bank.fp([value] + keys).float(), dim=-1)
        sims = (V[1:] @ V[0]).tolist()
        j = int(np.argmax(sims))
        return keys[j] if sims[j] >= self.soft_tau else None

    def decide(self, address: str, value: str, source: str) -> str:
        hit = self._match(address, value)
        if not self.slots[address]:
            action = WRITE
        elif hit is not None:
            action = CONFIRM
        else:
            action = DISPUTE
        key = hit if hit is not None else value
        slot = self.slots[address].setdefault(key, {"support": 0, "sources": [], "forms": set()})
        slot["support"] += 1
        slot["forms"].add(value)
        if len(slot["sources"]) < 8:
            slot["sources"].append(source)
        self.counts[action] += 1
        self.journal.append((address, value, action, key))
        return action

    # ---- what the tape knows once it is written -------------------------------------------
    def n_slots(self) -> int:
        return sum(len(v) for v in self.slots.values())

    def disputed(self) -> list[str]:
        return [a for a, v in self.slots.items() if len(v) > 1]

    def verdict(self, address: str):
        """majority value and its consistency, or None where the tape never settled."""
        v = self.slots.get(address) or {}
        if not v:
            return None, 0.0, 0
        ranked = sorted(v.items(), key=lambda kv: -kv[1]["support"])
        total = sum(s["support"] for s in v.values())
        lead, second = ranked[0][1]["support"], (ranked[1][1]["support"] if len(ranked) > 1 else 0)
        if lead == second:
            return None, lead / total, total      # a tie is not a verdict - 278's rule, reused
        return ranked[0][0], lead / total, total


# --------------------------------------------------------------------------- assertion sources

VALUE_STOP = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "meanwhile", "later", "between", "however", "although", "during",
    "after", "before", "since", "while", "these", "those", "there", "their", "another", "several",
    "following", "according", "despite", "instead", "finally", "initially", "originally",
}
REL_RE = re.compile(r"[A-Za-z][a-z]{2,}")
# A hand-written list of English link words is exactly the brittleness the rest of this system
# avoids: it breaks on paraphrase, jargon, dialect and any other language. It is kept only as an
# ABLATION arm. What actually separates an assertion from an adjacency is structural and
# language-free - at least one token standing between the two names - and which of the surviving
# frames is a relation is then decided by 281's statistics, none of which read English.
LINK = {
    "was", "were", "is", "are", "became", "become", "named", "called", "known", "appointed",
    "elected", "born", "died", "founded", "located", "situated", "built", "released", "married",
    "served", "won", "wrote", "directed", "produced", "played", "signed", "joined", "replaced",
    "succeeded", "awarded", "seat", "capital", "member", "part", "author", "director",
}
# consecutive capitalised words are ONE name, not an anchor and a value
NP_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+(?=[A-Z]))?)+")


def sentence_initial(prefix: str) -> bool:
    """A capitalised word that opens a sentence is capitalised by grammar, not by being a name."""
    t = prefix.rstrip()
    return (not t) or t[-1] in ".!?\u201d\"" or t.endswith(" .")


def common_nouns(lines, min_lower: int = 3) -> set[str]:
    """Words the corpus also uses in lowercase are common nouns, whatever their capitalisation
    at the start of a phrase. Without this the anchors are cardinal, county, news and energy -
    titles and categories rather than subjects - and what gets addressed is the collocation of
    two words rather than a fact about somebody."""
    low = Counter()
    for ln in lines:
        for w in re.findall(r"\b[a-z]{3,}\b", ln):
            low[w] += 1
    return {w for w, c in low.items() if c >= min_lower}


def corpus_assertions(lines, rng, max_addr: int, min_mentions: int, mode: str, n_rel: int = 3,
                      common: set[str] | None = None, require_link: bool = False,
                      min_frame_words: int = 1):
    """An assertion is a natural (address, value) pair from real text, nothing authored.

    The first run addressed by the anchor word alone and the result was meaningless: "magazine"
    collected November, Constantine and Keats, and "chris" collected Mullin, Meanwhile and
    Turner. An anchor is not an address, because it carries no RELATION - every sentence in the
    corpus that happens to contain the word lands on it, and a dispute rate of 0.93 measures the
    corpus's vocabulary rather than its contradictions.

    The address is now the anchor together with the content words that stand between it and the
    value, which is what says WHAT is being asserted. "Chris Mullin" and "Chris ... Meanwhile"
    stop being the same address. --address anchor restores the old behaviour for comparison.

    Values are also filtered: a capitalised word that opens a sentence is capitalised by grammar,
    and months and discourse markers are not entities however the regex feels about them.
    """
    by_addr: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ln in lines:
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e.lower() in VALUE_STOP:
                continue
            prefix = ln[: m.start()]
            if sentence_initial(prefix):
                continue
            lo = max(0, m.start() - 140)
            # the anchor is a whole noun phrase, and it must not touch the value: adjacency is
            # one name split in two, which is what 96% of 281's frames turned out to be
            nps = [a for a in NP_RE.finditer(ln[lo: m.start()])
                   if a.group(0).strip() and a.group(0).strip() != e]
            if not nps:
                continue
            a = nps[-1]
            name = a.group(0).strip()
            head = name.split()[-1].lower()
            if common and head in common and len(name.split()) == 1:
                continue
            between = ln[lo + a.end(): m.start()]
            rel_all = [w.lower() for w in REL_RE.findall(between)
                       if w.lower() not in VALUE_STOP]
            if len(rel_all) < min_frame_words:
                continue                      # nothing between the names: one name cut in half
            if require_link and not any(w in LINK for w in rel_all):
                continue                      # ablation arm only, and it reads English
            rel = rel_all[-n_rel:]
            addr = name.lower() if mode == "anchor" else name.lower() + "|" + " ".join(rel)
            by_addr[addr].append((e, ln[lo: m.end() + 20]))
    addrs = [a for a, v in by_addr.items() if len(v) >= min_mentions]
    rng.shuffle(addrs)
    addrs = addrs[:max_addr]
    out = []
    for a in addrs:
        for i, (e, ctx) in enumerate(by_addr[a]):
            out.append({"address": a, "value": e, "source": f"wiki:{a}:{i}", "ctx": ctx})
    return out, addrs


def fp_addresses(assertions, bank, tau: float, min_overlap: int, min_mentions: int,
                 addr_key: str = "two"):
    """Address by the coordinate the ink writes, verified by the words the contexts share.

    A string address demands the same anchor and the same relation words in the same forms, so a
    fact stated twice in two phrasings lands on two addresses and never gets to CONFIRM. The key
    is what the ink is actually for: norm(fp(anchor) + ctx_fp(context)) is a coordinate that a
    different surface form still reproduces.

    277 also showed what happens if that coordinate is trusted alone - nearest_other sits at 0.91
    and unrelated words are one hop apart. So it PROPOSES and the discrete channel disposes: two
    assertions may share an address only if the key cosine clears tau AND their write contexts
    share at least min_overlap content words. Approximate where it proposes, exact where it
    decides.
    """
    keys, akeys, ckeys, words = [], [], [], []
    for a in assertions:
        anchor = a["address"].split("|")[0]
        c = bank.ctx_fp(a["ctx"], exclude=a["value"])
        k = bank.fp([anchor])[0]
        keys.append(F.normalize((k + c) if c is not None else k, dim=-1))
        akeys.append(F.normalize(k, dim=-1))
        ckeys.append(F.normalize(c, dim=-1) if c is not None else F.normalize(k, dim=-1))
        words.append({w.lower() for w in REL_RE.findall(a["ctx"]) if w.lower() not in VALUE_STOP}
                     - {a["value"].lower()})
    K = F.normalize(torch.stack(keys).float(), dim=-1)
    # Summing identity into situation is what crowds this space. 283 measured it on 936
    # addresses: the summed key puts 0.345 of them within tau of another, the two channels
    # scored by their minimum put 0.061 there, and bare anchors 0.018 - so the context was not
    # protecting the address, it was flattening it, because every English sentence has much the
    # same character statistics. Two mentions now have to agree on WHO and on WHAT WAS SAID
    # separately, which is the same conjunction the word overlap already enforces, applied to
    # the channel that proposes rather than only to the one that disposes.
    A = F.normalize(torch.stack(akeys).float(), dim=-1)
    C = F.normalize(torch.stack(ckeys).float(), dim=-1)

    # single-link over ALL members, not against one representative. Keying on the first member
    # made the grouping depend on corpus order and split one anchor four ways - fp13, fp14, fp16
    # and fp17 were all cardinal, and fp14 and fp17 held the same Priest/Deacon split.
    members: list[list[int]] = []
    assigned = [-1] * len(assertions)
    for i in range(len(assertions)):
        best, best_s = -1, tau
        for g, mem in enumerate(members):
            idx = torch.tensor(mem, device=K.device)
            sims = (torch.minimum(A[idx] @ A[i], C[idx] @ C[i]) if addr_key == "two"
                    else K[idx] @ K[i]).tolist()
            for j, sim in enumerate(sims):
                if sim >= best_s and len(words[i] & words[mem[j]]) >= min_overlap:
                    best, best_s = g, sim
        if best < 0:
            assigned[i] = len(members)
            members.append([i])
        else:
            assigned[i] = best
            members[best].append(i)

    groups: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(assigned):
        groups[g].append(i)
    out, addrs = [], []
    for g, members in groups.items():
        if len(members) < min_mentions:
            continue
        name = f"fp{g}:{assertions[members[0]]['address']}"
        addrs.append(name)
        for n, i in enumerate(members):
            a = dict(assertions[i])
            # the string the corpus itself wrote, kept beside the grouped address: grouping is
            # what 293 puts on trial, so its output may not also be the label. See ident_index.
            a["straddr"] = a["address"]
            a["address"] = name
            a["source"] = f"wiki:{name}:{n}"
            out.append(a)
    return out, addrs


def fp_addresses_anchor_block(assertions, bank, tau: float, min_overlap: int,
                              min_mentions: int, link: str = "or",
                              overlap_from: str = "ctx"):
    """Merge only inside one normalized anchor; never across names.

    283 showed the global two-channel tau trades foreign for capacity because cosine density
    grows with corpus N. Bare anchors barely crowd (0.018) while context flattens keys, so the
    hard cut belongs on the name: different anchors never share an address. Soft linking runs
    only among mentions of that name, where N is local and a fixed tau does not need retuning.

    link:
      or  — 283b: overlap OR two-channel (recovered capacity, but glued distinct relations → ties).
      and — 283c: overlap AND two-channel. Same numbers as write2, scoped to the block.
    overlap_from:
      ctx — content words of the whole write context (283b).
      rel — words from the address relation tail after '|', so born/died do not share an edge
            through ambient English.

    min_mentions is applied AFTER clustering. Callers should extract with min_mentions=1 so
    paraphrases that each appear once can still form an address once merged.
    """
    if not assertions:
        return [], []
    if link not in ("or", "and"):
        raise ValueError(f"unknown link={link!r}")
    if overlap_from not in ("ctx", "rel"):
        raise ValueError(f"unknown overlap_from={overlap_from!r}")

    akeys, ckeys, words, anchors = [], [], [], []
    for a in assertions:
        anchor = a["address"].split("|")[0]
        anchors.append(anchor)
        c = bank.ctx_fp(a["ctx"], exclude=a["value"])
        k = bank.fp([anchor])[0]
        akeys.append(F.normalize(k, dim=-1))
        ckeys.append(F.normalize(c, dim=-1) if c is not None else F.normalize(k, dim=-1))
        if overlap_from == "rel":
            rel = (a["address"].split("|", 1) + [""])[1]
            words.append({w.lower() for w in REL_RE.findall(rel) if w.lower() not in VALUE_STOP})
        else:
            words.append({w.lower() for w in REL_RE.findall(a["ctx"]) if w.lower() not in VALUE_STOP}
                         - {a["value"].lower()})
    A = F.normalize(torch.stack(akeys).float(), dim=-1)
    C = F.normalize(torch.stack(ckeys).float(), dim=-1)

    by_anchor: dict[str, list[int]] = defaultdict(list)
    for i, anc in enumerate(anchors):
        by_anchor[anc].append(i)

    parent = list(range(len(assertions)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for idxs in by_anchor.values():
        if len(idxs) < 2:
            continue
        idx_t = torch.tensor(idxs, device=A.device)
        anc_sims = A[idx_t] @ A[idx_t].T
        ctx_sims = C[idx_t] @ C[idx_t].T
        two = torch.minimum(anc_sims, ctx_sims)
        n = len(idxs)
        for a in range(n):
            for b in range(a + 1, n):
                i, j = idxs[a], idxs[b]
                ov = len(words[i] & words[j]) >= min_overlap
                ink = float(two[a, b]) >= tau
                linked = (ov or ink) if link == "or" else (ov and ink)
                if linked:
                    union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(assertions)):
        groups[find(i)].append(i)

    out, addrs = [], []
    g_out = 0
    for members in groups.values():
        if len(members) < min_mentions:
            continue
        name = f"fp{g_out}:{assertions[members[0]]['address']}"
        g_out += 1
        addrs.append(name)
        for n, i in enumerate(members):
            a = dict(assertions[i])
            # the string the corpus itself wrote, kept beside the grouped address: grouping is
            # what 293 puts on trial, so its output may not also be the label. See ident_index.
            a["straddr"] = a["address"]
            a["address"] = name
            a["source"] = f"wiki:{name}:{n}"
            out.append(a)
    return out, addrs


def model_assertions(ext, addrs, n_samples: int, temperature: float, max_new: int):
    """The same question asked N times at temperature. What varies is the model, not the tape."""
    out = []
    for a in addrs:
        prompt = CUE.format(S=a)
        enc = ext.tok(prompt, return_tensors="pt")
        enc = {k: v.to(ext.device) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            gen = ext.model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                     temperature=temperature, top_p=0.95,
                                     num_return_sequences=n_samples,
                                     pad_token_id=ext.tok.eos_token_id)
        for i in range(gen.size(0)):
            txt = ext.tok.decode(gen[i][n_in:], skip_special_tokens=True)
            m = ENT_RE.search(txt) or re.search(r"[A-Za-z][a-z]{4,}", txt)
            if m:
                out.append({"address": a, "value": m.group(0).strip(),
                            "source": f"lm:{a}:{i}", "ctx": txt[:120]})
    return out


def build(assertions, bank, soft_tau):
    t = Tape(bank, soft_tau)
    for a in assertions:
        t.decide(a["address"], a["value"], a["source"])
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--source", choices=("corpus", "model"), default="corpus")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--addresses", type=int, default=0)
    ap.add_argument("--min-mentions", type=int, default=3)
    ap.add_argument("--require-link", action="store_true",
                    help="ablation: demand an English link word between anchor and value. Off by "
                         "default - a word list does not survive paraphrase or dialect, and the "
                         "structural rule plus 281's statistics do the same job without one.")
    ap.add_argument("--min-frame-words", type=int, default=1,
                    help="tokens that must stand between the two names. 1 is the language-free "
                         "rule that kills the adjacency pairs; 0 restores them.")
    ap.add_argument("--no-entity-anchors", dest="entity_anchors", action="store_false",
                    help="keep common nouns as anchors (the first fp run's behaviour)")
    ap.add_argument("--address-tau", type=float, default=0.90,
                    help="fp mode: key cosine at or above which two assertions may share an "
                         "address. The ink proposes; the overlap below verifies.")
    ap.add_argument("--address-overlap", type=int, default=2,
                    help="fp mode: content words the two write contexts must share before the "
                         "ink's proposal is accepted")
    ap.add_argument("--address", choices=("anchor", "anchor_rel", "fp"), default="anchor_rel",
                    help="anchor alone is not an address - it carries no relation, so every "
                         "sentence containing the word lands on it. anchor_rel adds the content "
                         "words between the anchor and the value. anchor reproduces the first run.")
    ap.add_argument("--samples", type=int, default=8, help="model source: samples per address")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--soft-match", type=float, default=0.0, metavar="TAU",
                    help="let the ink propose that two surface forms are one value, at cosine "
                         ">= TAU within an address. 0 = exact string match only.")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_addr = args.addresses or (60 if args.smoke else 600)
    max_lines = 3000 if args.smoke else 25000

    log(f"Stage279 write decision start {datetime.now(timezone.utc).isoformat()} "
        f"device={device} source={args.source} soft_match={args.soft_match}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    p1 = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    p1.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    p1.eval()
    for p in p1.parameters():
        p.requires_grad_(False)
    bank = FpBank(p1, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(3_000_000 if args.smoke else 20_000_000)
    lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400][:max_lines]

    common = common_nouns(lines) if args.entity_anchors else None
    if common:
        log(f"  common-noun filter: {len(common)} words excluded as anchors")
    corpus, addrs = corpus_assertions(lines, rng, n_addr, args.min_mentions, args.address,
                                      common=common, require_link=args.require_link,
                                      min_frame_words=args.min_frame_words)
    if args.address == "fp":
        corpus, addrs = fp_addresses(corpus, bank, args.address_tau, args.address_overlap,
                                     args.min_mentions)
        log(f"  fp addressing: tau={args.address_tau} overlap={args.address_overlap} "
            f"-> {len(addrs)} addresses, {len(corpus)} assertions")
    log(f"  corpus: {len(addrs)} addresses, {len(corpus)} assertions "
        f"({time.time()-t0:.0f}s)")
    if len(addrs) < 8:
        log("  not enough multi-mention addresses")
        return 1

    tape = build(corpus, bank, args.soft_match)
    n_assert = len(corpus)
    disputed = tape.disputed()
    supports = [s["support"] for v in tape.slots.values() for s in v.values()]
    corpus_stats = {
        "assertions": n_assert, "addresses": len(tape.slots), "slots": tape.n_slots(),
        "actions": dict(tape.counts),
        "dedup_rate": tape.counts[CONFIRM] / max(1, n_assert),
        "dispute_rate": len(disputed) / max(1, len(tape.slots)),
        "slots_saved_vs_append": 1.0 - tape.n_slots() / max(1, n_assert),
        "support_mean": float(np.mean(supports)) if supports else 0.0,
        "support_max": int(max(supports)) if supports else 0,
    }
    log("  corpus tape: " + json.dumps(corpus_stats))

    # ---- replay control: the same corpus written twice must never DISPUTE with itself -------
    replay = Tape(bank, args.soft_match)
    for a in corpus:
        replay.decide(a["address"], a["value"], a["source"])
    before = dict(replay.counts)
    for a in corpus:
        replay.decide(a["address"], a["value"], a["source"] + "#2")
    second = {k: replay.counts[k] - before.get(k, 0) for k in (WRITE, CONFIRM, DISPUTE)}
    log(f"  replay control (second identical pass): {json.dumps(second)}")

    # ---- paraphrase control: a value re-written in another surface form ---------------------
    def variants(v: str):
        return [v.lower(), v + " ,", " " + v]

    para = Tape(bank, args.soft_match)
    for a in corpus:
        para.decide(a["address"], a["value"], a["source"])
    p_before = dict(para.counts)
    prng = random.Random(SEED + 4)
    for a in corpus:
        para.decide(a["address"], prng.choice(variants(a["value"])), a["source"] + "~")
    p_new = {k: para.counts[k] - p_before.get(k, 0) for k in (WRITE, CONFIRM, DISPUTE)}
    para_merge = p_new[CONFIRM] / max(1, sum(p_new.values()))
    log(f"  paraphrase control: {json.dumps(p_new)} merged={para_merge:.3f}")

    # ---- the model audit, if asked ----------------------------------------------------------
    model_stats, calib = {}, {}
    if args.source == "model":
        import _stage262_trunk_swap as s262
        if os.name == "nt" and hasattr(s262, "ensure_short_hf_home"):
            s262.ensure_short_hf_home()
        ext = s262.ExternalTrunk(args.model, device)
        subset = addrs[: (12 if args.smoke else 120)]
        lm = model_assertions(ext, subset, args.samples, args.temperature, args.max_new)
        mt = build(lm, bank, args.soft_match)
        md = mt.disputed()
        model_stats = {
            "model": args.model, "samples_per_address": args.samples,
            "temperature": args.temperature, "assertions": len(lm),
            "addresses": len(mt.slots), "slots": mt.n_slots(),
            "actions": dict(mt.counts),
            "self_dispute_rate": len(md) / max(1, len(mt.slots)),
        }
        log("  model tape: " + json.dumps(model_stats))

        # the number this stage exists for: does behavioural consistency predict being right
        rows = []
        for a in mt.slots:
            mv, cons, tot = mt.verdict(a)
            gold, _, _ = tape.verdict(a)
            if gold is None or tot < 2:
                continue
            rows.append((cons, int(mv is not None and mv.lower() == gold.lower())))
        if len(rows) >= 8:
            rows.sort(key=lambda r: r[0])
            n3 = max(1, len(rows) // 3)
            lo = float(np.mean([c for _, c in rows[:n3]]))
            hi = float(np.mean([c for _, c in rows[-n3:]]))
            calib = {"n": len(rows), "acc_low_consistency": lo, "acc_high_consistency": hi,
                     "lift": hi - lo,
                     "consistency_mean": float(np.mean([c for c, _ in rows]))}
        else:
            calib = {"n": len(rows), "note": "too few addresses shared with the corpus tape"}
        log("  calibration: " + json.dumps(calib))
        del ext
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- gates -----------------------------------------------------------------------------
    g_no_false_dispute = second[DISPUTE] == 0 and second[WRITE] == 0
    g_dedup = corpus_stats["dedup_rate"] >= 0.10
    g_disputes = corpus_stats["dispute_rate"] >= 0.05
    g_compression = corpus_stats["slots_saved_vs_append"] > 0.0
    g_soft_merges = None if args.soft_match <= 0.0 else (para_merge >= 0.50)
    calib_tested = args.source == "model" and calib.get("n", 0) >= 8
    g_calibrated = (None if not calib_tested else calib.get("lift", -1.0) >= 0.15)

    if not g_no_false_dispute:
        overall = "WRITE_UNSOUND"          # the matcher invents disagreement; nothing else reads
    elif not (g_dedup and g_compression):
        overall = "NO_DEDUP"
    elif g_calibrated is None and args.source == "model":
        overall = "SUPPORT_UNTESTED"        # too few shared addresses to say anything either way
    elif args.source == "model" and not g_calibrated:
        overall = "SUPPORT_NOT_CALIBRATED"  # the counter is decoration, and this says so
    elif g_disputes and (g_calibrated is not False):
        overall = "WRITE_DECISION_OK"
    else:
        overall = "WRITE_DECISION_PARTIAL"

    examples = []
    for a in disputed[:12]:
        vs = sorted(tape.slots[a].items(), key=lambda kv: -kv[1]["support"])
        examples.append({"address": a,
                         "values": [{"value": v, "support": s["support"],
                                     "sources": s["sources"][:3]} for v, s in vs[:4]]})

    out = {
        "stage": 279, "overall": overall, "source": args.source,
        "soft_match": args.soft_match, "trained_parameters": 0,
        "min_mentions": args.min_mentions, "address_mode": args.address,
        "smoke": args.smoke, "seed": SEED,
        "gates": {
            "G_no_false_dispute_on_replay": g_no_false_dispute,
            "G_dedup_happens": g_dedup,
            "G_tape_compresses": g_compression,
            "G_disputes_found": g_disputes,
            "G_soft_match_merges_forms": g_soft_merges,
            "G_support_predicts_truth": g_calibrated,
        },
        "corpus_tape": corpus_stats,
        "replay_second_pass": second,
        "paraphrase_second_pass": {**p_new, "merged_fraction": para_merge},
        "model_tape": model_stats, "calibration": calib,
        "disputed_examples": examples,
        "note": (
            "The write side, which GOAL.md named as the open half. WRITE / CONFIRM / DISPUTE "
            "against an address, and the three consequences that follow without any further "
            "mechanism: the tape stops growing on repetition, the witness structure that "
            "271-278 had to manufacture with a template arrives from the data as a support "
            "count, and that count is a reliability estimate computed for free. The replay "
            "control is the validity gate - writing the same corpus twice must produce no "
            "disputes and no new slots, whatever the matcher, because a matcher that invents "
            "disagreement is worse than one that misses it. With --source model the same "
            "subject is sampled N times at temperature and the share of samples that agree is "
            "the model's uncertainty measured from what it says rather than from its logits; "
            "G_support_predicts_truth checks that number against the corpus majority, and if "
            "the lift is absent the support count is decoration and the verdict says so."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    tag = "" if args.source == "corpus" else "_model"
    (RES / f"stage279_decision{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RES / f"stage279_mini{tag}.md").write_text(
        f"# Stage 279 write as a decision ({args.source})\n\n**{overall}**"
        f"{' · SMOKE' if args.smoke else ''} · trained parameters **0**\n\n"
        f"- {corpus_stats['assertions']} assertions -> **{corpus_stats['slots']}** slots over "
        f"{corpus_stats['addresses']} addresses "
        f"({corpus_stats['slots_saved_vs_append']:.1%} saved against an append)\n"
        f"- actions {json.dumps(corpus_stats['actions'])}, dedup "
        f"**{corpus_stats['dedup_rate']:.3f}**, disputed addresses "
        f"**{corpus_stats['dispute_rate']:.3f}**\n"
        f"- replay control (identical second pass): {json.dumps(second)}\n"
        + (f"- model self-dispute **{model_stats.get('self_dispute_rate', float('nan')):.3f}**, "
           f"consistency lift **{calib.get('lift', float('nan')):.3f}** "
           f"(low {calib.get('acc_low_consistency', float('nan')):.3f} -> high "
           f"{calib.get('acc_high_consistency', float('nan')):.3f})\n" if model_stats else "")
        + "\n## Gates\n\n"
        + "".join(f"- {k}: **{v}**\n" for k, v in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
