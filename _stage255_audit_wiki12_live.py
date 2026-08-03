"""Read-only audit of wiki:12 recall metrics (does not touch running process)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

import _stage24x_lib as L
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tapelm_ext import DomainAdapter
from tokenizers import Tokenizer

RECALL_SEED = 255 + 9000
RUN = Path("results/stream255/wiki12")


def main() -> int:
    st = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
    tape_d = torch.load(RUN / "tape.pt", map_location="cpu", weights_only=False)
    K, V = tape_d["K"], tape_d["values"]
    Wq = DomainAdapter(256)
    Wq.load_state_dict(torch.load(RUN / "query_adapter.pt", map_location="cpu", weights_only=False))
    Wq.eval()

    device = torch.device("cpu")
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load("checkpoints/stage191_p1_curve.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)

    pf = st["probe_facts"]["wiki"]
    eval_f = [f for f in pf if not f["wq_train"]]
    train_f = [f for f in pf if f["wq_train"]]
    all_vals = list(dict.fromkeys([f["value"] for f in pf] + V))

    print(f"slots={len(V)} eval={len(eval_f)} train={len(train_f)} chunk={st['chunk_i']}")

    for label, W in [("frozen", None), ("W_q", Wq)]:
        m = L.tape_recall_metrics(eval_f, all_vals, bank, K, V, RECALL_SEED, W_bwd=W)
        print(f"  {label} eval held-out: {m}")

    # per-fact ranks (adapted)
    by_val: dict[str, list[int]] = {}
    for j, v in enumerate(V):
        by_val.setdefault(v, []).append(j)
    Kf = K.float()
    ranks_f, ranks_a = [], []
    for f in eval_f:
        for W, ranks in [(None, ranks_f), (Wq, ranks_a)]:
            q = L._tape_query(bank, f, W).detach().float()
            sc = Kf @ q
            gold = f["value"]
            gsc = float(sc[by_val[gold]].max())
            ranks.append(1 + int((sc > gsc).sum().item()))
    print(f"  frozen ranks: min={min(ranks_f)} max={max(ranks_f)} mean={sum(ranks_f)/len(ranks_f):.1f}")
    print(f"  W_q ranks:    min={min(ranks_a)} max={max(ranks_a)} unique={sorted(set(ranks_a))}")

    # train-fact leak check: score train facts with W_q (should be easy)
    m_train = L.tape_recall_metrics(train_f, all_vals, bank, K, V, RECALL_SEED, W_bwd=Wq)
    print(f"  W_q on TRAIN half (should not be in eval): {m_train}")

    # value collision: do held-out values appear many times on tape?
    for f in eval_f[:3]:
        n = len(by_val.get(f["value"], []))
        print(f"  value '{f['value']}' slots on tape: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
