# TapeLM — start here

You opened a research repo about an **unusual LM stack**: a frozen **character-curve encoder** plus **fingerprint memory** in the same representation space as generation. This folder is the curated surface — readable in a few minutes, reproducible when you have checkpoints.

The full stage tree (`_stage170` … `_stage212`, logs, legacy SOTE) lives at the **repository root**. You do not need to read all of it to understand the thesis.

---

## Path A — read only (no GPU, no checkpoints)

1. **[`OVERVIEW.md`](OVERVIEW.md)** — why the architecture is different from parametric LMs and text RAG; what worked; what we falsified.
2. **[`decisions/`](decisions/)** — machine-readable `stage*_decision.json` for the main scorecard stages (browse on GitHub).
3. **[`../results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md)** — full program narrative and numbers.
4. **[`../results/pre_publish_frontier.md`](../results/pre_publish_frontier.md)** — closed internalization frontier (210–212).

**Verdict table in the terminal:**

```bash
python artifact/scripts/show_map.py
```

---

## Path B — smoke the environment

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/check_env.py
```

---

## Path C — reproduce (GPU + checkpoints)

See **[`../docs/CHECKPOINTS.md`](../docs/CHECKPOINTS.md)**. Minimum artifacts:

- `checkpoints/stage191_p1_curve.pt`
- `checkpoints/stage191_p2_gpt.pt`

From the **repository root**:

```bash
python artifact/scripts/run_demo.py
python artifact/scripts/run_stage.py 204
python artifact/scripts/run_stage.py 203
```

`run_demo.py` runs the TapeLM assemble scorecard (`_stage196_tapelm.py`). Stage numbers map to root scripts via `run_stage.py` (including `207_max` for wiki-scale variant B).

After new local runs, refresh curated JSON:

```bash
python artifact/scripts/sync_decisions.py
```

---

## Folder layout

```
artifact/
  README.md              ← you are here
  OVERVIEW.md            thesis + falsification map
  requirements.txt
  decisions/             curated verdict JSON (+ optional mini.md)
  scripts/
    show_map.py          print verdict table
    check_env.py         imports and CUDA
    run_demo.py          Stage 196
    run_stage.py         run stage by number
    sync_decisions.py    copy key results → decisions/
```

---

## Cite & license

MIT — [`../LICENSE`](../LICENSE). Metadata for citation: [`../CITATION.cff`](../CITATION.cff).  
Suggested GitHub description and topic tags: [`../docs/PUBLISHING.md`](../docs/PUBLISHING.md).
