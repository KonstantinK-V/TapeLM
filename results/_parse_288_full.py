import re, json
from pathlib import Path
t = Path("results/_stage288_full.out").read_text(encoding="utf-8", errors="replace")
for name in ["CONTROL", "HELD"]:
    m = re.search(rf"  {name} ({{.*}})", t)
    if not m:
        print(name, "missing")
        continue
    d = json.loads(m.group(1))
    det, rep = d["detection"], d["repair_true_flag"]
    print("====", name)
    print(" forged", round(det["model_forged"], 3), "votes", round(det["votes_forged"], 3), "rand", round(det["random_floor"], 3))
    print(" dup", round(det["model_dup"], 3), "votes_dup", round(det["votes_dup"], 3))
    print(" clean", round(det["model_clean_pass"], 3), "auc", round(det["clean_margin_auc"], 3), "z", round(det["clean_margin_auc_z"], 2))
    print(" repair_rew", round(rep["model_reward"], 3), "acc", round(rep["model_accuracy"], 3), "votes_rew", round(rep["votes_reward"], 3))
    print(" unk_auc", round(rep["unknown_margin_auc"], 3), "z", round(rep["unknown_margin_auc_z"], 2),
          "unrec", rep["n_unrecoverable"], "rec", rep["n_recoverable"])
    print(" observer", round(d["observer_verdict_restored"]["model"], 3))
idx = t.rfind('"overall"')
print(t[idx - 30 : idx + 600])
