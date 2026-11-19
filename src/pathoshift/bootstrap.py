"""IC 95% (bootstrap, 2000 reamostragens) do F1-macro e da diferença proposta - baseline (pareado)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def main(data_root="../dados", out="outputs"):
    out = Path(out)
    rng = np.random.default_rng(42)
    res = {}
    for split in ("validation", "robustness"):
        y = pd.read_csv(Path(data_root) / "manifests" / f"{split}.csv", dtype=str).set_index("image_id").label
        b = pd.read_csv(out / "predictions" / "baseline" / f"predictions_{split}.csv").set_index("image_id").predicted_label
        p = pd.read_csv(out / "predictions" / "proposal" / f"predictions_{split}.csv").set_index("image_id").predicted_label
        y, b, p = y.values, b.reindex(y.index).values, p.reindex(y.index).values
        fb, fp, d = [], [], []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            x1, x2 = f1_score(y[i], b[i], average="macro"), f1_score(y[i], p[i], average="macro")
            fb.append(x1); fp.append(x2); d.append(x2 - x1)
        ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
        res[split] = {"baseline_ci95": ci(fb), "proposal_ci95": ci(fp), "diff_ci95": ci(d)}
    ab = pd.read_csv(out / "results" / "ablation.csv")
    res["spearman_proxy_vs_robustness"] = round(float(ab.f1_validation_proxy.rank().corr(ab.f1_robustness.rank())), 3)
    res["spearman_clean_vs_robustness"] = round(float(ab.f1_validation.rank().corr(ab.f1_robustness.rank())), 3)
    (out / "results" / "bootstrap_ci.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
