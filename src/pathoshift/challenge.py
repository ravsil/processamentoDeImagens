"""PathoShift-CRC Mini Challenge — pipeline completo da entrega final.

Subcomandos (python -m pathoshift.challenge <cmd> --data-root ... ):
  eda         análise inicial dos dados (classes, dimensões, cor, foco, exemplos)
  features    extrai atributos (cor + GLCM + LBP) para todas as variantes de pré-processamento
  experiment  treina baseline + ablações, seleciona na VALIDAÇÃO, avalia robustez, gera figuras
  predict     inferência de um modelo salvo sobre um manifesto -> CSV no contrato oficial
  all         eda + features + experiment
"""

from __future__ import annotations

import os

# Paralelismo é feito por processos; evita sobreinscrição de threads BLAS/OpenMP em cada processo.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from pathoshift.data.official import CLASSES, CLASS_TO_IDX, load_official_manifests, read_tile
from pathoshift.models.classical import HistopathologyFeatureExtractor
from pathoshift.preprocessing.perturbations import apply_manifest_perturbation, random_acquisition_shift
from pathoshift.preprocessing.pipeline import PreprocConfig, Preprocessor

SEED = 42
AUG_SEED = 1000      # seed base da aumentação de treino
PROXY_SEED = 2000    # seed base da validação perturbada "proxy" (seleção)
C_GRID = (1.0, 10.0)

# Variantes de pré-processamento avaliadas (ablação).
VARIANTS: Dict[str, PreprocConfig] = {
    "raw": PreprocConfig(),
    "reinhard": PreprocConfig(color="reinhard"),
    "macenko": PreprocConfig(color="macenko"),
    "gauss1": PreprocConfig(denoise="gaussian", denoise_param=1.0),
    "median3": PreprocConfig(denoise="median", denoise_param=3),
    "reinhard+gauss1": PreprocConfig(color="reinhard", denoise="gaussian", denoise_param=1.0),
    "macenko+gauss1": PreprocConfig(color="macenko", denoise="gaussian", denoise_param=1.0),
}
# Conjuntos de imagens para os quais os atributos são extraídos.
#   train / validation: imagens limpas
#   train_aug: 1 cópia de cada imagem de treino com mudança de aquisição aleatória
#   validation_proxy: validação com mudanças aleatórias (seed própria) -> usada na SELEÇÃO
#   robustness: perturbações oficiais de robustness.csv -> usado SOMENTE na avaliação final
FEATURE_SETS = ("train", "train_aug", "validation", "validation_proxy", "robustness")


# ============================================================ utilidades
def _manifest_dir(data_root: Path) -> Path:
    return data_root / "manifests"


def _out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_reinhard_target(data_root: Path, train: pd.DataFrame, per_class: int = 100) -> dict:
    """Estatísticas LAB de referência = média das estatísticas de 900 imagens de TREINO."""
    import cv2
    sample = train.groupby("label", group_keys=False).apply(
        lambda d: d.sample(n=min(per_class, len(d)), random_state=SEED))
    means, stds = [], []
    for p in sample.image_path:
        lab = cv2.cvtColor(read_tile(data_root, p), cv2.COLOR_RGB2LAB).astype(np.float32)
        means.append(lab.mean(axis=(0, 1)))
        stds.append(lab.std(axis=(0, 1)))
    return {"mean": np.mean(means, axis=0).tolist(), "std": np.mean(stds, axis=0).tolist()}


def system_info() -> dict:
    import sklearn, skimage, cv2
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scikit-image": skimage.__version__,
        "opencv": cv2.__version__,
        "gpu_used": False,
    }
    try:
        import psutil  # opcional
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


# ============================================================ extração de atributos
_W: dict = {}


def _init_worker(data_root: str, variants: Dict[str, PreprocConfig], reinhard_target: dict):
    import cv2
    cv2.setNumThreads(1)
    _W["root"] = data_root
    _W["pre"] = {k: Preprocessor(v, reinhard_target) for k, v in variants.items()}
    _W["fx"] = HistopathologyFeatureExtractor()


def _load_input(task: dict) -> np.ndarray:
    img = read_tile(_W["root"], task["image_path"])
    mode = task["mode"]
    if mode == "manifest":
        img = apply_manifest_perturbation(img, task["perturbation"], task["parameters"], int(task["seed"]))
    elif mode == "random":
        img = random_acquisition_shift(img, np.random.default_rng(int(task["seed"])))
    return img


def _worker(task: dict) -> Dict[str, np.ndarray]:
    img = _load_input(task)
    return {k: _W["fx"].extract_all(pre(img)) for k, pre in _W["pre"].items()}


def build_tasks(df: pd.DataFrame, mode: str, seed_base: int = 0) -> List[dict]:
    tasks = []
    for i, r in enumerate(df.itertuples(index=False)):
        t = {"image_path": r.image_path, "mode": mode}
        if mode == "manifest":
            t.update(perturbation=r.perturbation, parameters=r.parameters, seed=r.seed)
        elif mode == "random":
            t["seed"] = seed_base + i
        tasks.append(t)
    return tasks


def extract_features(tasks: List[dict], data_root: Path, variants: Dict[str, PreprocConfig],
                     reinhard_target: dict, n_jobs: int) -> Dict[str, np.ndarray]:
    from tqdm import tqdm
    out: Dict[str, list] = {k: [] for k in variants}
    with Pool(n_jobs, initializer=_init_worker, initargs=(str(data_root), variants, reinhard_target)) as pool:
        for res in tqdm(pool.imap(_worker, tasks, chunksize=16), total=len(tasks), ncols=80):
            for k, v in res.items():
                out[k].append(v)
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def split_tasks(m: Dict[str, pd.DataFrame]) -> Dict[str, List[dict]]:
    return {
        "train": build_tasks(m["train"], "clean"),
        "train_aug": build_tasks(m["train"], "random", AUG_SEED * 100000),
        "validation": build_tasks(m["validation"], "clean"),
        "validation_proxy": build_tasks(m["validation"], "random", PROXY_SEED * 100000),
        "robustness": build_tasks(m["robustness"], "manifest"),
    }


def cmd_features(args) -> None:
    data_root, out = Path(args.data_root), _out(Path(args.output) / "features")
    m = load_official_manifests(_manifest_dir(data_root))
    target_file = out / "reinhard_target.json"
    if not target_file.exists():
        target_file.write_text(json.dumps(compute_reinhard_target(data_root, m["train"]), indent=2))
    target = json.loads(target_file.read_text())
    timings = {}
    for name, tasks in split_tasks(m).items():
        if (out / f"{name}__raw.npy").exists() and not args.force:
            print(f"[features] {name}: cache encontrado")
            continue
        print(f"[features] {name}: {len(tasks)} imagens x {len(VARIANTS)} variantes")
        t0 = time.time()
        feats = extract_features(tasks, data_root, VARIANTS, target, args.jobs)
        timings[name] = time.time() - t0
        for k, arr in feats.items():
            np.save(out / f"{name}__{k}.npy", arr)
    if timings:
        (out / "timing_features.json").write_text(json.dumps(timings, indent=2))


def load_feats(out: Path, set_name: str, variant: str) -> np.ndarray:
    return np.load(out / "features" / f"{set_name}__{variant}.npy")


# ============================================================ modelo
def make_model(C: float, probability: bool):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=C, gamma="scale", probability=probability, random_state=SEED))


def macro_f1(y, p) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro"))


def full_metrics(y, p, prob: Optional[np.ndarray] = None) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(len(CLASSES)), zero_division=0)
    res = {
        "accuracy": float(accuracy_score(y, p)),
        "f1_macro": float(np.mean(f1)),
        "by_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                     for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(CLASSES))).tolist(),
    }
    if prob is not None:
        res["ece"] = expected_calibration_error(y, prob)
        res["mean_confidence"] = float(prob.max(1).mean())
    return res


def expected_calibration_error(y, prob, n_bins: int = 15) -> float:
    conf, pred = prob.max(1), prob.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(ece)


def write_predictions(path: Path, ids, prob: np.ndarray) -> None:
    pd.DataFrame({
        "image_id": list(ids),
        "predicted_label": [CLASSES[i] for i in prob.argmax(1)],
        "confidence": np.round(prob.max(1), 6),
    }).to_csv(path, index=False, encoding="utf-8")


# ============================================================ experimento
def cmd_experiment(args) -> None:
    data_root, out = Path(args.data_root), Path(args.output)
    res_dir, fig_dir = _out(out / "results"), _out(out / "figures")
    m = load_official_manifests(_manifest_dir(data_root))
    y_tr = m["train"].label.map(CLASS_TO_IDX).values
    y_va = m["validation"].label.map(CLASS_TO_IDX).values
    y_rb = m["robustness"].label.map(CLASS_TO_IDX).values

    # ---------- 1) Grade de configurações avaliada APENAS em validação (limpa + proxy)
    rows = []
    fitted = {}
    for variant in VARIANTS:
        for aug in (False, True):
            X = load_feats(out, "train", variant)
            y = y_tr
            if aug:
                X = np.vstack([X, load_feats(out, "train_aug", variant)])
                y = np.concatenate([y_tr, y_tr])
            Xva, Xpx = load_feats(out, "validation", variant), load_feats(out, "validation_proxy", variant)
            for C in C_GRID:
                t0 = time.time()
                model = make_model(C, probability=False).fit(X, y)
                ttrain = time.time() - t0
                f_va = macro_f1(y_va, model.predict(Xva))
                f_px = macro_f1(y_va, model.predict(Xpx))
                name = f"{variant}{'+aug' if aug else ''}"
                print(f"[grid] {name:24s} C={C:<4} F1 val={f_va:.4f} proxy={f_px:.4f} ({ttrain:.0f}s)")
                rows.append(dict(config=name, variant=variant, augmentation=aug, C=C,
                                 f1_validation=f_va, f1_validation_proxy=f_px,
                                 selection_score=(f_va + f_px) / 2, train_seconds=ttrain))
                fitted[(name, C)] = model
    grid = pd.DataFrame(rows)

    # Regra de seleção definida a priori:
    #   baseline  = atributos sobre imagem crua, sem aumentação; C escolhido por F1 da validação limpa.
    #   cada config = melhor C pelo selection_score (média F1 validação limpa e proxy).
    #   proposta  = config com maior selection_score (excluindo o baseline).
    base_row = grid[grid.config == "raw"].sort_values("f1_validation", ascending=False).iloc[0]
    best_per_cfg = grid.sort_values("selection_score", ascending=False).groupby("config", sort=False).head(1)
    prop_row = best_per_cfg[best_per_cfg.config != "raw"].iloc[0]
    selection = {
        "baseline": {"config": "raw", "variant": "raw", "augmentation": False, "C": float(base_row.C)},
        "proposal": {"config": prop_row.config, "variant": prop_row.variant,
                     "augmentation": bool(prop_row.augmentation), "C": float(prop_row.C)},
        "rule": "baseline: C por F1 validação; proposta: max mean(F1 validação, F1 validação-proxy)",
        "frozen_before_robustness": True,
    }
    (res_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print("[seleção]", json.dumps(selection, indent=1))

    # ---------- 2) Ablação na robustez (avaliação apenas; a seleção já está congelada acima)
    pert = m["robustness"].perturbation.values
    abl = []
    for _, r in best_per_cfg.iterrows():
        model = fitted[(r.config, r.C)]
        p_rb = model.predict(load_feats(out, "robustness", r.variant))
        d = dict(config=r.config, C=r.C, f1_validation=r.f1_validation,
                 f1_validation_proxy=r.f1_validation_proxy, f1_robustness=macro_f1(y_rb, p_rb))
        d["f1_drop"] = d["f1_validation"] - d["f1_robustness"]
        for k in sorted(set(pert)):
            d[f"f1_{k}"] = macro_f1(y_rb[pert == k], p_rb[pert == k])
        abl.append(d)
    ablation = pd.DataFrame(abl).sort_values("f1_validation_proxy", ascending=False)
    grid.to_csv(res_dir / "grid_validation.csv", index=False)
    ablation.to_csv(res_dir / "ablation.csv", index=False)

    # ---------- 3) Modelos finais (com probabilidades Platt) para baseline e proposta
    final = {}
    for role, sel in (("baseline", selection["baseline"]), ("proposal", selection["proposal"])):
        X = load_feats(out, "train", sel["variant"])
        y = y_tr
        if sel["augmentation"]:
            X = np.vstack([X, load_feats(out, "train_aug", sel["variant"])])
            y = np.concatenate([y_tr, y_tr])
        t0 = time.time()
        model = make_model(sel["C"], probability=True).fit(X, y)
        ttrain = time.time() - t0
        target = json.loads((out / "features" / "reinhard_target.json").read_text())
        bundle = {"model": model, "preproc": VARIANTS[sel["variant"]].to_dict(), "reinhard_target": target,
                  "classes": CLASSES, "selection": sel}
        _out(out / "models")
        joblib.dump(bundle, out / "models" / f"{role}.joblib", compress=3)
        pv = model.predict_proba(load_feats(out, "validation", sel["variant"]))
        pr = model.predict_proba(load_feats(out, "robustness", sel["variant"]))
        ppx = model.predict_proba(load_feats(out, "validation_proxy", sel["variant"]))
        final[role] = {"sel": sel, "pv": pv, "pr": pr, "ppx": ppx, "train_seconds": ttrain}
        pred_dir = _out(out / "predictions" / role)
        write_predictions(pred_dir / "predictions_validation.csv", m["validation"].image_id, pv)
        write_predictions(pred_dir / "predictions_robustness.csv", m["robustness"].image_id, pr)

    # CSVs oficiais = proposta selecionada
    write_predictions(Path(args.deliverable_dir) / "predictions_validation.csv", m["validation"].image_id,
                      final["proposal"]["pv"])
    write_predictions(Path(args.deliverable_dir) / "predictions_robustness.csv", m["robustness"].image_id,
                      final["proposal"]["pr"])

    # ---------- 4) Métricas, figuras e análise de erros
    summary = {"system": system_info(), "selection": selection, "seed": SEED, "roles": {}}
    for role, f in final.items():
        pv, pr = f["pv"].argmax(1), f["pr"].argmax(1)
        mv, mr = full_metrics(y_va, pv, f["pv"]), full_metrics(y_rb, pr, f["pr"])
        summary["roles"][role] = {
            "config": f["sel"]["config"], "C": f["sel"]["C"], "train_seconds": f["train_seconds"],
            "validation": mv, "robustness": mr,
            "validation_proxy_f1": macro_f1(y_va, f["ppx"].argmax(1)),
            "f1_drop_validation_to_robustness": mv["f1_macro"] - mr["f1_macro"],
            "robustness_by_perturbation": {k: full_metrics(y_rb[pert == k], pr[pert == k], f["pr"][pert == k])
                                           for k in sorted(set(pert))},
        }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, ablation, res_dir / "summary.md")

    from pathoshift import figures
    figures.make_result_figures(summary, ablation, final, m, data_root, fig_dir, y_va, y_rb)
    print((res_dir / "summary.md").read_text(encoding="utf-8"))


def write_markdown_summary(summary: dict, ablation: pd.DataFrame, path: Path) -> None:
    L = ["# Resumo dos resultados\n"]
    L.append("| Modelo | Config | F1-macro val | Acc val | F1-macro rob | Acc rob | Queda F1 | ECE val | ECE rob |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for role, r in summary["roles"].items():
        v, b = r["validation"], r["robustness"]
        L.append(f"| {role} | {r['config']} (C={r['C']:g}) | {v['f1_macro']:.4f} | {v['accuracy']:.4f} | "
                 f"{b['f1_macro']:.4f} | {b['accuracy']:.4f} | {r['f1_drop_validation_to_robustness']:.4f} | "
                 f"{v['ece']:.3f} | {b['ece']:.3f} |")
    L.append("\n## F1-macro por perturbação (robustness.csv)\n")
    perts = list(next(iter(summary["roles"].values()))["robustness_by_perturbation"])
    L.append("| Modelo | " + " | ".join(perts) + " |")
    L.append("|---|" + "---|" * len(perts))
    for role, r in summary["roles"].items():
        L.append(f"| {role} | " + " | ".join(f"{r['robustness_by_perturbation'][k]['f1_macro']:.4f}" for k in perts) + " |")
    L.append("\n## F1 por classe\n")
    L.append("| Classe | " + " | ".join(f"{role} val | {role} rob" for role in summary["roles"]) + " |")
    L.append("|---|" + "---|" * (2 * len(summary["roles"])))
    for c in CLASSES:
        L.append(f"| {c} | " + " | ".join(
            f"{r['validation']['by_class'][c]['f1']:.3f} | {r['robustness']['by_class'][c]['f1']:.3f}"
            for r in summary["roles"].values()) + " |")
    L.append("\n## Ablação (melhor C por configuração; seleção feita só com validação/proxy)\n")
    L.append(ablation.to_markdown(index=False, floatfmt=".4f"))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================ predição (inferência)
def cmd_predict(args) -> None:
    """Inferência reprodutível: modelo salvo + manifesto -> CSV no contrato oficial."""
    data_root = Path(args.data_root)
    bundle = joblib.load(args.model)
    m = load_official_manifests(_manifest_dir(data_root))
    split = args.split
    df = m[split]
    mode = "manifest" if split == "robustness" else "clean"
    cfg = PreprocConfig(**bundle["preproc"])
    t0 = time.time()
    feats = extract_features(build_tasks(df, mode), data_root, {"v": cfg}, bundle["reinhard_target"], args.jobs)["v"]
    prob = bundle["model"].predict_proba(feats)
    write_predictions(Path(args.out_csv), df.image_id, prob)
    print(f"[predict] {len(df)} imagens em {time.time() - t0:.1f}s ({args.jobs} processos) -> {args.out_csv}")


# ============================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["eda", "features", "experiment", "predict", "all"])
    ap.add_argument("--data-root", required=True, help="raiz do pacote de imagens (contém manifests/ e NCT-CRC-HE-100K-NONORM/)")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--deliverable-dir", default=".", help="onde gravar predictions_*.csv oficiais")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", help="(predict) caminho do .joblib")
    ap.add_argument("--split", choices=["validation", "robustness"], help="(predict) manifesto")
    ap.add_argument("--out-csv", help="(predict) CSV de saída")
    args = ap.parse_args(argv)
    np.random.seed(SEED)

    if args.command in ("eda", "all"):
        from pathoshift import figures
        figures.run_eda(Path(args.data_root), _out(Path(args.output) / "eda"), args.jobs)
    if args.command in ("features", "all"):
        cmd_features(args)
    if args.command in ("experiment", "all"):
        cmd_experiment(args)
    if args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
