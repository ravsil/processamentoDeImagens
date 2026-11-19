"""Figuras da EDA, dos resultados e da análise de erros (matplotlib, para o relatório)."""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pathoshift.data.official import CLASSES, load_official_manifests, read_tile
from pathoshift.preprocessing.perturbations import apply_manifest_perturbation
from pathoshift.preprocessing.pipeline import PreprocConfig, Preprocessor
from pathoshift.preprocessing.quality import compute_focus_measure, compute_tissue_mask

# Paleta categórica (ordem fixa): baseline, proposta, terceira série
C_BASE, C_PROP, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ============================================================ EDA
_ROOT = {}


def _init(root):
    cv2.setNumThreads(1)
    _ROOT["r"] = root


def _stats(rel):
    p = Path(_ROOT["r"]) / rel
    with Image.open(p) as im:
        size, mode = im.size, im.mode
        img = np.asarray(im.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, tissue = compute_tissue_mask(img)
    return dict(width=size[0], height=size[1], mode=mode,
                mean_r=img[..., 0].mean(), mean_g=img[..., 1].mean(), mean_b=img[..., 2].mean(),
                luminance=gray.mean(), contrast=gray.std(), saturation=hsv[..., 1].mean(),
                laplacian_var=compute_focus_measure(img), tissue_ratio=tissue,
                file_kb=p.stat().st_size / 1024)


def run_eda(data_root: Path, out: Path, jobs: int) -> None:
    m = load_official_manifests(data_root / "manifests")
    df = pd.concat([m["train"], m["validation"]], ignore_index=True)
    if (out / "image_stats.csv").exists():
        st = pd.read_csv(out / "image_stats.csv")
    else:
        with Pool(jobs, initializer=_init, initargs=(str(data_root),)) as pool:
            stats = pool.map(_stats, df.image_path.tolist(), chunksize=64)
        st = pd.concat([df[["image_id", "label", "split"]], pd.DataFrame(stats)], axis=1)
        st.to_csv(out / "image_stats.csv", index=False)

    summary = {
        "n_images": len(st),
        "class_counts": {s: st[st.split == s].label.value_counts().sort_index().to_dict() for s in ("train", "validation")},
        "sizes": {f"{w}x{h}": int(n) for (w, h), n in st.groupby(["width", "height"]).size().items()},
        "modes": st["mode"].value_counts().to_dict(),
        "robustness_perturbations": {f"{a} ({b})": int(n) for (a, b), n in
                                     m["robustness"].groupby(["perturbation", "parameters"]).size().items()},
        "per_class_median": st.groupby("label")[["luminance", "saturation", "contrast", "laplacian_var",
                                                  "tissue_ratio", "mean_r", "mean_g", "mean_b"]].median().round(2).to_dict(),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # 1) distribuição de classes (treino x validação)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    x = np.arange(len(CLASSES))
    for off, (split, col) in zip((-0.2, 0.2), (("train", C_BASE), ("validation", C_PROP))):
        cnt = st[st.split == split].label.value_counts().reindex(CLASSES).values
        ax.bar(x + off, cnt, 0.38, color=col, label=f"{split} (n={cnt.sum()})", edgecolor="white", linewidth=1)
    ax.set_xticks(x, CLASSES); ax.set_ylabel("imagens"); ax.legend(frameon=False, ncol=2)
    ax.set_title("Distribuição de classes por partição")
    fig.savefig(out / "class_distribution.png"); plt.close(fig)

    # 2) estatísticas de cor e qualidade por classe (treino)
    tr = st[st.split == "train"]
    cols = [("luminance", "Luminância média (0–255)"), ("saturation", "Saturação HSV média"),
            ("laplacian_var", "Foco: variância do Laplaciano"), ("tissue_ratio", "Fração de tecido")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, (c, t) in zip(axes, cols):
        data = [tr[tr.label == k][c].values for k in CLASSES]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for b in bp["boxes"]:
            b.set_facecolor("#c9dcf3"); b.set_edgecolor(C_BASE)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, 10), CLASSES, rotation=90); ax.set_title(t)
        if c == "laplacian_var":
            ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(out / "color_quality_by_class.png"); plt.close(fig)

    # 3) exemplos representativos (4 por classe, treino, seed fixa)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(4, 9, figsize=(11, 5.2))
    for j, c in enumerate(CLASSES):
        paths = m["train"][m["train"].label == c].image_path.values
        for i, p in enumerate(rng.choice(paths, 4, replace=False)):
            axes[i, j].imshow(read_tile(data_root, p)); axes[i, j].axis("off")
        axes[0, j].set_title(c)
    fig.suptitle("Exemplos de treino (4 por classe)", y=0.99)
    fig.tight_layout(); fig.savefig(out / "examples_per_class.png"); plt.close(fig)

    # 4) perturbações oficiais e efeito do pré-processamento
    rob = m["robustness"]
    picks = [rob[rob.perturbation == k].iloc[0] for k in sorted(rob.perturbation.unique())]
    pre = Preprocessor(PreprocConfig(color="reinhard", denoise="gaussian", denoise_param=1.0),
                       json.loads((out.parent / "features" / "reinhard_target.json").read_text())
                       if (out.parent / "features" / "reinhard_target.json").exists() else None)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.3))
    base = read_tile(data_root, picks[0].image_path)
    imgs = [("original", base)] + [(f"{r.perturbation}\n{r.parameters}",
                                     apply_manifest_perturbation(base, r.perturbation, r.parameters, int(r.seed)))
                                    for r in picks]
    for j, (t, im) in enumerate(imgs):
        axes[0, j].imshow(im); axes[0, j].set_title(t, fontsize=7); axes[0, j].axis("off")
        axes[1, j].imshow(pre(im)); axes[1, j].axis("off")
    axes[0, 0].text(-0.15, 0.5, "entrada", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.15, 0.5, "Reinhard+Gauss", transform=axes[1, 0].transAxes, rotation=90, va="center")
    fig.suptitle("Perturbações de robustness.csv (mesma imagem) e efeito do pré-processamento")
    fig.tight_layout(); fig.savefig(out / "perturbation_examples.png"); plt.close(fig)
    print(json.dumps(summary, indent=1, default=str)[:3000])


# ============================================================ resultados
def _cm_plot(ax, cm, title):
    cm = np.asarray(cm, float)
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(9), CLASSES, rotation=90, fontsize=7); ax.set_yticks(range(9), CLASSES, fontsize=7)
    ax.set_xlabel("predito"); ax.set_ylabel("verdadeiro"); ax.set_title(title, fontsize=9); ax.grid(False)


def make_result_figures(summary, ablation, final, m, data_root, out, y_va, y_rb):
    roles = summary["roles"]
    # 1) matrizes de confusão
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.2))
    for i, role in enumerate(("baseline", "proposal")):
        r = roles[role]
        _cm_plot(axes[i, 0], r["validation"]["confusion_matrix"],
                 f"{role} ({r['config']}) — validação\nF1-macro={r['validation']['f1_macro']:.3f}")
        _cm_plot(axes[i, 1], r["robustness"]["confusion_matrix"],
                 f"{role} ({r['config']}) — robustez\nF1-macro={r['robustness']['f1_macro']:.3f}")
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png"); plt.close(fig)

    # 2) F1-macro por condição (limpa + cada perturbação)
    perts = list(roles["baseline"]["robustness_by_perturbation"])
    conds = ["validação\n(limpa)"] + [p.replace("_", "\n") for p in perts]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    x = np.arange(len(conds))
    for off, role, col in ((-0.2, "baseline", C_BASE), (0.2, "proposal", C_PROP)):
        r = roles[role]
        vals = [r["validation"]["f1_macro"]] + [r["robustness_by_perturbation"][p]["f1_macro"] for p in perts]
        bars = ax.bar(x + off, vals, 0.38, color=col, label=f"{role}: {r['config']}", edgecolor="white", linewidth=1)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1, color=INK2)
    ax.set_xticks(x, conds); ax.set_ylim(0, 1.08); ax.set_ylabel("F1-macro")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(out / "f1_by_condition.png"); plt.close(fig)

    # 3) ablação
    ab = ablation.sort_values("f1_validation_proxy")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(ab) + 1))
    y = np.arange(len(ab))
    for off, col, c, lab in ((-0.27, C_BASE, "f1_validation", "validação limpa"),
                             (0, C_THIRD, "f1_validation_proxy", "validação-proxy (seleção)"),
                             (0.27, C_PROP, "f1_robustness", "robustness.csv (só avaliação)")):
        ax.barh(y + off, ab[c], 0.26, color=col, label=lab, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, ab.config); ax.set_xlim(0, 1); ax.set_xlabel("F1-macro")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.08), fontsize=7)
    fig.savefig(out / "ablation.png"); plt.close(fig)

    # 4) F1 por classe: validação vs robustez para as duas abordagens
    fig, ax = plt.subplots(figsize=(7, 2.7))
    x = np.arange(9)
    for off, role, col in ((-0.2, "baseline", C_BASE), (0.2, "proposal", C_PROP)):
        r = roles[role]
        v = [r["validation"]["by_class"][c]["f1"] for c in CLASSES]
        b = [r["robustness"]["by_class"][c]["f1"] for c in CLASSES]
        ax.bar(x + off, b, 0.38, color=col, label=f"{role} — robustez", edgecolor="white", linewidth=1)
        ax.scatter(x + off, v, marker="_", s=120, color=INK, linewidths=2, zorder=3,
                   label="F1 validação limpa" if role == "baseline" else None)
    ax.set_xticks(x, CLASSES); ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(out / "f1_per_class.png"); plt.close(fig)

    # 5) diagrama de confiabilidade (calibração) da proposta
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for prob, y, col, lab in ((final["proposal"]["pv"], y_va, C_BASE, "validação"),
                              (final["proposal"]["pr"], y_rb, C_PROP, "robustez")):
        conf, pred = prob.max(1), prob.argmax(1)
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (conf > lo) & (conf <= hi)
            if s.sum() >= 10:
                xs.append(conf[s].mean()); ys.append((pred[s] == y[s]).mean())
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=4, label=lab)
    ax.plot([0, 1], [0, 1], "--", color=INK2, lw=1)
    ax.set_xlabel("confiança média"); ax.set_ylabel("acurácia"); ax.legend(frameon=False)
    ax.set_title("Calibração — proposta")
    fig.savefig(out / "calibration.png"); plt.close(fig)

    # 6) galerias de acertos e erros (proposta, robustez) — imagens perturbadas reconstruídas
    rob = m["robustness"].reset_index(drop=True)
    prob = final["proposal"]["pr"]
    pred, conf = prob.argmax(1), prob.max(1)
    base_pred = final["baseline"]["pr"].argmax(1)
    wrong = np.where(pred != y_rb)[0]
    right = np.where(pred == y_rb)[0]
    rng = np.random.default_rng(0)
    sel_wrong = wrong[np.argsort(-conf[wrong])][:12]
    sel_right = rng.choice(right, 6, replace=False) if len(right) >= 6 else right
    fixed = np.where((pred == y_rb) & (base_pred != y_rb))[0]   # erros do baseline corrigidos pela proposta
    sel_fixed = rng.choice(fixed, min(6, len(fixed)), replace=False) if len(fixed) else fixed

    def gallery(idx, title, fname, ncol=6):
        if len(idx) == 0:
            return
        nrow = int(np.ceil(len(idx) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.75 * ncol, 2.35 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), idx):
            r = rob.iloc[i]
            im = apply_manifest_perturbation(read_tile(data_root, r.image_path), r.perturbation, r.parameters, int(r.seed))
            ax.imshow(im)
            ax.set_title(f"V:{r.label} P:{CLASSES[pred[i]]} ({conf[i]:.2f})\nB:{CLASSES[base_pred[i]]} | {r.perturbation}",
                         fontsize=6.5, color=INK)
        fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(out / fname); plt.close(fig)

    gallery(sel_wrong, "Erros mais confiantes da proposta (robustez). V=verdadeiro, P=proposta, B=baseline", "errors_robustness.png")
    gallery(sel_right, "Acertos da proposta (robustez, amostra aleatória)", "correct_robustness.png")
    gallery(sel_fixed, "Erros do baseline corrigidos pela proposta (robustez)", "fixed_by_proposal.png")

    # erros por (classe verdadeira -> predita) e perturbação, para a análise textual
    err = pd.DataFrame({"true": [CLASSES[i] for i in y_rb[wrong]], "pred": [CLASSES[i] for i in pred[wrong]],
                        "perturbation": rob.perturbation.values[wrong]})
    err.value_counts().head(25).to_csv(out.parent / "results" / "top_error_pairs_robustness.csv")
