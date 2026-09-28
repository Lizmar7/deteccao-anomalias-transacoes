"""
Pipeline completo de detecção de anomalias/fraude em transações.

Uso:
    python main.py

Saídas:
    results/metrics.json      -> métricas de todos os modelos
    results/threshold_sweep.csv -> análise de threshold do melhor modelo
    img/*.png                 -> gráficos (distribuição de classes, curvas
                                  ROC/PR, matriz de confusão, SHAP)
"""

import json
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data_utils import load_data, basic_report, preprocess
from src.train import train_all_models
from src.evaluate import evaluate_all, threshold_sweep
from src.explain import explain_random_forest

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "img"
RESULTS_DIR = ROOT / "results"
sns.set_theme(style="whitegrid")


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts().rename({0: "Normal", 1: "Fraude"})
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                      palette=["#4C72B0", "#C44E52"], legend=False)
    ax.set_yscale("log")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.title("Distribuição de classes (escala log)")
    plt.ylabel("Nº de transações")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "class_distribution.png", dpi=120)
    plt.close()


def plot_roc_pr_curves(results, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r.y_score)
        axes[0].plot(fpr, tpr, label=f"{r.name} (AUC={r.roc_auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, r.y_score)
        axes[1].plot(rec, prec, label=f"{r.name} (AP={r.pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("Falso Positivo Rate")
    axes[0].set_ylabel("Verdadeiro Positivo Rate")
    axes[0].set_title("Curva ROC")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall (mais informativa aqui)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(IMG_DIR / "roc_pr_curves.png", dpi=120)
    plt.close()


def plot_confusion_matrix(result, out_name: str) -> None:
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(result.confusion, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
    plt.title(f"Matriz de confusão — {result.name}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(IMG_DIR / out_name, dpi=120)
    plt.close()


def plot_threshold_sweep(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 5))
    plt.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    plt.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    plt.plot(df["threshold"], df["f1"], marker="o", label="F1-score")
    plt.xlabel("Threshold de decisão")
    plt.ylabel("Score")
    plt.title("Impacto do threshold em Precision / Recall / F1\n(melhor modelo supervisionado)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "threshold_sweep.png", dpi=120)
    plt.close()


def main():
    IMG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("1) Carregando dados...")
    df = load_data()
    report = basic_report(df)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    plot_class_distribution(df)

    print("\n2) Pré-processando (split estratificado + escala)...")
    split = preprocess(df)
    contamination = float(np.mean(split.y_train))

    print("\n3) Treinando modelos (Isolation Forest, LOF, Reg. Logística, "
          "Random Forest, Reg. Logística + SMOTE)...")
    models = train_all_models(split.X_train, split.y_train)

    print("\n4) Avaliando modelos com métricas para classes desbalanceadas...")
    results = evaluate_all(models, split.X_test, split.y_test, contamination)

    metrics_table = []
    for r in results:
        metrics_table.append({
            "modelo": r.name,
            "precision": round(r.precision, 4),
            "recall": round(r.recall, 4),
            "f1_score": round(r.f1, 4),
            "roc_auc": round(r.roc_auc, 4),
            "pr_auc": round(r.pr_auc, 4),
        })
        print(f"  - {r.name:38s} | Precision={r.precision:.3f} Recall={r.recall:.3f} "
              f"F1={r.f1:.3f} ROC-AUC={r.roc_auc:.3f} PR-AUC={r.pr_auc:.3f}")

    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_table, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_roc_pr_curves(results, split.y_test)

    # Melhor modelo supervisionado por PR-AUC (métrica mais robusta a desbalanceamento)
    supervised_results = [r for r, tm in zip(results, models) if tm.kind == "supervised"]
    best = max(supervised_results, key=lambda r: r.pr_auc)
    print(f"\nMelhor modelo (por PR-AUC): {best.name}")

    for r in results:
        safe_name = unicodedata.normalize("NFKD", r.name).encode("ascii", "ignore").decode("ascii")
        safe_name = safe_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(r, f"confusion_{safe_name}.png")

    print("\n5) Analisando o efeito do threshold no melhor modelo...")
    sweep_rows = threshold_sweep(split.y_test, best.y_score)
    pd.DataFrame(sweep_rows).to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    plot_threshold_sweep(sweep_rows)

    print("\n6) Gerando explicabilidade (SHAP) para o Random Forest...")
    rf_tm = next(tm for tm in models if tm.name.startswith("Random Forest"))
    try:
        shap_path = explain_random_forest(rf_tm.model, split.X_test, split.feature_names, IMG_DIR)
        print(f"   SHAP summary salvo em: {shap_path}")
    except Exception as e:
        print(f"   [aviso] Não foi possível gerar o gráfico SHAP: {e}")

    print("\nPipeline concluído. Veja results/ e img/ para as saídas completas.")


if __name__ == "__main__":
    main()
