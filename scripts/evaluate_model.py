"""
evaluate_model.py
Ejecuta predicciones sobre el conjunto de test, calcula métricas detalladas
(Precision, Recall, F1, ROC/AUC) y genera gráficos PNG para el informe.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv

load_dotenv()

PREDICTION_ENDPOINT = os.getenv("PREDICTION_ENDPOINT")
PREDICTION_KEY = os.getenv("PREDICTION_KEY")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE, "data", "test")
OUTPUT_DIR = os.path.join(BASE, "graficos")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Nombres en español para los gráficos
SPANISH_NAMES = {
    "adho mukha svanasana": "Perro boca abajo",
    "vriksasana": "Árbol",
    "virabhadrasana i": "Guerrero I",
    "virabhadrasana ii": "Guerrero II",
    "virabhadrasana iii": "Guerrero III",
    "phalakasana": "Plancha",
    "utkatasana": "Silla",
    "bhujangasana": "Cobra",
    "natarajasana": "Danzarín",
    "ustrasana": "Camello",
    "balasana": "Niño",
    "padmasana": "Loto",
}


def load_project_info():
    info_path = os.path.join(SCRIPTS_DIR, "project_info.txt")
    info = {}
    with open(info_path) as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            info[key] = val
    return info


def predict_test_set(predictor, project_id, iteration_name):
    """Ejecuta predicciones sobre todas las imágenes de test."""
    class_dirs = sorted([
        d for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ])
    classes = class_dirs
    results = []

    for class_name in class_dirs:
        class_path = os.path.join(TEST_DIR, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            with open(img_path, "rb") as f:
                try:
                    prediction = predictor.classify_image(
                        project_id, iteration_name, f.read()
                    )
                except Exception as e:
                    print(f"  Error prediciendo {img_name}: {e}")
                    time.sleep(1)
                    continue

            # Extraer probabilidades por clase
            probs = {p.tag_name: p.probability for p in prediction.predictions}
            predicted_class = max(probs, key=probs.get)
            confidence = probs[predicted_class]

            results.append({
                "image": img_name,
                "true_label": class_name,
                "predicted_label": predicted_class,
                "confidence": confidence,
                **{f"prob_{c}": probs.get(c, 0.0) for c in classes},
            })

            # Rate limiting
            time.sleep(0.15)

        print(f"  {class_name}: {len(images)} imágenes procesadas")

    return pd.DataFrame(results), classes


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Genera y guarda la matriz de confusión como heatmap."""
    labels_es = [SPANISH_NAMES.get(c, c) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_es, yticklabels=labels_es,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Valor real", fontsize=12)
    ax.set_title("Matriz de confusión", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matriz de confusión guardada: {output_path}")


def plot_metrics_bars(report_df, output_path):
    """Genera gráfico de barras con Precision, Recall y F1 por clase."""
    labels_es = [SPANISH_NAMES.get(c, c) for c in report_df.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels_es))
    width = 0.25

    bars1 = ax.bar(x - width, report_df["precision"], width, label="Precision", color="#3266ad")
    bars2 = ax.bar(x, report_df["recall"], width, label="Recall", color="#1d9e75")
    bars3 = ax.bar(x + width, report_df["f1-score"], width, label="F1-score", color="#d85a30")

    ax.set_ylabel("Valor", fontsize=12)
    ax.set_title("Precision, Recall y F1-score por clase", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_es, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Agregar valores sobre las barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico de métricas guardado: {output_path}")


def plot_roc_curves(y_true_bin, y_prob, classes, output_path):
    """Genera curvas ROC (one-vs-rest) para cada clase con AUC."""
    labels_es = [SPANISH_NAMES.get(c, c) for c in classes]
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    auc_scores = {}
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = roc_auc
        label_es = labels_es[i]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label_es} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de falsos positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)", fontsize=12)
    ax.set_title("Curvas ROC por clase (one-vs-rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curvas ROC guardadas: {output_path}")
    return auc_scores


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    info = load_project_info()
    project_id = info["PROJECT_ID"]

    credentials = ApiKeyCredentials(
        in_headers={"Prediction-key": PREDICTION_KEY}
    )
    predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, credentials)

    # Publicar el modelo si no está publicado
    # (asumimos que ya fue publicado por publish_model.py o manualmente)
    iteration_name = "yoga-classifier-v1"

    print("=" * 60)
    print("EVALUACIÓN DEL MODELO - Conjunto de test")
    print("=" * 60)

    # Predicciones
    print("\nEjecutando predicciones sobre el conjunto de test...")
    df, classes = predict_test_set(predictor, project_id, iteration_name)

    # Guardar resultados crudos
    csv_path = os.path.join(OUTPUT_DIR, "predicciones_test.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados: {csv_path}")

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values

    # 1. Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    report_df = pd.DataFrame(report).T
    class_report = report_df.loc[classes]
    print(classification_report(y_true, y_pred, labels=classes, digits=4))

    # Accuracy global
    accuracy = (y_true == y_pred).mean()
    print(f"Accuracy global: {accuracy:.4f}")

    # 2. Matriz de confusión
    print("\nGenerando gráficos...")
    plot_confusion_matrix(
        y_true, y_pred, classes,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    # 3. Barras de métricas
    plot_metrics_bars(
        class_report,
        os.path.join(OUTPUT_DIR, "metrics_bars.png")
    )

    # 4. Curvas ROC
    y_true_bin = label_binarize(y_true, classes=classes)
    prob_cols = [f"prob_{c}" for c in classes]
    y_prob = df[prob_cols].values

    auc_scores = plot_roc_curves(
        y_true_bin, y_prob, classes,
        os.path.join(OUTPUT_DIR, "roc_curves.png")
    )

    # 5. Resumen AUC
    print(f"\n{'='*60}")
    print("AUC POR CLASE")
    print(f"{'='*60}")
    for class_name in classes:
        label_es = SPANISH_NAMES.get(class_name, class_name)
        print(f"  {label_es:<20} AUC = {auc_scores[class_name]:.4f}")
    macro_auc = np.mean(list(auc_scores.values()))
    print(f"\n  AUC macro-promedio: {macro_auc:.4f}")

    # 6. Guardar resumen JSON
    summary = {
        "accuracy": float(accuracy),
        "macro_auc": float(macro_auc),
        "per_class": {},
    }
    for c in classes:
        summary["per_class"][c] = {
            "precision": float(class_report.loc[c, "precision"]),
            "recall": float(class_report.loc[c, "recall"]),
            "f1": float(class_report.loc[c, "f1-score"]),
            "auc": float(auc_scores[c]),
            "support": int(class_report.loc[c, "support"]),
        }

    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResumen guardado: {summary_path}")


if __name__ == "__main__":
    evaluate()
