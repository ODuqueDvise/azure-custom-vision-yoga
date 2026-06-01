"""
improve_and_compare.py
Analiza los errores del modelo, aplica mejoras (Advanced Training),
reentrena, evalúa de nuevo y compara métricas antes/después.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from dotenv import load_dotenv

load_dotenv()

TRAINING_ENDPOINT = os.getenv("TRAINING_ENDPOINT")
TRAINING_KEY = os.getenv("TRAINING_KEY")
PREDICTION_ENDPOINT = os.getenv("PREDICTION_ENDPOINT")
PREDICTION_KEY = os.getenv("PREDICTION_KEY")
PREDICTION_RESOURCE_ID = os.getenv("PREDICTION_RESOURCE_ID")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE, "data", "test")
OUTPUT_DIR = os.path.join(BASE, "graficos")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

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

PUBLISH_NAME_V2 = "yoga-classifier-v2"


def load_project_info():
    info_path = os.path.join(SCRIPTS_DIR, "project_info.txt")
    info = {}
    with open(info_path) as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            info[key] = val
    return info


def analyze_errors():
    """Analiza las predicciones erróneas del modelo v1."""
    csv_path = os.path.join(OUTPUT_DIR, "predicciones_test.csv")
    df = pd.read_csv(csv_path)

    errors = df[df["true_label"] != df["predicted_label"]]
    total = len(df)
    n_errors = len(errors)

    print(f"\n{'='*60}")
    print("ANÁLISIS DE ERRORES - Modelo v1")
    print(f"{'='*60}")
    print(f"Total predicciones: {total}")
    print(f"Errores: {n_errors} ({n_errors/total:.1%})")

    if n_errors > 0:
        print(f"\nConfusiones más frecuentes:")
        confusion_pairs = errors.groupby(
            ["true_label", "predicted_label"]
        ).size().sort_values(ascending=False)
        for (true, pred), count in confusion_pairs.head(10).items():
            true_es = SPANISH_NAMES.get(true, true)
            pred_es = SPANISH_NAMES.get(pred, pred)
            print(f"  {true_es} → {pred_es}: {count} errores")

        # Clases con peor rendimiento
        print(f"\nClases con más errores:")
        class_errors = errors.groupby("true_label").size().sort_values(ascending=False)
        for class_name, count in class_errors.items():
            class_total = len(df[df["true_label"] == class_name])
            label_es = SPANISH_NAMES.get(class_name, class_name)
            print(f"  {label_es}: {count}/{class_total} "
                  f"({count/class_total:.0%} error)")

    return errors


def retrain_advanced(trainer, project_id):
    """Reentrena el modelo con Advanced Training (mayor duración)."""
    print(f"\nIniciando Advanced Training (mayor presupuesto de tiempo)...")

    # Advanced Training con 1 hora de presupuesto
    iteration = trainer.train_project(
        project_id,
        training_type="Advanced",
        reserved_budget_in_hours=1,
    )

    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project_id, iteration.id)
        print(f"  Estado: {iteration.status}")
        if iteration.status == "Failed":
            print("  ERROR: El reentrenamiento falló.")
            return None
        time.sleep(30)

    print(f"  Reentrenamiento completado: {iteration.id}")
    return iteration


def evaluate_v2(predictor, project_id, classes):
    """Evalúa el modelo v2 sobre el test set."""
    results = []
    for class_name in classes:
        class_path = os.path.join(TEST_DIR, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            with open(img_path, "rb") as f:
                try:
                    prediction = predictor.classify_image(
                        project_id, PUBLISH_NAME_V2, f.read()
                    )
                except Exception as e:
                    print(f"  Error: {e}")
                    time.sleep(1)
                    continue

            probs = {p.tag_name: p.probability for p in prediction.predictions}
            predicted_class = max(probs, key=probs.get)

            results.append({
                "true_label": class_name,
                "predicted_label": predicted_class,
                **{f"prob_{c}": probs.get(c, 0.0) for c in classes},
            })
            time.sleep(0.15)

    return pd.DataFrame(results)


def plot_comparison(v1_metrics, v2_metrics, classes, output_path):
    """Genera gráfico comparativo de F1-score antes/después."""
    labels_es = [SPANISH_NAMES.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels_es))
    width = 0.35

    ax.bar(x - width/2, [v1_metrics[c]["f1"] for c in classes],
           width, label="v1 (Quick Training)", color="#3266ad", alpha=0.8)
    ax.bar(x + width/2, [v2_metrics[c]["f1"] for c in classes],
           width, label="v2 (Advanced Training)", color="#1d9e75", alpha=0.8)

    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title("Comparación F1-score: Quick Training vs Advanced Training",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_es, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparación guardada: {output_path}")


def improve():
    info = load_project_info()
    project_id = info["PROJECT_ID"]

    # 1. Analizar errores
    errors = analyze_errors()

    # 2. Reentrenar con Advanced Training
    t_creds = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, t_creds)

    iteration_v2 = retrain_advanced(trainer, project_id)
    if iteration_v2 is None:
        return

    # 3. Publicar v2
    print(f"\nPublicando modelo v2 como '{PUBLISH_NAME_V2}'...")
    trainer.publish_iteration(
        project_id, iteration_v2.id, PUBLISH_NAME_V2, PREDICTION_RESOURCE_ID
    )

    # 4. Métricas internas de v2
    perf_v2 = trainer.get_iteration_performance(project_id, iteration_v2.id)
    print(f"\nMétricas v2 (internas):")
    print(f"  Precision: {perf_v2.precision:.2%}")
    print(f"  Recall:    {perf_v2.recall:.2%}")

    # 5. Evaluar v2 sobre test set
    p_creds = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, p_creds)

    classes = sorted([
        d for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ])

    print("\nEvaluando modelo v2 sobre test set...")
    df_v2 = evaluate_v2(predictor, project_id, classes)

    # 6. Calcular métricas v2
    y_true = df_v2["true_label"].values
    y_pred = df_v2["predicted_label"].values
    report_v2 = classification_report(y_true, y_pred, labels=classes, output_dict=True)

    accuracy_v2 = (y_true == y_pred).mean()
    print(f"\nAccuracy v2: {accuracy_v2:.4f}")

    # 7. Cargar métricas v1
    summary_v1_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(summary_v1_path) as f:
        summary_v1 = json.load(f)

    v1_metrics = summary_v1["per_class"]
    v2_metrics = {}
    for c in classes:
        v2_metrics[c] = {
            "precision": report_v2[c]["precision"],
            "recall": report_v2[c]["recall"],
            "f1": report_v2[c]["f1-score"],
        }

    # 8. Comparación
    print(f"\n{'='*60}")
    print("COMPARACIÓN v1 vs v2")
    print(f"{'='*60}")
    print(f"{'Clase':<20} {'F1 v1':>8} {'F1 v2':>8} {'Δ':>8}")
    print("-" * 46)
    for c in classes:
        f1_v1 = v1_metrics[c]["f1"]
        f1_v2 = v2_metrics[c]["f1"]
        delta = f1_v2 - f1_v1
        label_es = SPANISH_NAMES.get(c, c)
        sign = "+" if delta >= 0 else ""
        print(f"  {label_es:<18} {f1_v1:>8.4f} {f1_v2:>8.4f} {sign}{delta:>7.4f}")

    acc_v1 = summary_v1["accuracy"]
    print(f"\n  Accuracy v1: {acc_v1:.4f}")
    print(f"  Accuracy v2: {accuracy_v2:.4f}")
    print(f"  Δ Accuracy:  {'+' if accuracy_v2 >= acc_v1 else ''}{accuracy_v2 - acc_v1:.4f}")

    # 9. Gráfico comparativo
    plot_comparison(
        v1_metrics, v2_metrics, classes,
        os.path.join(OUTPUT_DIR, "comparison_f1.png")
    )

    # 10. Guardar resumen v2
    summary_v2 = {
        "accuracy": float(accuracy_v2),
        "per_class": v2_metrics,
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_v2_summary.json"), "w") as f:
        json.dump(summary_v2, f, indent=2, ensure_ascii=False)

    print(f"\nProceso de mejora completado.")


if __name__ == "__main__":
    improve()
