"""
prepare_dataset.py
Selecciona las clases de yoga, divide en train/test (80/20) con estratificación,
y copia las imágenes en carpetas organizadas.
"""
import os
import shutil
import random
from collections import defaultdict

random.seed(42)

# Clases seleccionadas: nombre sánscrito → nombre en español
CLASSES = {
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SRC = os.path.join(BASE, "dataset")
TRAIN_DIR = os.path.join(BASE, "data", "train")
TEST_DIR = os.path.join(BASE, "data", "test")
SPLIT_RATIO = 0.80  # 80% train, 20% test


def prepare():
    # Limpiar directorios previos
    for d in [TRAIN_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    stats = defaultdict(dict)
    total_train = 0
    total_test = 0

    for sanskrit_name, spanish_name in CLASSES.items():
        src_dir = os.path.join(DATASET_SRC, sanskrit_name)
        if not os.path.isdir(src_dir):
            print(f"  [WARN] Carpeta no encontrada: {src_dir}")
            continue

        # Obtener todas las imágenes
        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Crear carpetas y copiar
        train_class_dir = os.path.join(TRAIN_DIR, sanskrit_name)
        test_class_dir = os.path.join(TEST_DIR, sanskrit_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(os.path.join(src_dir, img), train_class_dir)
        for img in test_imgs:
            shutil.copy2(os.path.join(src_dir, img), test_class_dir)

        stats[sanskrit_name] = {
            "spanish": spanish_name,
            "total": len(images),
            "train": len(train_imgs),
            "test": len(test_imgs),
        }
        total_train += len(train_imgs)
        total_test += len(test_imgs)

    # Resumen
    print("=" * 70)
    print("PREPARACIÓN DEL DATASET - Poses de Yoga")
    print("=" * 70)
    print(f"{'Clase (sánscrito)':<30} {'Español':<18} {'Total':>6} {'Train':>6} {'Test':>6}")
    print("-" * 70)
    for name, s in sorted(stats.items()):
        print(f"{name:<30} {s['spanish']:<18} {s['total']:>6} {s['train']:>6} {s['test']:>6}")
    print("-" * 70)
    print(f"{'TOTAL':<50} {total_train + total_test:>6} {total_train:>6} {total_test:>6}")
    print(f"\nSplit ratio: {SPLIT_RATIO:.0%} train / {1-SPLIT_RATIO:.0%} test")
    print(f"Clases: {len(stats)}")
    print(f"Train: {TRAIN_DIR}")
    print(f"Test:  {TEST_DIR}")


if __name__ == "__main__":
    prepare()
