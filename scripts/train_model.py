"""
train_model.py
Crea el proyecto en Custom Vision, sube las imágenes de entrenamiento,
lanza Quick Training e imprime las métricas de la iteración.
"""
import os
import time
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
)
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPTS_DIR, ".env"))

TRAINING_ENDPOINT = os.getenv("TRAINING_ENDPOINT")
TRAINING_KEY = os.getenv("TRAINING_KEY")

# Debug: verificar que se cargaron las credenciales
if not TRAINING_ENDPOINT or not TRAINING_KEY:
    print("ERROR: No se encontraron las credenciales.")
    print(f"  TRAINING_ENDPOINT = {TRAINING_ENDPOINT}")
    print(f"  TRAINING_KEY = {'***' + TRAINING_KEY[-4:] if TRAINING_KEY else 'None'}")
    print(f"  Buscando .env en: {os.path.join(SCRIPTS_DIR, '.env')}")
    exit(1)
else:
    print(f"Endpoint: {TRAINING_ENDPOINT}")
    print(f"Key: ***{TRAINING_KEY[-4:]}")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE, "data", "train")

PROJECT_NAME = "Yoga Pose Classification"
DOMAIN_NAME = "General (compact) [S1]"  # Compact domain para exportar si se necesita
BATCH_SIZE = 64  # Custom Vision acepta hasta 64 imágenes por lote


def train():
    credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)

    # Obtener dominio General [A2] (clasificación de imágenes)
    domains = trainer.get_domains()
    # Preferir General (compact) [S1] o General [A2]
    domain = None
    for d in domains:
        if d.name == "General (compact) [S1]" and d.type == "Classification":
            domain = d
            break
    if domain is None:
        for d in domains:
            if "General" in d.name and d.type == "Classification":
                domain = d
                break

    print(f"Dominio seleccionado: {domain.name} ({domain.id})")

    # Crear proyecto
    print(f"\nCreando proyecto: {PROJECT_NAME}")
    project = trainer.create_project(
        PROJECT_NAME,
        description="Clasificación de poses de yoga - Actividad 5 VPC",
        domain_id=domain.id,
        classification_type="Multiclass",
    )
    print(f"  Proyecto creado: {project.id}")

    # Guardar project_id para los otros scripts
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_info.txt")
    with open(info_path, "w") as f:
        f.write(f"PROJECT_ID={project.id}\n")

    # Registrar etiquetas y subir imágenes
    class_dirs = sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])

    tags = {}
    for class_name in class_dirs:
        tag = trainer.create_tag(project.id, class_name)
        tags[class_name] = tag
        print(f"  Etiqueta creada: {class_name} ({tag.id})")

    # Subir imágenes por clase en lotes
    total_uploaded = 0
    for class_name in class_dirs:
        class_path = os.path.join(TRAIN_DIR, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        tag_id = tags[class_name].id

        # Subir en lotes de BATCH_SIZE
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i + BATCH_SIZE]
            entries = []
            for img_name in batch:
                img_path = os.path.join(class_path, img_name)
                with open(img_path, "rb") as f:
                    entries.append(ImageFileCreateEntry(
                        name=img_name,
                        contents=f.read(),
                        tag_ids=[tag_id],
                    ))

            result = trainer.create_images_from_files(
                project.id,
                ImageFileCreateBatch(images=entries),
            )

            ok = sum(1 for img in result.images if img.status == "OK")
            dup = sum(1 for img in result.images if img.status == "OKDuplicate")
            total_uploaded += ok
            print(f"  {class_name}: lote {i//BATCH_SIZE + 1} → "
                  f"{ok} OK, {dup} duplicadas")

    print(f"\nTotal imágenes subidas: {total_uploaded}")

    # Entrenar
    print("\nIniciando Quick Training...")
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        status = iteration.status
        print(f"  Estado: {status}")
        if status == "Failed":
            print("  ERROR: El entrenamiento falló.")
            return
        time.sleep(10)

    print(f"\nEntrenamiento completado: iteración {iteration.id}")

    # Métricas de la iteración
    perf = trainer.get_iteration_performance(project.id, iteration.id)
    print(f"\n{'='*60}")
    print(f"MÉTRICAS DEL MODELO (iteración: {iteration.id[:8]}...)")
    print(f"{'='*60}")
    print(f"  Precision global: {perf.precision:.2%}")
    print(f"  Recall global:    {perf.recall:.2%}")
    print(f"\n{'Clase':<30} {'Precision':>10} {'Recall':>10} {'AP':>10}")
    print("-" * 60)
    for tag_perf in perf.per_tag_performance:
        print(f"  {tag_perf.name:<28} {tag_perf.precision:>10.2%} "
              f"{tag_perf.recall:>10.2%} {tag_perf.average_precision:>10.2%}")

    # Guardar iteration_id
    with open(info_path, "a") as f:
        f.write(f"ITERATION_ID={iteration.id}\n")

    print(f"\nInfo guardada en: {info_path}")


if __name__ == "__main__":
    train()
