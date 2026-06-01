"""
publish_model.py
Publica la iteración entrenada como endpoint de predicción.
Debe ejecutarse después de train_model.py y antes de evaluate_model.py.
"""
import os
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv

load_dotenv()

TRAINING_ENDPOINT = os.getenv("TRAINING_ENDPOINT")
TRAINING_KEY = os.getenv("TRAINING_KEY")
PREDICTION_RESOURCE_ID = os.getenv("PREDICTION_RESOURCE_ID")

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLISH_NAME = "yoga-classifier-v1"


def publish():
    info_path = os.path.join(SCRIPTS_DIR, "project_info.txt")
    info = {}
    with open(info_path) as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            info[key] = val

    project_id = info["PROJECT_ID"]
    iteration_id = info["ITERATION_ID"]

    credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)

    print(f"Publicando iteración {iteration_id[:8]}... como '{PUBLISH_NAME}'")
    trainer.publish_iteration(
        project_id, iteration_id, PUBLISH_NAME, PREDICTION_RESOURCE_ID
    )
    print(f"  Modelo publicado exitosamente.")
    print(f"  Nombre de publicación: {PUBLISH_NAME}")
    print(f"  Project ID: {project_id}")


if __name__ == "__main__":
    publish()
