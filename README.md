# Evaluacion de un modelo de clasificacion de poses de yoga

Modelo multiclase que clasifica imagenes de 12 poses de yoga utilizando Azure Custom Vision. Se evalua el rendimiento del modelo mediante metricas de Precision, Recall, F1-score, curvas ROC y AUC, y se aplica una iteracion de mejora con Advanced Training para comparar resultados. La infraestructura se aprovisiona con Terraform y todo el flujo se automatiza con scripts de Python.

## Dataset

El dataset original proviene de Kaggle: [Yoga Pose Image Classification Dataset](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset) (107 clases, ~6000 imagenes). Para este proyecto se seleccionaron 12 clases con al menos 54 imagenes cada una, divididas en train (80%) y test (20%).

## Estructura del proyecto

```
graficos/                   (generados por evaluate_model.py e improve_and_compare.py)
  confusion_matrix.png
  metrics_bars.png
  roc_curves.png
  comparison_f1.png
iac/
  main.tf                   Resource Group + Custom Vision Training + Prediction
  variables.tf              Nombres y region configurables
  outputs.tf                Endpoints y claves como outputs
scripts/
  .env.example              Plantilla de credenciales
  requirements.txt          Dependencias Python
  prepare_dataset.py        Selecciona clases y divide train/test
  train_model.py            Crea proyecto, sube imagenes, Quick Training
  publish_model.py          Publica el endpoint de prediccion
  evaluate_model.py         Predicciones test + metricas + graficos
  improve_and_compare.py    Advanced Training + comparacion v1 vs v2
```

## Clases seleccionadas

| Nombre sanscrito          | Nombre en espanol   | Total | Train | Test |
|---------------------------|---------------------|-------|-------|------|
| adho mukha svanasana      | Perro boca abajo    | 69    | 55    | 14   |
| balasana                  | Nino                | 71    | 56    | 15   |
| bhujangasana              | Cobra               | 73    | 58    | 15   |
| natarajasana              | Danzarin            | 72    | 57    | 15   |
| padmasana                 | Loto                | 68    | 54    | 14   |
| phalakasana               | Plancha             | 57    | 45    | 12   |
| ustrasana                 | Camello             | 87    | 69    | 18   |
| utkatasana                | Silla               | 73    | 58    | 15   |
| virabhadrasana i          | Guerrero I          | 54    | 43    | 11   |
| virabhadrasana ii         | Guerrero II         | 55    | 44    | 11   |
| virabhadrasana iii        | Guerrero III        | 61    | 48    | 13   |
| vriksasana                | Arbol               | 62    | 49    | 13   |
| **Total**                 |                     | **802** | **636** | **166** |

## Resultados

### Modelo v1 (Quick Training)

- Accuracy: 84.34%
- AUC macro-promedio: 0.9918
- Clase con peor rendimiento: vriksasana (Arbol) con F1 = 0.14

### Modelo v2 (Advanced Training, 1 hora)

- Accuracy: 93.98% (+9.64 puntos)
- Mejora mas notable: vriksasana paso de F1 = 0.14 a F1 = 0.96

## Prerequisitos

- Azure CLI: `brew install azure-cli`
- Terraform: `brew install terraform`
- Python 3.9 o superior
- Cuenta de Azure con suscripcion activa

## Uso

### 1. Autenticarse en Azure

```bash
az login
```

### 2. Desplegar infraestructura

```bash
cd iac/
terraform init
terraform plan
terraform apply -auto-approve
terraform output -json
```

### 3. Configurar credenciales

```bash
cd ../scripts/
cp .env.example .env
# Editar .env con los valores de terraform output
```

### 4. Ejecutar pipeline

```bash
pip install -r requirements.txt
python prepare_dataset.py
python train_model.py
python publish_model.py
python evaluate_model.py
python improve_and_compare.py
```

### 5. Limpiar recursos

```bash
cd ../iac/
terraform destroy -auto-approve
```

## Graficos generados

El script `evaluate_model.py` genera automaticamente:

- `confusion_matrix.png` — Matriz de confusion sobre el conjunto de test
- `metrics_bars.png` — Precision, Recall y F1-score por clase
- `roc_curves.png` — Curvas ROC (one-vs-rest) con AUC por clase

El script `improve_and_compare.py` genera:

- `comparison_f1.png` — Comparacion de F1-score entre Quick Training y Advanced Training

## Referencias

- Microsoft. Custom Vision documentation. https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/
- HashiCorp. Terraform AzureRM Provider. https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs
- Scikit-learn. Classification metrics. https://scikit-learn.org/stable/modules/model_evaluation.html
- Saxena, S. Yoga Pose Image Classification Dataset. https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset
