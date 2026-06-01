# Evaluacion de un modelo de clasificacion de poses de yoga

Modelo multiclase que clasifica imagenes de 12 poses de yoga utilizando Azure Custom Vision. Se evalua el rendimiento del modelo mediante metricas de Precision, Recall, F1-score, curvas ROC y AUC. La infraestructura se aprovisiona con Terraform y todo el flujo se automatiza con scripts de Python.

## Estructura del proyecto

```
dataset/            (107 clases, ~6000 imagenes originales de Kaggle)
data/
  train/            (12 clases seleccionadas, 80%)
  test/             (12 clases seleccionadas, 20%)
graficos/           (generados por evaluate_model.py)
iac/
  main.tf
  variables.tf
  outputs.tf
scripts/
  .env.example
  requirements.txt
  prepare_dataset.py
  train_model.py
  publish_model.py
  evaluate_model.py
  improve_and_compare.py
```

## Clases seleccionadas

| Nombre sanscrito          | Nombre en espanol   | Imagenes |
|---------------------------|---------------------|----------|
| adho mukha svanasana      | Perro boca abajo    | 69       |
| balasana                  | Nino                | 71       |
| bhujangasana              | Cobra               | 73       |
| natarajasana              | Danzarin            | 72       |
| padmasana                 | Loto                | 68       |
| phalakasana               | Plancha             | 57       |
| ustrasana                 | Camello             | 87       |
| utkatasana                | Silla               | 73       |
| virabhadrasana i          | Guerrero I          | 54       |
| virabhadrasana ii         | Guerrero II         | 55       |
| virabhadrasana iii        | Guerrero III        | 61       |
| vriksasana                | Arbol               | 62       |

## Prerequisitos

- Azure CLI: `brew install azure-cli`
- Terraform: `brew install terraform`
- Python 3.9 o superior
- Cuenta de Azure con suscripcion activa

## Paso 1: Autenticarse en Azure

```bash
az login
az account show
```

## Paso 2: Desplegar infraestructura

```bash
cd iac/
terraform init
terraform plan
terraform apply -auto-approve
```

Obtener credenciales:

```bash
terraform output -json
```

## Paso 3: Configurar credenciales

```bash
cd ../scripts/
cp .env.example .env
```

Editar `.env` con los valores de `terraform output`.

## Paso 4: Preparar dataset

```bash
pip install -r requirements.txt
python prepare_dataset.py
```

Selecciona 12 clases del dataset original y divide en train (80%) y test (20%).

## Paso 5: Entrenar el modelo

```bash
python train_model.py
```

Crea el proyecto en Custom Vision, sube las imagenes de entrenamiento y ejecuta Quick Training.

## Paso 6: Publicar el modelo

```bash
python publish_model.py
```

## Paso 7: Evaluar el modelo

```bash
python evaluate_model.py
```

Ejecuta predicciones sobre el conjunto de test y genera:
- Matriz de confusion (graficos/confusion_matrix.png)
- Precision, Recall y F1-score por clase (graficos/metrics_bars.png)
- Curvas ROC con AUC por clase (graficos/roc_curves.png)
- Resumen en JSON (graficos/evaluation_summary.json)

## Paso 8: Mejorar y comparar

```bash
python improve_and_compare.py
```

Analiza errores del modelo v1, reentrena con Advanced Training, y compara metricas antes y despues.

## Paso 9: Limpiar recursos

```bash
cd ../iac/
terraform destroy -auto-approve
```

## Referencias

- Microsoft. Custom Vision documentation. https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/
- HashiCorp. Terraform AzureRM Provider. https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs
- Scikit-learn. Classification metrics. https://scikit-learn.org/stable/modules/model_evaluation.html
