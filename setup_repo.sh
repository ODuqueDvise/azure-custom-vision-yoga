#!/bin/bash
# Script para inicializar y subir el proyecto a GitHub
# Ejecutar desde la carpeta "Actividad 5/"

set -e

REPO_NAME="azure-custom-vision-yoga"

# Verificar que gh esta instalado
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) no encontrado. Instalando..."
    brew install gh
fi

# Verificar autenticacion
if ! gh auth status &> /dev/null 2>&1; then
    echo "Autenticandose en GitHub..."
    gh auth login
fi

# Crear repositorio publico
echo "Creando repositorio publico: $REPO_NAME"
gh repo create "$REPO_NAME" --public --description "Evaluacion de clasificacion de poses de yoga con Azure Custom Vision - Metricas P/R/F1/ROC/AUC" || true

# Inicializar git
git init
git add .
git commit -m "Evaluacion de modelo de clasificacion de poses de yoga

- Terraform (IaC) para aprovisionar recursos Azure Custom Vision
- Scripts Python para entrenar, evaluar y mejorar el modelo
- Dataset: 12 clases de yoga (802 imagenes)
- Metricas: Precision, Recall, F1-score, ROC/AUC
- Comparacion Quick Training vs Advanced Training"

# Obtener usuario de GitHub
GH_USER=$(gh api user -q .login)
git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git" 2>/dev/null || \
    git remote set-url origin "https://github.com/$GH_USER/$REPO_NAME.git"

git branch -M main
git push -u origin main

echo ""
echo "Repositorio publicado: https://github.com/$GH_USER/$REPO_NAME"
