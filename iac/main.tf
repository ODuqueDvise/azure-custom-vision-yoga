terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location

  tags = {
    project = "yoga-pose-classification"
    course  = "vision-por-computador"
  }
}

resource "azurerm_cognitive_account" "training" {
  name                = var.training_resource_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  kind                = "CustomVision.Training"
  sku_name            = "S0"

  tags = {
    type = "training"
  }
}

resource "azurerm_cognitive_account" "prediction" {
  name                = var.prediction_resource_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  kind                = "CustomVision.Prediction"
  sku_name            = "S0"

  tags = {
    type = "prediction"
  }
}
