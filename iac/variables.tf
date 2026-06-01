variable "resource_group_name" {
  description = "Nombre del Resource Group"
  type        = string
  default     = "rg-yoga-classification-oduque"
}

variable "location" {
  description = "Región de Azure"
  type        = string
  default     = "eastus"
}

variable "training_resource_name" {
  description = "Nombre del recurso Custom Vision Training"
  type        = string
  default     = "cv-training-yoga-oduque"
}

variable "prediction_resource_name" {
  description = "Nombre del recurso Custom Vision Prediction"
  type        = string
  default     = "cv-prediction-yoga-oduque"
}
