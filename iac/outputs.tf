output "training_endpoint" {
  value     = azurerm_cognitive_account.training.endpoint
  sensitive = false
}

output "training_key" {
  value     = azurerm_cognitive_account.training.primary_access_key
  sensitive = true
}

output "prediction_endpoint" {
  value     = azurerm_cognitive_account.prediction.endpoint
  sensitive = false
}

output "prediction_key" {
  value     = azurerm_cognitive_account.prediction.primary_access_key
  sensitive = true
}

output "prediction_resource_id" {
  value     = azurerm_cognitive_account.prediction.id
  sensitive = false
}
