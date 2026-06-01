resource "aws_secretsmanager_secret" "rag_secrets" {
  name = "rag-platform-secrets"
}

resource "aws_secretsmanager_secret_version" "rag_secrets_value" {
  secret_id     = aws_secretsmanager_secret.rag_secrets.id
  secret_string = jsonencode({
    GEMINI_API_KEY    = var.gemini_api_key   # set via TF_VAR_gemini_api_key
  })

  lifecycle {
    ignore_changes = [secret_string]  # prevents future applies from overwriting
  }
}
