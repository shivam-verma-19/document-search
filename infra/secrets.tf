resource "aws_secretsmanager_secret" "rag_secrets" {
  name = "rag-platform-secrets"
}

resource "aws_secretsmanager_secret_version" "rag_secrets_value" {
  secret_id = aws_secretsmanager_secret.rag_secrets.id

  secret_string = jsonencode({
    OPENAI_API_KEY   = var.openai_api_key
    OPENSEARCH_ENDPOINT = aws_opensearchserverless_collection.rag.collection_endpoint
  })
}