resource "aws_secretsmanager_secret" "rag_secrets" {
  name = "rag-platform-secrets"
}

resource "aws_secretsmanager_secret_version" "rag_secrets_value" {
  secret_id = aws_secretsmanager_secret.rag_secrets.id

  secret_string = jsonencode({
    OPENAI_API_KEY   = "your_openai_key"
    PINECONE_API_KEY = "your_pinecone_key"
  })
}