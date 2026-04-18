resource "aws_lambda_function" "rag_lambda" {
  function_name = "${var.project_name}-api"

  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler = "app.main.handler"
  runtime = "python3.10"

  role = aws_iam_role.lambda_role.arn

  timeout     = 30
  memory_size = 1024

  environment {
    variables = {
      OPENAI_API_KEY   = "your_openai_key"
      PINECONE_API_KEY = "your_pinecone_key"
      AWS_REGION       = var.aws_region
    }
  }
}
environment {
  variables = {
    SECRET_NAME = aws_secretsmanager_secret.rag_secrets.name
  }
}