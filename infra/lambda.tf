resource "aws_lambda_function" "rag_api" {
  function_name = "${var.project_name}-api"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler = "backend.app.main.handler"
  runtime = "python3.10"
  role    = aws_iam_role.lambda_role.arn

  timeout     = 30
  memory_size = 1024

  environment {
    variables = {
      QUEUE_URL               = aws_sqs_queue.rag_queue.id
      SECRET_NAME             = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME             = aws_s3_bucket.uploads.bucket
      USE_BEDROCK             = "true"
      BEDROCK_CLAUDE_MODEL_ID = "anthropic.claude-sonnet-4-5"
      BEDROCK_LLAMA_MODEL_ID  = "meta.llama3-8b-instruct-v1:0"

      OPENSEARCH_ENDPOINT = aws_opensearchserverless_collection.rag.collection_endpoint

      # Variables for JWT verification
      COGNITO_USER_POOL_ID = aws_cognito_user_pool.user_pool.id
      COGNITO_CLIENT_ID    = aws_cognito_user_pool_client.client.id
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic,
    aws_iam_role_policy_attachment.lambda_policy_attach,
    aws_opensearchserverless_access_policy.access,
  ]
}

resource "aws_lambda_function" "rag_ingest_worker" {
  function_name = "${var.project_name}-ingest-worker"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler = "backend.app.worker_lambda.handler"
  runtime = "python3.10"
  role    = aws_iam_role.lambda_role.arn

  timeout     = 120
  memory_size = 1024

  environment {
    variables = {
      SECRET_NAME = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME = aws_s3_bucket.uploads.bucket

      OPENSEARCH_ENDPOINT = aws_opensearchserverless_collection.rag.collection_endpoint
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic,
    aws_iam_role_policy_attachment.lambda_policy_attach,
    aws_opensearchserverless_access_policy.access,
  ]
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.rag_api.function_name}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "worker_lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.rag_ingest_worker.function_name}"
  retention_in_days = 7
}