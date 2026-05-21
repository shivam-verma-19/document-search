locals {
  deployment_zip = "${path.root}/../backend/deployment.zip"
}

resource "aws_lambda_function" "rag_api" {
  function_name = "${var.project_name}-api"
  s3_bucket = aws_s3_bucket.lambda_deployments.id
  s3_key    = aws_s3_object.lambda_zip.key

  source_code_hash = filebase64sha256("${path.module}/../backend/deployment.zip")

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

      FAISS_PERSIST_DIR = "/tmp/faiss"

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
  ]
}

resource "aws_lambda_function" "rag_ingest_worker" {
  function_name = "${var.project_name}-ingest-worker"
  s3_bucket = aws_s3_bucket.lambda_deployments.id
  s3_key    = aws_s3_object.lambda_zip.key

  source_code_hash = filebase64sha256("${path.module}/../backend/deployment.zip")

  handler = "backend.app.worker_lambda.handler"
  runtime = "python3.10"
  role    = aws_iam_role.lambda_role.arn

  timeout     = 120
  memory_size = 1024

  environment {
    variables = {
      SECRET_NAME = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME = aws_s3_bucket.uploads.bucket

      FAISS_PERSIST_DIR = "/tmp/faiss"
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic,
    aws_iam_role_policy_attachment.lambda_policy_attach,
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

resource "aws_s3_object" "lambda_zip" {
  bucket = aws_s3_bucket.lambda_deployments.id
  key    = "deployment.zip"
  source = local.deployment_zip

  etag = filemd5(local.deployment_zip)
}