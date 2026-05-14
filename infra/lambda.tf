resource "aws_lambda_function" "rag_api" {
  function_name = "${var.project_name}-api"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler = "app.main.handler"
  runtime = "python3.10"
  role    = aws_iam_role.lambda_role.arn

  timeout     = 30
  memory_size = 1024

  environment {
    variables = {
      QUEUE_URL               = aws_sqs_queue.rag_queue.id
      AWS_REGION              = var.aws_region
      SECRET_NAME             = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME             = aws_s3_bucket.uploads.bucket
      USE_BEDROCK             = "true"
      BEDROCK_CLAUDE_MODEL_ID = "anthropic.claude-sonnet-4-5"
      BEDROCK_LLAMA_MODEL_ID  = "meta.llama3-8b-instruct-v1:0"
      OLLAMA_BASE_URL         = var.ollama_base_url
      OLLAMA_MODEL            = var.ollama_model
      OLLAMA_TIMEOUT          = "60"
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }
}

resource "aws_lambda_function" "rag_ingest_worker" {
  function_name = "${var.project_name}-ingest-worker"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler = "app.worker_lambda.handler"
  runtime = "python3.10"
  role    = aws_iam_role.lambda_role.arn

  timeout     = 120
  memory_size = 1024

  environment {
    variables = {
      AWS_REGION  = var.aws_region
      SECRET_NAME = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME = aws_s3_bucket.uploads.bucket
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.rag_api.function_name}"
  retention_in_days = 7
}