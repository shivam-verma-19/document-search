resource "aws_lambda_function" "rag_api" {
  function_name    = "${var.project_name}-api"
  s3_bucket        = aws_s3_object.lambda_deployment.bucket
  s3_key           = aws_s3_object.lambda_deployment.key
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler     = "backend.app.main.handler"
  runtime     = "python3.10"
  role        = aws_iam_role.lambda_role.arn

  timeout     = 30
  memory_size = 1024

  environment {
    variables = {
      QUEUE_URL    = aws_sqs_queue.rag_queue.id
      SECRET_NAME  = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME  = aws_s3_bucket.uploads.bucket

      # S3 Vectors — must match resource names in s3.tf
      S3_VECTOR_BUCKET_NAME = aws_s3vectors_vector_bucket.rag_vectors.vector_bucket_name
      S3_VECTOR_INDEX_NAME  = aws_s3vectors_index.rag_doc_index.index_name
      EMBEDDING_DIMENSION   = "768"

      # Gemini model
      GEMINI_MODEL = "gemini-2.5-flash"

      # Cognito for JWT verification
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
  function_name    = "${var.project_name}-ingest-worker"
  s3_bucket        = aws_s3_object.lambda_deployment.bucket
  s3_key           = aws_s3_object.lambda_deployment.key
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler     = "backend.app.worker_lambda.handler"
  runtime     = "python3.10"
  role        = aws_iam_role.lambda_role.arn

  timeout     = 120
  memory_size = 1024

  environment {
    variables = {
      SECRET_NAME  = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME  = aws_s3_bucket.uploads.bucket

      S3_VECTOR_BUCKET_NAME = aws_s3vectors_vector_bucket.rag_vectors.vector_bucket_name
      S3_VECTOR_INDEX_NAME  = aws_s3vectors_index.rag_doc_index.index_name
      EMBEDDING_DIMENSION   = "768"
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
