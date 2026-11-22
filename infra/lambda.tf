resource "aws_lambda_function" "rag_api" {
  function_name    = "${var.project_name}-api"
  s3_bucket        = aws_s3_object.lambda_deployment.bucket
  s3_key           = aws_s3_object.lambda_deployment.key
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler     = "backend.app.main.handler"
  runtime     = "python3.13"
  role        = aws_iam_role.lambda_role.arn

  timeout     = 30
  memory_size = 1024

  environment {
    variables = {
      APP_ENV                    = "production"
      QUEUE_URL                  = aws_sqs_queue.rag_queue.id
      SECRET_NAME                = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME                = aws_s3_bucket.uploads.bucket
      ENABLE_RERANKER            = "false"
      EMBEDDING_BACKEND          = "gemini"
      EMBEDDING_DIMENSION        = "768"
      GEMINI_EMBEDDING_MODEL     = "models/gemini-embedding-001"
      HYDE_ENABLED               = "false"
      ENABLE_BACKGROUND_EVAL     = "false"
      ALLOWED_UPLOAD_EXTENSIONS  = "pdf,txt,csv"
      ALLOWED_UPLOAD_MIMES       = "application/pdf,text/plain,text/csv,application/vnd.ms-excel"
      MAX_UPLOAD_SIZE            = "10485760"
      DYNAMODB_BM25_TABLE        = aws_dynamodb_table.bm25_cache.name
      DYNAMODB_METRICS_TABLE     = aws_dynamodb_table.metrics.name
      DYNAMODB_EVAL_TABLE        = aws_dynamodb_table.eval.name
      DYNAMODB_SESSIONS_TABLE    = aws_dynamodb_table.sessions.name

      # S3 Vectors — must match resource names in s3.tf
      S3_VECTOR_BUCKET_NAME = aws_s3vectors_vector_bucket.rag_vectors.vector_bucket_name
      S3_VECTOR_INDEX_NAME  = aws_s3vectors_index.rag_doc_index.index_name

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
  runtime     = "python3.13"
  role        = aws_iam_role.lambda_role.arn

  timeout     = 120
  memory_size = 1024

  environment {
    variables = {
      SECRET_NAME                = aws_secretsmanager_secret.rag_secrets.name
      BUCKET_NAME                = aws_s3_bucket.uploads.bucket
      EMBEDDING_BACKEND          = "gemini"
      EMBEDDING_DIMENSION        = "768"
      GEMINI_EMBEDDING_MODEL     = "models/gemini-embedding-001"
      DYNAMODB_BM25_TABLE        = aws_dynamodb_table.bm25_cache.name
      DYNAMODB_IDEMPOTENCY_TABLE = aws_dynamodb_table.processed_keys.name

      S3_VECTOR_BUCKET_NAME     = aws_s3vectors_vector_bucket.rag_vectors.vector_bucket_name
      S3_VECTOR_INDEX_NAME      = aws_s3vectors_index.rag_doc_index.index_name
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

resource "aws_lambda_function" "cognito_pre_signup" {
  function_name    = "${var.project_name}-cognito-pre-signup"
  s3_bucket        = aws_s3_object.lambda_deployment.bucket
  s3_key           = aws_s3_object.lambda_deployment.key
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler     = "backend.app.cognito_triggers.pre_signup_handler"
  runtime     = "python3.13"
  role        = aws_iam_role.cognito_pre_signup.arn
  timeout     = 5
  memory_size = 128

  depends_on = [
    aws_iam_role_policy_attachment.cognito_pre_signup_logs,
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

resource "aws_cloudwatch_log_group" "cognito_pre_signup_logs" {
  name              = "/aws/lambda/${aws_lambda_function.cognito_pre_signup.function_name}"
  retention_in_days = 7
}
