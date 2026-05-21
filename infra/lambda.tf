locals {
  deployment_zip = "${path.root}/../backend/deployment.zip"
}

# Builds deployment.zip from source whenever any .py file changes.
# This runs before Terraform tries to read the zip, so filemd5() never fails.
resource "null_resource" "build_lambda_zip" {
  triggers = {
    source_hash = sha1(join("", [
      for f in fileset("${path.root}/../backend", "**/*.py") :
      filesha1("${path.root}/../backend/${f}")
    ]))
    requirements_hash = filesha1("${path.root}/../backend/requirements-lambda.txt")
  }

  provisioner "local-exec" {
    working_dir = "${path.root}/../backend"
    command     = <<-EOT
      set -e
      rm -rf package deployment.zip
      mkdir -p ./package/backend

      pip install -r requirements-lambda.txt \
        --target ./package \
        --upgrade \
        --quiet

      # Strip heavy ML packages — Bedrock handles these
      rm -rf ./package/torch \
             ./package/torchvision \
             ./package/torchaudio \
             ./package/sentence_transformers \
             ./package/transformers \
             ./package/tokenizers \
             ./package/huggingface_hub \
             ./package/safetensors \
             ./package/nvidia* \
             ./package/*.egg-info

      cp __init__.py ./package/backend/__init__.py
      cp -r app ./package/backend/

      cd package
      zip -r ../deployment.zip . \
        -x "*__pycache__*" \
        -x "*.pyc" \
        -x "*.dist-info/*" \
        -x "*.egg-info/*"

      cd ..
      echo "Built deployment.zip: $(du -sh deployment.zip | cut -f1)"

      SIZE=$(du -m deployment.zip | cut -f1)
      if [ "$SIZE" -gt 70 ]; then
        echo "ERROR: Lambda zip exceeds 70MB limit ($SIZE MB)"
        exit 1
      fi
    EOT
  }
}

resource "aws_s3_object" "lambda_zip" {
  bucket = aws_s3_bucket.lambda_deployments.id
  key    = "deployment.zip"
  source = local.deployment_zip

  # etag forces S3 re-upload when the zip content changes
  etag = filemd5(local.deployment_zip)

  depends_on = [null_resource.build_lambda_zip]
}

resource "aws_lambda_function" "rag_api" {
  function_name = "${var.project_name}-api"
  s3_bucket     = aws_s3_bucket.lambda_deployments.id
  s3_key        = aws_s3_object.lambda_zip.key

  # Derived from the S3 object — no local file read at plan time
  source_code_hash = aws_s3_object.lambda_zip.etag

  handler     = "backend.app.main.handler"
  runtime     = "python3.10"
  role        = aws_iam_role.lambda_role.arn
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
    null_resource.build_lambda_zip,
  ]
}

resource "aws_lambda_function" "rag_ingest_worker" {
  function_name = "${var.project_name}-ingest-worker"
  s3_bucket     = aws_s3_bucket.lambda_deployments.id
  s3_key        = aws_s3_object.lambda_zip.key

  # Derived from the S3 object — no local file read at plan time
  source_code_hash = aws_s3_object.lambda_zip.etag

  handler     = "backend.app.worker_lambda.handler"
  runtime     = "python3.10"
  role        = aws_iam_role.lambda_role.arn
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
    null_resource.build_lambda_zip,
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
