########################################
# 🔹 LAMBDA EXECUTION ROLE
########################################
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

########################################
# 🔹 BASIC LAMBDA LOGGING
########################################
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

########################################
# 🔹 CUSTOM LAMBDA POLICY
########################################
resource "aws_iam_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [

      ##################################
      # DynamoDB (cache + metrics)
      ##################################
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:DeleteItem"
        ]
        Resource = "arn:aws:dynamodb:*:*:table/rag-*"
      },

      ##################################
      # S3 (file uploads)
      ##################################
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "arn:aws:s3:::rag-pipeline-upload-bucket/*"
      },
      {
        Effect = "Allow"
        Action = "s3:ListBucket"
        Resource = "arn:aws:s3:::rag-pipeline-upload-bucket"
      },

      ##################################
      # SQS (queue processing)
      ##################################
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.rag_queue.arn
      },

      ##################################
      # Secrets Manager (API keys)
      ##################################
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "*"
      },

      ##################################
      # OpenSearch Serverless (CRITICAL)
      ##################################
      {
        Effect = "Allow"
        Action = [
          "aoss:APIAccessAll"
        ]
        Resource = "*"
      }
    ]
  })
}

########################################
# 🔹 ATTACH POLICY TO LAMBDA ROLE
########################################
resource "aws_iam_role_policy_attachment" "lambda_policy_attach" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

########################################
# 🔹 SECRETS ACCESS INLINE POLICY
########################################
resource "aws_iam_role_policy" "secrets_access" {
  name = "lambda-secrets-access"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue"
      ]
      Resource = aws_secretsmanager_secret.rag_secrets.arn
    }]
  })
}

########################################
# 🔹 API LAMBDA ROLE
########################################
resource "aws_iam_role" "api_lambda_role" {
  name = "${var.project_name}-api-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

########################################
# 🔹 WORKER LAMBDA ROLE
########################################
resource "aws_iam_role" "worker_lambda_role" {
  name = "${var.project_name}-worker-role"
  assume_role_policy = aws_iam_role.api_lambda_role.assume_role_policy
}