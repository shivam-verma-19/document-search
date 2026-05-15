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
          "sqs:GetQueueAttributes",
          "sqs:GetQueueUrl"
        ]
        Resource = [
          aws_sqs_queue.rag_queue.arn,
          aws_sqs_queue.dlq.arn
        ]
      },

      ##################################
      # Secrets Manager
      # ✅ FIX 10: removed duplicate — was also in inline policy below
      ##################################
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.rag_secrets.arn
      },

      ##################################
      # OpenSearch Serverless
      ##################################
      {
        Effect = "Allow"
        Action = [
          "aoss:APIAccessAll"
        ]
        Resource = "*"
      },

      ##################################
      # ✅ FIX 8: Bedrock — was missing entirely
      # bedrock_router.py calls bedrock-runtime for Claude + Llama
      ##################################
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel"
        ]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5",
          "arn:aws:bedrock:*::foundation-model/meta.llama3-8b-instruct-v1:0",
          "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1",
          "arn:aws:bedrock:*::foundation-model/amazon.rerank-v1:0"
        ]
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