resource "aws_lambda_function" "rag_lambda" {
  function_name = "${var.project_name}-api"

  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  handler = "app.main.handler"
  runtime = "python3.10"

  role = aws_iam_role.lambda_role.arn

  timeout     = 30
  memory_size = 1024

  provisioned_concurrent_executions = 2

  dead_letter_config {
    target_arn = aws_sqs_queue.dlq.arn
  }

  environment {
    variables = {
      QUEUE_URL   = aws_sqs_queue.rag_queue.id
      AWS_REGION       = var.aws_region
      SECRET_NAME = aws_secretsmanager_secret.rag_secrets.name
    }
  }

}