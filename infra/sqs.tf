resource "aws_sqs_queue" "rag_queue" {
  name = "rag-processing-queue"
}

resource "aws_sqs_queue" "dlq" {
  name = "rag-dlq"
}

resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn = aws_sqs_queue.rag_queue.arn
  function_name    = aws_lambda_function.rag_lambda.arn
}