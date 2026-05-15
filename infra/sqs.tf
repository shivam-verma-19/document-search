resource "aws_sqs_queue" "rag_queue" {
  name = "rag-processing-queue"

  visibility_timeout_seconds = 180

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 5
  })
}

resource "aws_sqs_queue" "dlq" {
  name = "rag-dlq"
}

resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn = aws_sqs_queue.rag_queue.arn
  function_name    = aws_lambda_function.rag_ingest_worker.arn

  batch_size = 1
}