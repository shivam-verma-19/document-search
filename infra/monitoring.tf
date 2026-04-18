resource "aws_cloudwatch_dashboard" "rag_dashboard" {
  dashboard_name = "rag-platform-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        x    = 0
        y    = 0
        width = 12
        height = 6

        properties = {
          title = "Lambda Invocations"
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.rag_lambda.function_name]
          ]
          region = var.aws_region
        }
      },
      {
        type = "metric"
        x    = 12
        y    = 0
        width = 12
        height = 6

        properties = {
          title = "Lambda Errors"
          metrics = [
            ["AWS/Lambda", "Errors", "FunctionName", aws_lambda_function.rag_lambda.function_name]
          ]
          region = var.aws_region
        }
      }
    ]
  })
}