output "s3_bucket_name" {
  value = aws_s3_bucket.uploads.bucket
}

output "dynamodb_cache_table" {
  value = aws_dynamodb_table.cache.name
}

output "dynamodb_metrics_table" {
  value = aws_dynamodb_table.metrics.name
}

output "api_url" {
  value = aws_apigatewayv2_api.http_api.api_endpoint
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.user_pool.id
}

output "cognito_client_id" {
  value = aws_cognito_user_pool_client.client.id
}