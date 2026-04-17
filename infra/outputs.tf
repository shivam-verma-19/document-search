output "s3_bucket_name" {
  value = aws_s3_bucket.uploads.bucket
}

output "dynamodb_cache_table" {
  value = aws_dynamodb_table.cache.name
}

output "dynamodb_metrics_table" {
  value = aws_dynamodb_table.metrics.name
}