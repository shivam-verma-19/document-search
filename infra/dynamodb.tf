resource "aws_dynamodb_table" "tf_lock" {
  name         = "rag-terraform-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# Cache table for query results
resource "aws_dynamodb_table" "cache" {
  name           = "rag-cache"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "query"

  attribute {
    name = "query"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name    = "${var.project_name}-cache"
    Purpose = "caching"
  }
}

# Metrics table for analytics
resource "aws_dynamodb_table" "metrics" {
  name           = "rag-metrics"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "id"
  range_key      = "timestamp"

  attribute {
    name = "id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  global_secondary_index {
    name            = "user-timestamp-index"
    hash_key        = "user_id"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  tags = {
    Name    = "${var.project_name}-metrics"
    Purpose = "analytics"
  }
}

# Evaluation table for model evaluation
resource "aws_dynamodb_table" "eval" {
  name           = "rag-eval"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "query"
  range_key      = "eval_id"

  attribute {
    name = "query"
    type = "S"
  }

  attribute {
    name = "eval_id"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name    = "${var.project_name}-eval"
    Purpose = "evaluation"
  }
}