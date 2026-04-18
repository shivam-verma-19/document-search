provider "aws" {
  region = "ap-south-1"
}

resource "aws_s3_bucket" "uploads" {
  bucket = "rag-upload-bucket"
}

resource "aws_dynamodb_table" "cache" {
  name = "rag-cache"
  billing_mode = "PAY_PER_REQUEST"

  hash_key = "query"

  attribute {
    name = "query"
    type = "S"
  }
}

resource "aws_dynamodb_table" "metrics" {
  name = "rag-metrics"
  billing_mode = "PAY_PER_REQUEST"

  hash_key = "id"

  attribute {
    name = "id"
    type = "S"
  }
}

resource "aws_s3_bucket" "tf_state" {
  bucket = "rag-terraform-state-bucket"
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.tf_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.tf_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}