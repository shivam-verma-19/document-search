provider "aws" {
  region = "ap-south-1"
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
