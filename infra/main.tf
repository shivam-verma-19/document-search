provider "aws" {
  region = "ap-south-1"
}

# S3 bucket for Terraform state
resource "aws_s3_bucket" "tf_state" {
  bucket = "rag-pipeline-terraform-state-bucket"
}

# Enable versioning
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.tf_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.tf_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}