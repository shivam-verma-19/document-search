resource "aws_s3_bucket" "uploads" {
  bucket = "rag-pipeline-upload-bucket"
}

# Server-side encryption configuration (separate resource to avoid deprecation warning)
resource "aws_s3_bucket_server_side_encryption_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "uploads_block" {
  bucket = aws_s3_bucket.uploads.id

  block_public_acls       = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "uploads_tls" {
  bucket = aws_s3_bucket.uploads.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "EnforceTLS"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource = [
        aws_s3_bucket.uploads.arn,
        "${aws_s3_bucket.uploads.arn}/*",
      ]
      Condition = {
        Bool = {
          "aws:SecureTransport": "false"
        }
      }
    }]
  })
}

resource "aws_s3_object" "lambda_deployment" {
  bucket = aws_s3_bucket.uploads.id
  key    = "deployments/lambda_backend.zip"
  source = var.lambda_zip_path
  
  # The etag forces Terraform to upload the file again if the code changes
  etag   = filemd5(var.lambda_zip_path)
}