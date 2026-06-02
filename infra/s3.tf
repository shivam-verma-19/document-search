resource "aws_s3_bucket" "uploads" {
  bucket = "rag-pipeline-upload-bucket"
}

# Server-side encryption
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
          "aws:SecureTransport" : "false"
        }
      }
    }]
  })
}

resource "aws_s3_object" "lambda_deployment" {
  bucket = aws_s3_bucket.uploads.id
  key    = "deployments/lambda_backend.zip"
  source = var.lambda_zip_path
  etag   = filemd5(var.lambda_zip_path)
}

# ─── S3 Vectors (vector store for RAG) ───────────────────────────────────────
# The backend uses s3_vectors_client.py which calls the s3vectors boto3 client.
# VECTOR_BUCKET_NAME and VECTOR_INDEX_NAME env vars must match these resource names.

resource "aws_s3vectors_vector_bucket" "rag_vectors" {
  vector_bucket_name = "rag-vector-bucket"
}

resource "aws_s3vectors_index" "rag_doc_index" {
  vector_bucket_name = aws_s3vectors_vector_bucket.rag_vectors.vector_bucket_name
  index_name         = "rag-doc-index"

  # Must match EXPECTED_DIMENSION in s3_vectors_client.py (Gemini gemini-embedding-001 = 768)
  data_type  = "float32"
  dimension  = 768
  distance_metric = "cosine"
}
