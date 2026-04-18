resource "aws_s3_bucket_public_access_block" "block" {
  bucket = aws_s3_bucket.uploads.id

  block_public_acls   = true
  block_public_policy = true
  restrict_public_buckets = true
}