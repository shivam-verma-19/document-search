variable "aws_region" {
  description = "AWS region"
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Project name"
  default     = "rag-platform"
}

variable "s3_bucket_name" {
  description = "S3 bucket for uploads"
  default     = "rag-upload-bucket"
}