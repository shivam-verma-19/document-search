variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  default     = "rag-platform"
}

variable "s3_bucket_name" {
  description = "S3 bucket for uploads"
  default     = "rag-pipeline-upload-bucket"
}

variable "openai_api_key" {
  sensitive = true
}

variable "pinecone_api_key" {
  sensitive = true
}

variable "lambda_zip_path" {
  default = "../backend/deployment.zip"
}

variable "alert_email" {
  description = "Email for CloudWatch alerts"
  type        = string
  default     = "shlok.shivam0227@gmail.com"
}