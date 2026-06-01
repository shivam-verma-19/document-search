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
  default     = "rag-pipeline-upload-bucket"
}

variable "lambda_zip_path" {
  default = "../backend/deployment.zip"
}

variable "alert_email" {
  description = "Email for CloudWatch alerts"
  type        = string
  default     = "shlok.shivam0227@gmail.com"
}

variable "gemini_api_key" {
  description = "Gemini API key stored in Secrets Manager"
  type        = string
  sensitive   = true
}
