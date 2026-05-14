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
variable "ollama_base_url" {
  description = "Base URL for the local Ollama server (local fallback tier)"
  type        = string
  default     = "http://localhost:11434"
}

variable "ollama_model" {
  description = "Ollama model name to use as local fallback (must be pulled via 'ollama pull <model>')"
  type        = string
  default     = "llama3"
}
