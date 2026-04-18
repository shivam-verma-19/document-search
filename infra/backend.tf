terraform {
  backend "s3" {
    bucket         = "rag-terraform-state-bucket"
    key            = "rag-platform/terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "rag-terraform-lock"
    encrypt        = true
  }
}