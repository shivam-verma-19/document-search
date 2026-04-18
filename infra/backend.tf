terraform {
  backend "s3" {
    bucket         = "rag-terraform-state"
    key            = "terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "rag-lock-table"
  }
}