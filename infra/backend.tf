terraform {
  backend "s3" {
    bucket         = "rag-pipeline-terraform-state-bucket"
    key            = "rag-platform/terraform.tfstate"
    region         = "ap-south-1"
    use_lockfile = false
    encrypt        = true
  }
}