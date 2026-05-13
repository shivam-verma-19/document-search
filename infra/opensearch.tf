resource "aws_opensearchserverless_collection" "rag_collection" {
  name = "rag-collection"
  type = "VECTORSEARCH"
}

resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "rag-encryption"
  type = "encryption"

  policy = jsonencode({
    Rules = [{
      ResourceType = "collection"
      Resource = ["collection/rag-collection"]
    }]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name = "rag-network"
  type = "network"

  policy = jsonencode([{
    Rules = [{
      ResourceType = "collection"
      Resource = ["collection/rag-collection"]
    }]
    AllowFromPublic = true
  }])
}

resource "aws_opensearchserverless_access_policy" "access" {
  name = "rag-access"
  type = "data"

  policy = jsonencode([{
    Rules = [{
      ResourceType = "index"
      Resource = ["index/rag-collection/*"]
      Permission = ["aoss:*"]
    }]
    Principal = ["*"]
  }])
}