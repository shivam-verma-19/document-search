import os

import boto3

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

_bedrock_client = None


def _get_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def rerank(query: str, docs: list) -> list:
    if not docs:
        return docs

    text_sources = [
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {"text": doc.page_content}
            }
        }
        for doc in docs
    ]

    response = _get_client().rerank(
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {
                    "modelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.rerank-v1:0"
                }
            }
        },
        sources=text_sources,
        queries=[{"type": "TEXT", "textQuery": {"text": query}}]
    )

    ranked_indices = [item["index"] for item in response["rerankingResults"]]
    return [docs[i] for i in ranked_indices]