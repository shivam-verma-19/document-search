import boto3, hashlib

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("rag-cache")


def hash_query(q):
    return hashlib.sha256(q.encode()).hexdigest()


def get_cache(query):
    res = table.get_item(Key={"query": hash_query(query)})
    return res.get("Item", {}).get("answer")


def set_cache(query, answer):
    table.put_item(Item={"query": hash_query(query), "answer": answer})
