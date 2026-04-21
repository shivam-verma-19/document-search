import boto3

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("rag-eval")


def store_eval(query, latency, precision):
    table.put_item(Item=
        {  
            "query": query, 
            "latency": latency, 
            "precision": precision
        }
    )
