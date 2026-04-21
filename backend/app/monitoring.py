import boto3

cloudwatch = boto3.client("cloudwatch")


def push_metric(name, value):
    cloudwatch.put_metric_data(
        Namespace="RAG-App",
        MetricData=[{"MetricName": name, "Value": value, "Unit": "Count"}],
    )
