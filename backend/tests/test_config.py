import boto3

from backend.app.config import _configure_aws_session


def test_configure_aws_session_uses_app_region_without_default_region(monkeypatch):
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)

    _configure_aws_session("ap-south-1")

    assert boto3.DEFAULT_SESSION.region_name == "ap-south-1"
    assert boto3.client("s3").meta.region_name == "ap-south-1"
