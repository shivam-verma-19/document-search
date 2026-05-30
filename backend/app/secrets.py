import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_SECRET_NAME = os.getenv("SECRET_NAME", "rag-platform-secrets")
_AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

_cache: dict = {}


def get_secret(key: str) -> str:
    """Return a secret value, preferring Secrets Manager over env vars."""
    # Env var takes precedence in local dev
    env_val = os.getenv(key)
    if env_val:
        return env_val

    if not _cache:
        _load_secrets()

    return _cache.get(key, "")


def _load_secrets() -> None:
    try:
        client = boto3.client("secretsmanager", region_name=_AWS_REGION)
        response = client.get_secret_value(SecretId=_SECRET_NAME)
        raw = response.get("SecretString", "{}")
        _cache.update(json.loads(raw))
        logger.info("Secrets loaded from Secrets Manager")
    except ClientError as e:
        logger.warning(f"Could not load secrets from Secrets Manager: {e}")
