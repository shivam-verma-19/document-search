# backend/app/config.py

from functools import lru_cache
from typing import Set

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    aws_region: str = Field(default="us-east-1")
    queue_url: str = Field(default="")
    bucket_name: str = Field(default="rag-pipeline-upload-bucket")

    cognito_user_pool_id: str = Field(default="")
    cognito_client_id: str = Field(default="")

    max_upload_size: int = Field(default=10 * 1024 * 1024)

    # ✅ FIXED
    allowed_upload_extensions: Set[str] = Field(default={"pdf", "txt", "docx"})
    allowed_upload_mimes: Set[str] = Field(
        default={
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
    )

    forbidden_upload_patterns: Set[str] = Field(
        default={"<script>", "javascript:", "base64,"}
    )

    # ✅ ADD THIS (CRITICAL)
    @field_validator("allowed_upload_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v):
        if isinstance(v, str):
            return {x.strip() for x in v.split(",")}
        return v

    @field_validator("allowed_upload_mimes", mode="before")
    @classmethod
    def parse_mimes(cls, v):
        if isinstance(v, str):
            return {x.strip() for x in v.split(",")}
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()
