from functools import lru_cache
from pydantic import Field
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
    allowed_upload_extensions: set[str] = Field(
        default={"pdf", "txt", "docx"}
    )
    allowed_upload_mimes: set[str] = Field(
        default={
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
    )
    forbidden_upload_patterns: set[str] = Field(
        default={"<script>", "javascript:", "base64,"}
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()