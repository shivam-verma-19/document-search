from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    aws_region: str = "ap-south-1"

    bucket_name: str = "rag-upload-bucket"
    queue_url: str = ""

    cognito_user_pool_id: str = ""
    cognito_client_id: str = ""

    # ✅ Accept as string first
    allowed_upload_extensions: str = "pdf,txt,docx"
    allowed_upload_mimes: str = "application/pdf,text/plain"
    forbidden_upload_patterns: str = "<script>,DROP TABLE"

    max_upload_size: int = 5 * 1024 * 1024

    class Config:
        env_file = ".env"

    # ✅ Convert to list dynamically
    @property
    def allowed_upload_extensions_list(self) -> List[str]:
        return [x.strip().lower() for x in self.allowed_upload_extensions.split(",")]

    @property
    def allowed_upload_mimes_list(self) -> List[str]:
        return [x.strip() for x in self.allowed_upload_mimes.split(",")]

    @property
    def forbidden_upload_patterns_list(self) -> List[str]:
        return [x.strip().lower() for x in self.forbidden_upload_patterns.split(",")]


@lru_cache
def get_settings():
    return Settings()
