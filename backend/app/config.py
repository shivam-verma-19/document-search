from pydantic_settings import BaseSettings, SettingsConfigDict
import json
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    aws_region: str = ""
    queue_url: str = ""
    bucket_name: str = ""

    cognito_user_pool_id: str = ""
    cognito_client_id: str = ""

    max_upload_size: int = 10485760

    allowed_upload_extensions: str = ""
    allowed_upload_mimes: str = ""
    forbidden_upload_patterns: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

    def _parse_list(self, value: str) -> List[str]:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x).lower() for x in parsed]
        except Exception:
            pass
        return [x.strip().lower() for x in value.split(",")]

    @property
    def allowed_upload_extensions_list(self) -> List[str]:
        return self._parse_list(self.allowed_upload_extensions)

    @property
    def allowed_upload_mimes_list(self) -> List[str]:
        return self._parse_list(self.allowed_upload_mimes)

    @property
    def forbidden_upload_patterns_list(self) -> List[str]:
        return self._parse_list(self.forbidden_upload_patterns)
    

@lru_cache
def get_settings() -> Settings:
    return Settings()