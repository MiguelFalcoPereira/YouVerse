from pathlib import Path
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: Path = Field(..., env="MODEL_PATH")
    labels_path: Path = Field(..., env="LABELS_PATH")
    top_k: int = Field(3, env="TOP_K")
    num_threads: int = Field(os.cpu_count() or 1, env="NUM_THREADS")

    @field_validator("model_path", "labels_path")
    @classmethod
    def path_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Path does not exist: {value}")
        return value

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("TOP_K must be a positive integer")
        return value

    @field_validator("num_threads")
    @classmethod
    def num_threads_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("NUM_THREADS must be a positive integer")
        return value

settings = Settings()
