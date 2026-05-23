"""App-wide settings: cache dirs, HF auth, output roots."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WHOA_", env_file=".env", extra="ignore")

    hf_token: str | None = None
    hf_home: Path | None = None
    output_root: Path = Path("outputs")
    user_data_dir: Path = Path.home() / ".whoa_llm"


def get_settings() -> Settings:
    s = Settings()
    if s.hf_token is None:
        s.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if s.hf_home is None and "HF_HOME" in os.environ:
        s.hf_home = Path(os.environ["HF_HOME"])
    s.output_root.mkdir(parents=True, exist_ok=True)
    s.user_data_dir.mkdir(parents=True, exist_ok=True)
    return s


def hf_login() -> bool:
    """Log into the Hugging Face Hub if a token is configured.

    Returns True if a login was performed, False if no token was found.
    Imports lazily so the package works without ``huggingface_hub`` installed.
    """
    s = get_settings()
    if not s.hf_token:
        return False
    from huggingface_hub import login

    login(token=s.hf_token, add_to_git_credential=False)
    return True
