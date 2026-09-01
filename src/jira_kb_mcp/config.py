"""Configuration loading for jira-kb-mcp.

Reads settings from environment variables (optionally loaded from a local
.env file). Nothing here is ever hardcoded — credentials live only in the
user's own .env, which is git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATA_DIR = Path.home() / ".jira-kb-mcp"


@dataclass(frozen=True)
class JiraConfig:
    base_url: str
    email: str
    api_token: str

    @property
    def auth(self) -> tuple[str, str]:
        return (self.email, self.api_token)


@dataclass(frozen=True)
class AppConfig:
    jira: JiraConfig
    data_dir: Path
    embedding_model: str


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            "Run 'jira-kb init' to configure your Jira connection, "
            "or set it in a .env file / your shell environment."
        )
    return value


def load_config(env_file: Path | None = None) -> AppConfig:
    """Load configuration from environment / .env file.

    Looks for a .env file in the current directory by default, then falls
    back to ~/.jira-kb-mcp/.env so the config survives across projects.
    """
    candidates = [env_file] if env_file else [Path.cwd() / ".env", DEFAULT_DATA_DIR / ".env"]
    for candidate in candidates:
        if candidate and candidate.exists():
            load_dotenv(candidate, override=False)

    base_url = _require("JIRA_URL").rstrip("/")
    email = _require("JIRA_EMAIL")
    api_token = _require("JIRA_API_TOKEN")

    data_dir = Path(os.environ.get("JIRA_KB_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    embedding_model = os.environ.get(
        "JIRA_KB_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    return AppConfig(
        jira=JiraConfig(base_url=base_url, email=email, api_token=api_token),
        data_dir=data_dir,
        embedding_model=embedding_model,
    )
