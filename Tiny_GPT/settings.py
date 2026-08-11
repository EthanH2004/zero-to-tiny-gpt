"""Load Tiny GPT's shared TOML configuration."""

from pathlib import Path
from typing import Any
import tomllib


PROJECT_DIRECTORY = Path(__file__).resolve().parent
CONFIGURATION_PATH = PROJECT_DIRECTORY / "config.toml"


def load_settings(
    path: Path = CONFIGURATION_PATH,
) -> dict[str, Any]:
    with path.open("rb") as configuration_file:
        return tomllib.load(configuration_file)


def project_path(path: str | Path) -> Path:
    resolved_path = Path(path)

    if resolved_path.is_absolute():
        return resolved_path

    return PROJECT_DIRECTORY / resolved_path

