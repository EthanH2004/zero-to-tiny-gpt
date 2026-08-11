"""Load paths and configuration for Tiny GPT V2."""

from pathlib import Path
from typing import Any
import tomllib


V2_DIRECTORY = Path(__file__).resolve().parent
CONFIGURATION_PATH = V2_DIRECTORY / "config.toml"


def load_settings() -> dict[str, Any]:
    with CONFIGURATION_PATH.open("rb") as configuration_file:
        return tomllib.load(configuration_file)


def v2_path(path: str | Path) -> Path:
    resolved_path = Path(path)

    if resolved_path.is_absolute():
        return resolved_path

    return V2_DIRECTORY / resolved_path

