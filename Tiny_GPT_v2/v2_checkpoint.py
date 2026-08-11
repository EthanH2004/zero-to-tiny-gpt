"""Checkpoint helpers that store the V2 BPE tokenizer with the model."""

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent / "Tiny_GPT"

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from tiny_gpt import ModelConfig, TinyGPT  # noqa: E402

from bpe_tokenizer import BPETokenizer  # noqa: E402


def save_checkpoint(
    path: Path,
    model: TinyGPT,
    tokenizer: BPETokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    training_loss: float,
    validation_loss: float,
    stage: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    checkpoint: dict[str, Any] = {
        "version": 2,
        "stage": stage,
        "model_config": asdict(model.config),
        "tokenizer": tokenizer.to_string(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": training_loss,
        "validation_loss": validation_loss,
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    device: torch.device | str,
) -> tuple[TinyGPT, BPETokenizer, dict[str, Any]]:
    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    config = ModelConfig(**checkpoint["model_config"])
    tokenizer = BPETokenizer.from_string(checkpoint["tokenizer"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, checkpoint

