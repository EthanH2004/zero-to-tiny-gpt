"""Generate text from the best V2 base checkpoint."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import torch


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent / "Tiny_GPT"

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from tiny_gpt import get_default_device  # noqa: E402

from v2_checkpoint import load_checkpoint  # noqa: E402
from v2_settings import v2_path  # noqa: E402


def main() -> None:
    parser = ArgumentParser(description="Generate text with Tiny GPT V2")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    arguments = parser.parse_args()
    checkpoint_path = v2_path("checkpoints/base_best.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError("Run V2 base training first")

    device = get_default_device()
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path, device)
    prompt_ids = tokenizer.encode(arguments.prompt)
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = model.generate(
        token_ids,
        max_new_tokens=arguments.tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
    )

    print("\n=== TINY GPT V2 GENERATION ===")
    print("Validation loss:", checkpoint.get("validation_loss"))
    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()

