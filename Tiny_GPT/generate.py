"""Load a trained Tiny GPT checkpoint and generate text."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from tiny_gpt import get_default_device, load_checkpoint
from settings import load_settings, project_path


def parse_arguments() -> Namespace:
    settings = load_settings()
    generation = settings["generation"]
    parser = ArgumentParser(description="Generate text with Tiny GPT")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_path(generation["checkpoint"]),
        help="Trained model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=generation["prompt"],
        help="Text that the model will continue",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=generation["length"],
        help="Number of new characters to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=generation["temperature"],
        help="Lower is predictable; higher is random",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=generation["top_k"],
        help="Optionally sample from only the top K choices",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=generation["seed"],
        help="Random seed used during sampling",
    )

    return parser.parse_args()


def generate_text() -> None:
    arguments = parse_arguments()

    if not arguments.prompt:
        raise ValueError("The prompt cannot be empty")

    if arguments.length <= 0:
        raise ValueError("length must be greater than zero")

    torch.manual_seed(arguments.seed)
    device = get_default_device()

    model, tokenizer, checkpoint = load_checkpoint(
        arguments.checkpoint,
        device,
    )

    prompt_ids = tokenizer.encode(arguments.prompt)
    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_ids = model.generate(
        token_ids=token_ids,
        max_new_tokens=arguments.length,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    print("\n=== TINY GPT GENERATION ===")
    print("Device:", device)
    print("Checkpoint:", arguments.checkpoint)
    print("Training step:", checkpoint.get("step", "unknown"))
    print("Saved loss:", checkpoint.get("loss", "unknown"))
    print("Temperature:", arguments.temperature)
    print("\n=== GENERATED TEXT ===")
    print(generated_text)


if __name__ == "__main__":
    generate_text()
