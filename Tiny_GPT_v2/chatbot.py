"""Chat interactively with the best dialogue-trained Tiny GPT V2."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import torch


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent / "Tiny_GPT"

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from tiny_gpt import get_default_device  # noqa: E402

from v2_checkpoint import load_checkpoint  # noqa: E402
from v2_settings import load_settings, v2_path  # noqa: E402


def extract_response(text: str) -> str:
    for stop_sequence in ("\nUser:", "\n\n"):
        if stop_sequence in text:
            text = text.split(stop_sequence, maxsplit=1)[0]

    return text.strip()


def chat() -> None:
    parser = ArgumentParser(description="Chat with Tiny GPT V2")
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    settings = load_settings()["chatbot"]
    checkpoint_path = (
        arguments.checkpoint
        if arguments.checkpoint is not None
        else v2_path(settings["checkpoint"])
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"V2 chat checkpoint not found: {checkpoint_path}\n"
            "Run the V2 overnight pipeline first."
        )

    torch.manual_seed(settings["seed"])
    device = get_default_device()
    model, tokenizer, checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )
    history = ""

    print("\n=== TINY GPT V2 CHATBOT ===")
    print("Device:", device)
    print("Parameters:", f"{model.number_of_parameters():,}")
    print("Tokenizer vocabulary:", tokenizer.vocabulary_size)
    print("Best validation loss:", checkpoint.get("validation_loss"))
    print("Commands: /clear resets history, /quit exits\n")

    while True:
        user_message = input("You: ").strip()

        if not user_message:
            continue

        if user_message.lower() == "/quit":
            print("Tiny GPT V2: Goodbye!")
            return

        if user_message.lower() == "/clear":
            history = ""
            print("Conversation history cleared.\n")
            continue

        prompt = f"{history}User: {user_message}\nAssistant:"
        prompt_ids = tokenizer.encode(prompt)
        token_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )
        generated_ids = model.generate(
            token_ids=token_ids,
            max_new_tokens=settings["response_tokens"],
            temperature=settings["temperature"],
            top_k=settings["top_k"],
        )
        response_ids = generated_ids[0, len(prompt_ids) :].tolist()
        response = extract_response(tokenizer.decode(response_ids))

        if not response:
            response = "..."

        print(f"Tiny GPT V2: {response}\n")
        history = f"{prompt} {response}\n"


if __name__ == "__main__":
    chat()

