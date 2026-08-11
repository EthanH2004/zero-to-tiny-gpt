"""Run an interactive conversation with a chat-trained Tiny GPT."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from settings import load_settings, project_path
from tiny_gpt import get_default_device, load_checkpoint


def parse_arguments() -> Namespace:
    settings = load_settings()["chatbot"]
    parser = ArgumentParser(description="Chat with Tiny GPT")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_path(settings["checkpoint"]),
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=settings["response_length"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings["temperature"],
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings["top_k"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings["seed"],
    )

    return parser.parse_args()


def extract_response(generated_text: str, prompt_length: int) -> str:
    response = generated_text[prompt_length:]

    for stop_sequence in ("\nUser:", "\n\n"):
        if stop_sequence in response:
            response = response.split(stop_sequence, maxsplit=1)[0]

    return response.strip()


def chat() -> None:
    arguments = parse_arguments()

    if not arguments.checkpoint.exists():
        raise FileNotFoundError(
            f"Chat checkpoint not found: {arguments.checkpoint}\n"
            "Fine-tune the base model before starting the chatbot."
        )

    torch.manual_seed(arguments.seed)
    device = get_default_device()
    model, tokenizer, checkpoint = load_checkpoint(
        arguments.checkpoint,
        device,
    )

    history = ""

    print("\n=== TINY GPT CHATBOT ===")
    print("Device:", device)
    print("Checkpoint:", arguments.checkpoint)
    print("Training step:", checkpoint.get("step", "unknown"))
    print("Commands: /clear resets history, /quit exits\n")

    while True:
        user_message = input("You: ").strip()

        if not user_message:
            continue

        if user_message.lower() == "/quit":
            print("Tiny GPT: Goodbye!")
            break

        if user_message.lower() == "/clear":
            history = ""
            print("Conversation history cleared.\n")
            continue

        prompt = f"{history}User: {user_message}\nAssistant:"

        try:
            prompt_ids = tokenizer.encode(prompt)
        except ValueError as error:
            print(f"Tiny GPT: {error}\n")
            continue

        token_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        generated_ids = model.generate(
            token_ids=token_ids,
            max_new_tokens=arguments.response_length,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
        )

        generated_text = tokenizer.decode(
            generated_ids[0].tolist()
        )
        response = extract_response(
            generated_text,
            prompt_length=len(prompt),
        )

        if not response:
            response = "..."

        print(f"Tiny GPT: {response}\n")
        history = f"{prompt} {response}\n"


if __name__ == "__main__":
    chat()
