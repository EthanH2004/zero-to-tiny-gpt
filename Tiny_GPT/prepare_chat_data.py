"""Prepare English assistant dialogue from OpenAssistant OASST1."""

from argparse import ArgumentParser, Namespace
import gzip
from io import TextIOWrapper
import json
from pathlib import Path
import random
from typing import Any
from urllib.request import Request, urlopen

from settings import load_settings, project_path


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Prepare chat fine-tuning data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace dialogue files that already exist",
    )
    return parser.parse_args()


def normalize_message(text: str) -> str:
    return " ".join(text.split())


def read_english_messages(url: str) -> dict[str, dict[str, Any]]:
    request = Request(
        url,
        headers={"User-Agent": "Tiny-GPT-learning-project"},
    )
    messages: dict[str, dict[str, Any]] = {}

    print("Downloading the OpenAssistant message archive...")

    with urlopen(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as compressed_file:
            with TextIOWrapper(compressed_file, encoding="utf-8") as text_file:
                for line_number, line in enumerate(text_file, start=1):
                    message = json.loads(line)

                    if message.get("lang") == "en":
                        messages[message["message_id"]] = message

                    if line_number % 20000 == 0:
                        print(f"Read {line_number:,} messages...")

    return messages


def collect_dialogue_pairs(
    messages: dict[str, dict[str, Any]],
    allowed_characters: set[str],
    maximum_prompt_characters: int,
    maximum_response_characters: int,
) -> list[str]:
    pairs = []

    for message in messages.values():
        if message.get("role") != "assistant":
            continue

        if message.get("rank") not in (None, 0):
            continue

        parent = messages.get(message.get("parent_id"))

        if parent is None or parent.get("role") != "prompter":
            continue

        prompt = normalize_message(parent["text"])
        response = normalize_message(message["text"])

        if not prompt or not response:
            continue

        if len(prompt) > maximum_prompt_characters:
            continue

        if len(response) > maximum_response_characters:
            continue

        dialogue = f"User: {prompt}\nAssistant: {response}\n\n"

        if set(dialogue).issubset(allowed_characters):
            pairs.append(dialogue)

    return pairs


def write_dialogues(
    pairs: list[str],
    output_path: Path,
    target_characters: int,
    starting_index: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    written_characters = 0
    pair_index = starting_index

    with temporary_path.open("w", encoding="utf-8") as output_file:
        while pair_index < len(pairs):
            dialogue = pairs[pair_index]
            output_file.write(dialogue)
            written_characters += len(dialogue)
            pair_index += 1

            if written_characters >= target_characters:
                break

    temporary_path.replace(output_path)
    print(f"Prepared: {output_path}")
    print(f"Characters: {written_characters:,}")

    return pair_index


def prepare_chat_data() -> None:
    arguments = parse_arguments()
    settings = load_settings()
    chat_dataset = settings["chat_dataset"]
    train_output = project_path(chat_dataset["train_output"])
    validation_output = project_path(chat_dataset["validation_output"])

    if (
        train_output.exists()
        and validation_output.exists()
        and not arguments.force
    ):
        print("Chat fine-tuning data already exists.")
        return

    base_text_path = project_path(settings["training"]["data"])

    if not base_text_path.exists():
        raise FileNotFoundError(
            "Prepare the TinyStories base data before chat data"
        )

    allowed_characters = set(
        base_text_path.read_text(encoding="utf-8")
    )
    messages = read_english_messages(chat_dataset["url"])
    pairs = collect_dialogue_pairs(
        messages=messages,
        allowed_characters=allowed_characters,
        maximum_prompt_characters=(
            chat_dataset["maximum_prompt_characters"]
        ),
        maximum_response_characters=(
            chat_dataset["maximum_response_characters"]
        ),
    )

    random.Random(chat_dataset["seed"]).shuffle(pairs)
    print("Usable English prompt/response pairs:", f"{len(pairs):,}")

    next_index = write_dialogues(
        pairs=pairs,
        output_path=train_output,
        target_characters=chat_dataset["train_characters"],
        starting_index=0,
    )
    write_dialogues(
        pairs=pairs,
        output_path=validation_output,
        target_characters=chat_dataset["validation_characters"],
        starting_index=next_index,
    )


if __name__ == "__main__":
    prepare_chat_data()
