"""Prepare complete English conversation paths from OpenAssistant OASST1."""

from argparse import ArgumentParser
import gzip
from io import TextIOWrapper
import json
from pathlib import Path
import random
from typing import Any
from urllib.request import Request, urlopen

from v2_settings import load_settings, v2_path


Message = dict[str, Any]


def normalize_message(text: str) -> str:
    return " ".join(text.split())


def valid_message(message: Message, maximum_characters: int) -> bool:
    text = normalize_message(str(message.get("text", "")))
    return (
        message.get("lang") == "en"
        and not message.get("deleted", False)
        and message.get("review_result") is not False
        and 0 < len(text) <= maximum_characters
    )


def reply_order(message: Message) -> tuple[int, str]:
    rank = message.get("rank")
    return (
        int(rank) if rank is not None else 1_000_000,
        str(message.get("message_id", "")),
    )


def collect_paths(
    message: Message,
    maximum_message_characters: int,
    maximum_paths: int,
) -> list[list[Message]]:
    if not valid_message(message, maximum_message_characters):
        return []

    valid_replies = [
        reply
        for reply in message.get("replies", [])
        if valid_message(reply, maximum_message_characters)
    ]
    valid_replies.sort(key=reply_order)

    if not valid_replies:
        return [[message]]

    paths: list[list[Message]] = []

    for reply in valid_replies:
        for reply_path in collect_paths(
            reply,
            maximum_message_characters,
            maximum_paths,
        ):
            paths.append([message, *reply_path])

            if len(paths) >= maximum_paths:
                return paths

    return paths


def format_conversation(messages: list[Message]) -> str:
    if len(messages) < 2:
        return ""

    parts = []

    for message in messages:
        role = "User" if message["role"] == "prompter" else "Assistant"
        parts.append(f"{role}: {normalize_message(message['text'])}")

    return "\n".join(parts) + "\n\n"


def download_conversation_groups(
    dataset: dict[str, Any],
) -> list[list[str]]:
    request = Request(
        dataset["url"],
        headers={"User-Agent": "Tiny-GPT-V2-learning-project"},
    )
    conversation_groups = []
    seen_conversations = set()
    conversation_count = 0

    print("Downloading OpenAssistant conversation trees...")

    with urlopen(request, timeout=180) as response:
        with gzip.GzipFile(fileobj=response) as compressed_file:
            with TextIOWrapper(compressed_file, encoding="utf-8") as file:
                for tree_number, line in enumerate(file, start=1):
                    tree = json.loads(line)
                    prompt = tree.get("prompt")

                    if prompt is None:
                        continue

                    paths = collect_paths(
                        prompt,
                        dataset["maximum_message_characters"],
                        dataset["maximum_paths_per_tree"],
                    )
                    tree_conversations = []

                    for path in paths:
                        conversation = format_conversation(path)

                        if conversation and conversation not in seen_conversations:
                            tree_conversations.append(conversation)
                            seen_conversations.add(conversation)

                    if tree_conversations:
                        conversation_groups.append(tree_conversations)
                        conversation_count += len(tree_conversations)

                    if tree_number % 2000 == 0:
                        print(
                            f"Read {tree_number:,} trees; "
                            f"kept {conversation_count:,} conversations"
                        )

    return conversation_groups


def write_conversation_groups(
    conversation_groups: list[list[str]],
    output_path: Path,
    target_characters: int,
    starting_index: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    written_characters = 0
    group_index = starting_index

    with temporary_path.open("w", encoding="utf-8") as output_file:
        while group_index < len(conversation_groups):
            for conversation in conversation_groups[group_index]:
                output_file.write(conversation)
                written_characters += len(conversation)

            group_index += 1

            if written_characters >= target_characters:
                break

    temporary_path.replace(output_path)
    print(f"Prepared: {output_path}")
    print(f"Characters: {written_characters:,}")

    return group_index


def prepare_chat_data(force: bool = False) -> None:
    dataset = load_settings()["chat_dataset"]
    train_output = v2_path(dataset["train_output"])
    validation_output = v2_path(dataset["validation_output"])

    if train_output.exists() and validation_output.exists() and not force:
        print("V2 chat data already exists.")
        return

    print("\n=== PREPARING V2 MULTI-TURN DIALOGUE ===")
    conversation_groups = download_conversation_groups(dataset)
    random.Random(dataset["seed"]).shuffle(conversation_groups)
    conversation_count = sum(len(group) for group in conversation_groups)
    total_characters = sum(
        len(conversation)
        for group in conversation_groups
        for conversation in group
    )

    print("Conversation trees:", f"{len(conversation_groups):,}")
    print("Unique conversations:", f"{conversation_count:,}")
    print("Available characters:", f"{total_characters:,}")

    validation_end = write_conversation_groups(
        conversation_groups=conversation_groups,
        output_path=validation_output,
        target_characters=dataset["validation_characters"],
        starting_index=0,
    )
    write_conversation_groups(
        conversation_groups=conversation_groups,
        output_path=train_output,
        target_characters=dataset["train_characters"],
        starting_index=validation_end,
    )


def main() -> None:
    parser = ArgumentParser(description="Prepare V2 multi-turn chat data")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    prepare_chat_data(force=arguments.force)


if __name__ == "__main__":
    main()
