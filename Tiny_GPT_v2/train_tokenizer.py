"""Train the V2 byte-level BPE tokenizer on base and dialogue text."""

from argparse import ArgumentParser

from bpe_tokenizer import BPETokenizer
from v2_settings import load_settings, v2_path


def train_tokenizer(force: bool = False) -> None:
    settings = load_settings()
    tokenizer_settings = settings["tokenizer"]
    output_path = v2_path(tokenizer_settings["output"])

    if output_path.exists() and not force:
        print(f"Tokenizer already exists: {output_path}")
        return

    training_files = [
        v2_path(settings["base_dataset"]["train_output"]),
        v2_path(settings["chat_dataset"]["train_output"]),
    ]

    for training_file in training_files:
        if not training_file.exists():
            raise FileNotFoundError(
                f"Prepare the data before training the tokenizer: "
                f"{training_file}"
            )

    print("\n=== TRAINING BYTE-LEVEL BPE TOKENIZER ===")
    tokenizer = BPETokenizer.train(
        files=training_files,
        vocabulary_size=tokenizer_settings["vocabulary_size"],
        minimum_frequency=tokenizer_settings["minimum_frequency"],
    )
    tokenizer.save(output_path)

    example = "User: Hello! How are you?\nAssistant: I am doing well."
    token_ids = tokenizer.encode(example)

    print("Tokenizer:", output_path)
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Example characters:", len(example))
    print("Example tokens:", len(token_ids))
    print("Decoded correctly:", tokenizer.decode(token_ids) == example)


def main() -> None:
    parser = ArgumentParser(description="Train the V2 BPE tokenizer")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    train_tokenizer(force=arguments.force)


if __name__ == "__main__":
    main()

