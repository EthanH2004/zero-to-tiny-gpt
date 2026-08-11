"""Download a reproducible educational-English FineWeb-Edu subset."""

from argparse import ArgumentParser
import json
from pathlib import Path
import random
from time import sleep
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from v2_settings import load_settings, v2_path


DATASET_API = "https://datasets-server.huggingface.co/rows"
ROWS_PER_REQUEST = 100


def request_rows(
    dataset_name: str,
    dataset_config: str,
    split: str,
    offset: int,
) -> dict[str, Any]:
    query = urlencode(
        {
            "dataset": dataset_name,
            "config": dataset_config,
            "split": split,
            "offset": offset,
            "length": ROWS_PER_REQUEST,
        }
    )
    request = Request(
        f"{DATASET_API}?{query}",
        headers={"User-Agent": "Tiny-GPT-V2-learning-project"},
    )

    for attempt in range(8):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt == 7:
                raise

            retry_after = error.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after)
                if retry_after is not None
                else min(5 * (attempt + 1), 30)
            )
            print(f"Dataset server busy; retrying in {wait_seconds:.0f}s")
            sleep(wait_seconds)
        except Exception:
            if attempt == 7:
                raise

            sleep(min(2**attempt, 30))

    raise RuntimeError("Dataset request unexpectedly failed")


def clean_document(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines).strip()


def download_split(
    dataset: dict[str, Any],
    output_path: Path,
    target_characters: int,
    seed: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"Already exists: {output_path}")
        return

    first_page = request_rows(
        dataset["name"],
        dataset["config"],
        dataset["split"],
        0,
    )
    total_rows = int(first_page["num_rows_total"])
    possible_offsets = range(0, total_rows, ROWS_PER_REQUEST)
    random_offsets = random.Random(seed).sample(
        possible_offsets,
        k=len(possible_offsets),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    downloaded_characters = 0
    downloaded_documents = 0
    next_report = 10_000_000

    with temporary_path.open("w", encoding="utf-8") as output_file:
        for offset in random_offsets:
            page = request_rows(
                dataset["name"],
                dataset["config"],
                dataset["split"],
                offset,
            )

            for item in page["rows"]:
                document = clean_document(str(item["row"]["text"]))

                if len(document) < 200:
                    continue

                output_file.write(document)
                output_file.write("\n\n")
                downloaded_characters += len(document) + 2
                downloaded_documents += 1

            if downloaded_characters >= next_report:
                print(
                    f"{downloaded_characters:,} / "
                    f"{target_characters:,} characters"
                )
                next_report += 10_000_000

            if downloaded_characters >= target_characters:
                break

    temporary_path.replace(output_path)
    print(f"Prepared: {output_path}")
    print(f"Documents: {downloaded_documents:,}")
    print(f"Characters: {downloaded_characters:,}")


def prepare_base_data(force: bool = False) -> None:
    dataset = load_settings()["base_dataset"]

    print("\n=== PREPARING V2 BASE DATA ===")
    print("Dataset:", dataset["name"])
    print("Configuration:", dataset["config"])

    download_split(
        dataset=dataset,
        output_path=v2_path(dataset["train_output"]),
        target_characters=dataset["train_characters"],
        seed=dataset["seed"],
        force=force,
    )
    download_split(
        dataset=dataset,
        output_path=v2_path(dataset["validation_output"]),
        target_characters=dataset["validation_characters"],
        seed=dataset["seed"] + 1,
        force=force,
    )


def main() -> None:
    parser = ArgumentParser(description="Prepare V2 FineWeb-Edu data")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    prepare_base_data(force=arguments.force)


if __name__ == "__main__":
    main()

