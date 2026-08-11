"""Download reproducible TinyStories train and validation subsets."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import random
from time import sleep
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from settings import load_settings, project_path


DATASET_API = "https://datasets-server.huggingface.co/rows"
ROWS_PER_REQUEST = 100
REQUEST_DELAY_SECONDS = 1.0


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Prepare TinyStories data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace data files that already exist",
    )
    return parser.parse_args()


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
        headers={"User-Agent": "Tiny-GPT-learning-project"},
    )

    for attempt in range(8):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt == 7:
                raise

            retry_after = error.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after)
                if retry_after is not None
                else min(10 * (attempt + 1), 30)
            )
            print(
                "Dataset server is busy; retrying in "
                f"{wait_seconds:.0f} seconds..."
            )
            sleep(wait_seconds)
        except Exception:
            if attempt == 7:
                raise

            sleep(min(2 ** attempt, 30))

    raise RuntimeError("Dataset request unexpectedly failed")


def download_split(
    dataset_name: str,
    dataset_config: str,
    split: str,
    output_path: Path,
    target_characters: int,
    seed: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"Already exists: {output_path}")
        return

    first_page = request_rows(
        dataset_name,
        dataset_config,
        split,
        offset=0,
    )
    total_rows = int(first_page["num_rows_total"])

    offsets = list(range(0, total_rows, ROWS_PER_REQUEST))
    random.Random(seed).shuffle(offsets)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    downloaded_characters = 0
    downloaded_stories = 0
    request_count = 0

    with temporary_path.open("w", encoding="utf-8") as output_file:
        for offset in offsets:
            page = request_rows(
                dataset_name,
                dataset_config,
                split,
                offset,
            )
            request_count += 1
            sleep(REQUEST_DELAY_SECONDS)

            for item in page["rows"]:
                story = str(item["row"]["text"]).strip()

                if not story:
                    continue

                output_file.write(story)
                output_file.write("\n\n")
                downloaded_characters += len(story) + 2
                downloaded_stories += 1

            if request_count % 10 == 0:
                print(
                    f"{split}: {downloaded_characters:,} / "
                    f"{target_characters:,} characters"
                )

            if downloaded_characters >= target_characters:
                break

    temporary_path.replace(output_path)

    print(f"Prepared: {output_path}")
    print(f"Stories: {downloaded_stories:,}")
    print(f"Characters: {downloaded_characters:,}")


def prepare_data() -> None:
    arguments = parse_arguments()
    settings = load_settings()
    dataset = settings["dataset"]

    print("\n=== PREPARING TINYSTORIES ===")
    print("Dataset:", dataset["name"])

    download_split(
        dataset_name=dataset["name"],
        dataset_config=dataset["config"],
        split="train",
        output_path=project_path(dataset["train_output"]),
        target_characters=dataset["train_characters"],
        seed=dataset["seed"],
        force=arguments.force,
    )
    download_split(
        dataset_name=dataset["name"],
        dataset_config=dataset["config"],
        split="validation",
        output_path=project_path(dataset["validation_output"]),
        target_characters=dataset["validation_characters"],
        seed=dataset["seed"] + 1,
        force=arguments.force,
    )


if __name__ == "__main__":
    prepare_data()
