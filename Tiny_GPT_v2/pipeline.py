"""Run the complete Tiny GPT V2 overnight workflow unattended."""

from argparse import ArgumentParser
import os
import subprocess
import sys
from time import perf_counter

from v2_settings import V2_DIRECTORY, load_settings, v2_path


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"


def run_step(name: str, script: str, arguments: tuple[str, ...] = ()) -> None:
    print(f"\n{'=' * 68}", flush=True)
    print(name, flush=True)
    print(f"{'=' * 68}", flush=True)
    command = [
        sys.executable,
        str(V2_DIRECTORY / script),
        *arguments,
    ]
    subprocess.run(command, cwd=V2_DIRECTORY, check=True)


def pipeline(force: bool = False, smoke_test: bool = False) -> None:
    started = perf_counter()
    keep_awake_process = None

    if load_settings()["pipeline"]["keep_awake"]:
        try:
            keep_awake_process = subprocess.Popen(
                ["caffeinate", "-dimsu", "-w", str(os.getpid())]
            )
            print(
                "Sleep prevention: active until this pipeline finishes",
                flush=True,
            )
        except (FileNotFoundError, OSError):
            print("Sleep prevention unavailable; keep the Mac awake manually")

    force_argument = ("--force",) if force else ()
    training_arguments = ("--smoke-test",) if smoke_test else force_argument

    try:
        run_step(
            "1/5  Download 100M characters of educational English",
            "prepare_base_data.py",
            force_argument,
        )
        run_step(
            "2/5  Prepare up to 15M characters of multi-turn dialogue",
            "prepare_chat_data.py",
            force_argument,
        )
        run_step(
            "3/5  Train the byte-level BPE tokenizer",
            "train_tokenizer.py",
            force_argument,
        )
        run_step(
            "4/5  Train the V2 base model",
            "train.py",
            ("--stage", "base", *training_arguments),
        )
        run_step(
            "5/5  Fine-tune V2 on multi-turn dialogue",
            "train.py",
            ("--stage", "chat", *training_arguments),
        )
    finally:
        if keep_awake_process is not None:
            keep_awake_process.terminate()

    print("\n=== V2 OVERNIGHT PIPELINE COMPLETE ===")
    print("Elapsed:", format_duration(perf_counter() - started))
    if smoke_test:
        print("Smoke test passed; permanent checkpoints were untouched.")
    else:
        print("Chat checkpoint:", v2_path("checkpoints/chat_best.pt"))
        print("Start it with option 8 in the main menu.")


def main() -> None:
    parser = ArgumentParser(description="Run the complete V2 pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate data and replace completed V2 checkpoints",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Verify the complete pipeline with two-step temporary training",
    )
    arguments = parser.parse_args()
    pipeline(
        force=arguments.force,
        smoke_test=arguments.smoke_test,
    )


if __name__ == "__main__":
    main()
