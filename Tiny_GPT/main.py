"""Simple menu for preparing, training, generating, and chatting."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from settings import load_settings, project_path


PROJECT_DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Action:
    name: str
    script: str
    arguments: tuple[str, ...] = ()


ACTIONS = {
    "1": Action("Prepare TinyStories data", "prepare_data.py"),
    "2": Action("Train the base model", "train.py"),
    "3": Action("Generate a story", "generate.py"),
    "4": Action("Prepare assistant dialogue", "prepare_chat_data.py"),
    "5": Action("Fine-tune for conversation", "train.py", ("--fine-tune",)),
    "6": Action("Start the chatbot", "chatbot.py"),
    "7": Action("Run complete V2 overnight pipeline", "../Tiny_GPT_v2/pipeline.py"),
    "8": Action("Start the V2 chatbot", "../Tiny_GPT_v2/chatbot.py"),
}


def yes_or_no(question: str) -> bool:
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def run_action(action: Action) -> None:
    script_path = PROJECT_DIRECTORY / action.script
    command = [sys.executable, str(script_path), *action.arguments]

    print(f"\nRunning: {action.name}")
    print("-" * 60)

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIRECTORY,
            check=False,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        return

    if result.returncode == 0:
        print(f"\nFinished: {action.name}")
    else:
        print(f"\nStopped with exit code {result.returncode}.")


def status_lines() -> list[str]:
    settings = load_settings()
    paths = {
        "TinyStories data": project_path(settings["training"]["data"]),
        "Base checkpoint": project_path(settings["training"]["output"]),
        "Dialogue data": project_path(settings["fine_tuning"]["data"]),
        "Chat checkpoint": project_path(settings["fine_tuning"]["output"]),
        "V2 base checkpoint": (
            PROJECT_DIRECTORY / "../Tiny_GPT_v2/checkpoints/base_best.pt"
        ),
        "V2 chat checkpoint": (
            PROJECT_DIRECTORY / "../Tiny_GPT_v2/checkpoints/chat_best.pt"
        ),
    }

    return [
        f"  {'READY' if path.exists() else 'MISSING':7}  {name}"
        for name, path in paths.items()
    ]


def main() -> None:
    while True:
        print("\n=== TINY GPT CONTROL CENTER ===")
        print("\nProject status:")
        print("\n".join(status_lines()))
        print("\nActions:")

        for number, action in ACTIONS.items():
            print(f"  {number}. {action.name}")

        print("  q. Quit")
        choice = input("\nChoose an action: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            print("Goodbye!")
            return

        action = ACTIONS.get(choice)

        if action is None:
            print("Please choose 1 through 8, or q.")
            continue

        if choice == "2":
            base_checkpoint = project_path(
                load_settings()["training"]["output"]
            )

            if base_checkpoint.exists() and not yes_or_no(
                "A base checkpoint already exists. Retrain and replace it?"
            ):
                continue

        if choice == "7" and not yes_or_no(
            "Start the unattended V2 pipeline (approximately 7.5–8 hours)?"
        ):
            continue

        run_action(action)


if __name__ == "__main__":
    main()
