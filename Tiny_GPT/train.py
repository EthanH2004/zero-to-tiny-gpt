"""Train or fine-tune Tiny GPT and save a checkpoint."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter

import torch

from tiny_gpt import (
    CharacterTokenizer,
    ModelConfig,
    TinyGPT,
    get_default_device,
    load_checkpoint,
    save_checkpoint,
)
from settings import load_settings, project_path


def parse_arguments() -> Namespace:
    settings = load_settings()
    model = settings["model"]

    profile_parser = ArgumentParser(add_help=False)
    profile_parser.add_argument("--fine-tune", action="store_true")
    selected_profile, _ = profile_parser.parse_known_args()

    if selected_profile.fine_tune:
        training = settings["fine_tuning"]
    else:
        training = settings["training"]

    parser = ArgumentParser(description="Train Tiny GPT")

    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Continue from the base checkpoint using dialogue data",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=project_path(training["data"]),
        help="Text file used for training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_path(training["output"]),
        help="Where the trained checkpoint will be saved",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=project_path(training["validation_data"]),
        help="Unseen text used to measure validation loss",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=(
            project_path(training["resume"])
            if "resume" in training
            else None
        ),
        help="Optional checkpoint to continue training",
    )

    parser.add_argument("--steps", type=int, default=training["steps"])
    parser.add_argument(
        "--batch-size", type=int, default=training["batch_size"]
    )
    parser.add_argument(
        "--learning-rate", type=float, default=training["learning_rate"]
    )
    parser.add_argument(
        "--log-every", type=int, default=training["log_every"]
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=training["evaluate_every"],
    )
    parser.add_argument(
        "--evaluation-batches",
        type=int,
        default=training["evaluation_batches"],
    )
    parser.add_argument("--seed", type=int, default=training["seed"])

    parser.add_argument(
        "--context-length", type=int, default=model["context_length"]
    )
    parser.add_argument(
        "--embedding-size", type=int, default=model["embedding_size"]
    )
    parser.add_argument(
        "--heads", type=int, default=model["number_of_heads"]
    )
    parser.add_argument(
        "--layers", type=int, default=model["number_of_layers"]
    )
    parser.add_argument("--dropout", type=float, default=model["dropout"])

    return parser.parse_args()


def read_training_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if not text:
        raise ValueError("The training file is empty")

    return text


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"

    return f"{minutes:d}m {remaining_seconds:02d}s"


def format_progress(
    completed_steps: int,
    total_steps: int,
    elapsed_seconds: float,
) -> str:
    progress = completed_steps / total_steps
    bar_width = 24
    filled_width = min(bar_width, int(progress * bar_width))
    progress_bar = "#" * filled_width + "-" * (bar_width - filled_width)
    steps_per_second = completed_steps / max(elapsed_seconds, 0.001)
    remaining_steps = total_steps - completed_steps
    remaining_seconds = remaining_steps / max(steps_per_second, 0.001)

    return (
        f"[{progress_bar}] {progress:6.2%}  "
        f"{steps_per_second:5.1f} steps/s  "
        f"ETA {format_duration(remaining_seconds)}"
    )


def create_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum_start = len(token_ids) - context_length

    if maximum_start <= 0:
        raise ValueError(
            "Training text must be longer than the context length"
        )

    start_positions = torch.randint(
        0,
        maximum_start,
        (batch_size,),
    )

    inputs = torch.stack(
        [
            token_ids[start : start + context_length]
            for start in start_positions
        ]
    )
    targets = torch.stack(
        [
            token_ids[start + 1 : start + context_length + 1]
            for start in start_positions
        ]
    )

    return inputs.to(device), targets.to(device)


def create_new_model(
    text: str,
    arguments: Namespace,
    device: torch.device,
) -> tuple[TinyGPT, CharacterTokenizer]:
    tokenizer = CharacterTokenizer.from_text(text)

    config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=arguments.context_length,
        embedding_size=arguments.embedding_size,
        number_of_heads=arguments.heads,
        number_of_layers=arguments.layers,
        dropout=arguments.dropout,
    )

    model = TinyGPT(config).to(device)
    return model, tokenizer


@torch.no_grad()
def calculate_validation_loss(
    model: TinyGPT,
    validation_ids: torch.Tensor,
    batch_size: int,
    number_of_batches: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses = []

    for _ in range(number_of_batches):
        inputs, targets = create_batch(
            validation_ids,
            batch_size,
            model.config.context_length,
            device,
        )
        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError("The model did not calculate validation loss")

        losses.append(loss.detach())

    if was_training:
        model.train()

    return torch.stack(losses).mean().item()


def train() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)

    device = get_default_device()
    text = read_training_text(arguments.data)
    validation_text = read_training_text(arguments.validation_data)

    if arguments.resume is None:
        model, tokenizer = create_new_model(text, arguments, device)
        starting_step = 0
        checkpoint = None
    else:
        model, tokenizer, checkpoint = load_checkpoint(
            arguments.resume,
            device,
        )
        starting_step = int(checkpoint.get("step", 0))

    encoded_text = tokenizer.encode(text)
    encoded_validation_text = tokenizer.encode(validation_text)
    token_ids = torch.tensor(encoded_text, dtype=torch.long)
    validation_ids = torch.tensor(
        encoded_validation_text,
        dtype=torch.long,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
    )

    if checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Keep Adam's learned momentum, but use this run's configured rate.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = arguments.learning_rate

    print("\n=== TINY GPT TRAINING ===")
    print("Mode:", "chat fine-tuning" if arguments.fine_tune else "base")
    print("Device:", device)
    print("Characters:", len(text))
    print("Validation characters:", len(validation_text))
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Parameters:", f"{model.number_of_parameters():,}")
    print("Starting step:", starting_step)

    model.train()
    latest_loss = 0.0
    training_started = perf_counter()

    if arguments.fine_tune:
        target_step = starting_step + arguments.steps
    else:
        target_step = arguments.steps

    total_steps = target_step - starting_step

    if total_steps <= 0:
        raise ValueError(
            "steps must be greater than the checkpoint's saved step"
        )

    for step in range(starting_step, target_step):
        inputs, targets = create_batch(
            token_ids,
            arguments.batch_size,
            model.config.context_length,
            device,
        )

        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError("The model did not calculate a loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        latest_loss = loss.detach().item()

        completed_steps = step - starting_step + 1

        if step % arguments.log_every == 0 or step + 1 == target_step:
            progress = format_progress(
                completed_steps=completed_steps,
                total_steps=total_steps,
                elapsed_seconds=perf_counter() - training_started,
            )
            print(
                f"Step: {step + 1:5d}/{target_step}  "
                f"Loss: {latest_loss:.4f}  {progress}"
            )

        if step % arguments.evaluate_every == 0:
            validation_loss = calculate_validation_loss(
                model=model,
                validation_ids=validation_ids,
                batch_size=arguments.batch_size,
                number_of_batches=arguments.evaluation_batches,
                device=device,
            )
            print(
                f"             Validation loss: {validation_loss:.4f}"
            )

    save_checkpoint(
        path=arguments.output,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        step=target_step,
        loss=latest_loss,
    )

    print("Training complete")
    print("Checkpoint:", arguments.output)
    print("Final loss:", f"{latest_loss:.4f}")
    print(
        "Training time:",
        format_duration(perf_counter() - training_started),
    )


if __name__ == "__main__":
    train()
