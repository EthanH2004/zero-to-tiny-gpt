"""Time-budgeted base training and dialogue fine-tuning for Tiny GPT V2."""

from argparse import ArgumentParser, Namespace
import math
from pathlib import Path
import tempfile
from time import perf_counter
import sys
from typing import Any

import torch


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent / "Tiny_GPT"

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from tiny_gpt import ModelConfig, TinyGPT, get_default_device  # noqa: E402

from bpe_tokenizer import BPETokenizer  # noqa: E402
from v2_checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from v2_settings import load_settings, v2_path  # noqa: E402


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Train Tiny GPT V2")
    parser.add_argument("--stage", choices=("base", "chat"), required=True)
    parser.add_argument(
        "--minutes",
        type=float,
        help="Override this stage's wall-clock training budget",
    )
    parser.add_argument(
        "--maximum-steps",
        type=int,
        help="Override the safety ceiling on training steps",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this stage's completed best checkpoint",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two steps on tiny data and write only to a temporary folder",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"

    return f"{minutes:d}m {remaining_seconds:02d}s"


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    text = path.read_text(encoding="utf-8")

    if not text:
        raise ValueError(f"Training data is empty: {path}")

    return text


def encode_text(tokenizer: BPETokenizer, text: str) -> torch.Tensor:
    token_ids = tokenizer.encode(text)
    return torch.tensor(token_ids, dtype=torch.long)


def create_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum_start = len(token_ids) - context_length - 1

    if maximum_start <= 0:
        raise ValueError("The data is shorter than the context length")

    starts = torch.randint(0, maximum_start, (batch_size, 1))
    offsets = torch.arange(context_length).unsqueeze(0)
    positions = starts + offsets
    inputs = token_ids[positions]
    targets = token_ids[positions + 1]
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def validation_loss(
    model: TinyGPT,
    validation_ids: torch.Tensor,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    for _ in range(batches):
        inputs, targets = create_batch(
            validation_ids,
            batch_size,
            model.config.context_length,
            device,
        )
        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError("Validation loss was not calculated")

        losses.append(loss.detach())

    model.train()
    return torch.stack(losses).mean().item()


def scheduled_learning_rate(
    elapsed_fraction: float,
    maximum_rate: float,
    minimum_rate: float,
    warmup_fraction: float,
) -> float:
    if elapsed_fraction < warmup_fraction:
        warmup_progress = elapsed_fraction / warmup_fraction
        return maximum_rate * max(warmup_progress, 0.1)

    decay_progress = (
        (elapsed_fraction - warmup_fraction)
        / (1.0 - warmup_fraction)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return minimum_rate + (maximum_rate - minimum_rate) * cosine


def create_base_model(
    tokenizer: BPETokenizer,
    model_settings: dict[str, Any],
    device: torch.device,
) -> TinyGPT:
    config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=model_settings["context_length"],
        embedding_size=model_settings["embedding_size"],
        number_of_heads=model_settings["number_of_heads"],
        number_of_layers=model_settings["number_of_layers"],
        dropout=model_settings["dropout"],
    )
    return TinyGPT(config).to(device)


def load_stage_model(
    stage: str,
    settings: dict[str, Any],
    device: torch.device,
) -> tuple[TinyGPT, BPETokenizer]:
    if stage == "base":
        tokenizer = BPETokenizer.load(
            v2_path(settings["tokenizer"]["output"])
        )
        model = create_base_model(tokenizer, settings["model"], device)
        return model, tokenizer

    resume_path = v2_path(settings["chat_training"]["resume"])

    if not resume_path.exists():
        raise FileNotFoundError(
            f"Base checkpoint required before chat fine-tuning: {resume_path}"
        )

    model, tokenizer, _ = load_checkpoint(resume_path, device)
    return model, tokenizer


def train_stage(
    stage: str,
    minutes_override: float | None = None,
    maximum_steps_override: int | None = None,
    force: bool = False,
    smoke_test: bool = False,
) -> None:
    settings = load_settings()
    section_name = "base_training" if stage == "base" else "chat_training"
    training = dict(settings[section_name])

    if smoke_test:
        smoke_directory = (
            Path(tempfile.gettempdir()) / "tiny-gpt-v2-smoke"
        )
        if stage == "chat":
            settings["chat_training"]["resume"] = str(
                smoke_directory / "base_best.pt"
            )
        output_path = smoke_directory / f"{stage}_best.pt"
        latest_path = smoke_directory / f"{stage}_latest.pt"
        completion_path = smoke_directory / f"{stage}_complete.txt"
        force = True
        minutes_override = 1
        maximum_steps_override = 2
        training["log_every"] = 1
        training["evaluate_every"] = 1
        training["evaluation_batches"] = 1
        training["checkpoint_every"] = 2
    else:
        output_path = v2_path(training["output"])
        latest_path = v2_path(training["latest"])
        completion_path = v2_path(training["complete"])

    if completion_path.exists() and not force:
        print(f"Completed checkpoint already exists: {output_path}")
        return

    if force:
        for generated_path in (
            output_path,
            latest_path,
            completion_path,
        ):
            generated_path.unlink(missing_ok=True)

    time_minutes = (
        minutes_override
        if minutes_override is not None
        else training["time_minutes"]
    )
    maximum_steps = (
        maximum_steps_override
        if maximum_steps_override is not None
        else training["maximum_steps"]
    )

    if time_minutes <= 0 or maximum_steps <= 0:
        raise ValueError("Training time and maximum steps must be positive")

    torch.manual_seed(training["seed"])
    device = get_default_device()

    if latest_path.exists():
        print("Resuming recovery checkpoint:", latest_path)
        model, tokenizer, recovery_checkpoint = load_checkpoint(
            latest_path,
            device,
        )
        starting_step = int(recovery_checkpoint.get("step", 0))
    else:
        model, tokenizer = load_stage_model(stage, settings, device)
        recovery_checkpoint = None
        starting_step = 0

    print("\nEncoding training text...")
    training_text = read_text(v2_path(training["data"]))
    if smoke_test:
        training_text = training_text[:100_000]
    token_ids = encode_text(tokenizer, training_text)
    del training_text

    print("Encoding validation text...")
    validation_text = read_text(v2_path(training["validation_data"]))
    if smoke_test:
        validation_text = validation_text[:50_000]
    validation_ids = encode_text(tokenizer, validation_text)
    del validation_text

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["maximum_learning_rate"],
    )

    if recovery_checkpoint is not None:
        optimizer.load_state_dict(
            recovery_checkpoint["optimizer_state_dict"]
        )

    if output_path.exists():
        _, _, best_checkpoint = load_checkpoint(output_path, "cpu")
        best_validation_loss = float(
            best_checkpoint.get("validation_loss", float("inf"))
        )
        del best_checkpoint
    else:
        best_validation_loss = float("inf")

    budget_seconds = time_minutes * 60
    training_started = perf_counter()
    latest_training_loss = float("inf")
    evaluations_without_improvement = 0
    final_step = 0

    print(f"\n=== TINY GPT V2 {stage.upper()} TRAINING ===")
    print("Device:", device)
    print("Parameters:", f"{model.number_of_parameters():,}")
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Context tokens:", model.config.context_length)
    print("Training tokens:", f"{len(token_ids):,}")
    print("Validation tokens:", f"{len(validation_ids):,}")
    print("Time budget:", format_duration(budget_seconds))
    print("Starting step:", starting_step)

    model.train()

    for step in range(starting_step + 1, maximum_steps + 1):
        elapsed_seconds = perf_counter() - training_started

        if elapsed_seconds >= budget_seconds:
            break

        elapsed_fraction = min(elapsed_seconds / budget_seconds, 1.0)
        learning_rate = scheduled_learning_rate(
            elapsed_fraction=elapsed_fraction,
            maximum_rate=training["maximum_learning_rate"],
            minimum_rate=training["minimum_learning_rate"],
            warmup_fraction=training["warmup_fraction"],
        )

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        inputs, targets = create_batch(
            token_ids,
            training["batch_size"],
            model.config.context_length,
            device,
        )
        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError("Training loss was not calculated")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        latest_training_loss = loss.detach().item()
        final_step = step

        if step == 1 or step % training["log_every"] == 0:
            elapsed_seconds = perf_counter() - training_started
            completed_this_run = step - starting_step
            steps_per_second = completed_this_run / max(
                elapsed_seconds,
                0.001,
            )
            remaining_seconds = max(0.0, budget_seconds - elapsed_seconds)
            progress = min(elapsed_seconds / budget_seconds, 1.0)
            bar_width = 24
            filled = int(progress * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"Step {step:6d}  Loss {latest_training_loss:.4f}  "
                f"LR {learning_rate:.6f}  [{bar}] {progress:6.2%}  "
                f"{steps_per_second:5.2f} steps/s  "
                f"ETA {format_duration(remaining_seconds)}"
            )

        if step == 1 or step % training["evaluate_every"] == 0:
            current_validation_loss = validation_loss(
                model=model,
                validation_ids=validation_ids,
                batch_size=training["batch_size"],
                batches=training["evaluation_batches"],
                device=device,
            )
            improved = (
                current_validation_loss
                < best_validation_loss - training["minimum_improvement"]
            )

            print(
                f"             Validation {current_validation_loss:.4f}  "
                f"Best {min(best_validation_loss, current_validation_loss):.4f}"
            )

            if improved or best_validation_loss == float("inf"):
                best_validation_loss = current_validation_loss
                evaluations_without_improvement = 0
                save_checkpoint(
                    path=output_path,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    step=step,
                    training_loss=latest_training_loss,
                    validation_loss=current_validation_loss,
                    stage=stage,
                )
                print("             Saved new best checkpoint")
            else:
                evaluations_without_improvement += 1

            if (
                evaluations_without_improvement
                >= training["early_stopping_patience"]
            ):
                print("Early stopping: validation stopped improving")
                break

        if step % training["checkpoint_every"] == 0:
            save_checkpoint(
                path=latest_path,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                step=step,
                training_loss=latest_training_loss,
                validation_loss=best_validation_loss,
                stage=stage,
            )

    if final_step == 0:
        raise RuntimeError("Training ended before completing one step")

    final_validation_loss = validation_loss(
        model=model,
        validation_ids=validation_ids,
        batch_size=training["batch_size"],
        batches=training["evaluation_batches"],
        device=device,
    )

    if final_validation_loss < best_validation_loss:
        best_validation_loss = final_validation_loss
        save_checkpoint(
            path=output_path,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=final_step,
            training_loss=latest_training_loss,
            validation_loss=final_validation_loss,
            stage=stage,
        )

    save_checkpoint(
        path=latest_path,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        step=final_step,
        training_loss=latest_training_loss,
        validation_loss=final_validation_loss,
        stage=stage,
    )

    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(
        f"complete\nstep={final_step}\n"
        f"best_validation_loss={best_validation_loss}\n",
        encoding="utf-8",
    )

    print("\nTraining stage complete")
    print("Best checkpoint:", output_path)
    print("Latest checkpoint:", latest_path)
    print("Steps:", final_step)
    print("Best validation loss:", f"{best_validation_loss:.4f}")
    print("Elapsed:", format_duration(perf_counter() - training_started))


def main() -> None:
    arguments = parse_arguments()
    train_stage(
        stage=arguments.stage,
        minutes_override=arguments.minutes,
        maximum_steps_override=arguments.maximum_steps,
        force=arguments.force,
        smoke_test=arguments.smoke_test,
    )


if __name__ == "__main__":
    main()
