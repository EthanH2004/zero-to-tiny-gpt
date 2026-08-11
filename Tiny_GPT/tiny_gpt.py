"""Reusable character-level Tiny GPT architecture."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocabulary_size: int
    context_length: int = 64
    embedding_size: int = 64
    number_of_heads: int = 4
    number_of_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be greater than zero")

        if self.context_length <= 0:
            raise ValueError("context_length must be greater than zero")

        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be greater than zero")

        if self.number_of_heads <= 0:
            raise ValueError("number_of_heads must be greater than zero")

        if self.number_of_layers <= 0:
            raise ValueError("number_of_layers must be greater than zero")

        if self.embedding_size % self.number_of_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by number_of_heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be at least 0 and less than 1")


class CharacterTokenizer:
    def __init__(self, vocabulary: list[str]):
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("The vocabulary cannot contain duplicates")

        self.vocabulary = list(vocabulary)
        self.character_to_id: dict[str, int] = {}
        self.id_to_character: dict[int, str] = {}

        for token_id, character in enumerate(vocabulary):
            self.character_to_id[character] = token_id
            self.id_to_character[token_id] = character

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        unknown_characters = sorted(
            set(text) - set(self.vocabulary)
        )

        if unknown_characters:
            raise ValueError(
                f"Characters are missing from the vocabulary: "
                f"{unknown_characters}"
            )

        token_ids = []

        for character in text:
            token_ids.append(self.character_to_id[character])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text = ""

        for token_id in token_ids:
            text += self.id_to_character[token_id]

        return text


class AttentionHead(nn.Module):
    causal_mask: torch.Tensor

    def __init__(
        self,
        embedding_size: int,
        head_size: int,
        context_length: int,
        dropout: float,
    ):
        super().__init__()

        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequence_length = embeddings.shape[1]

        queries = self.query(embeddings)
        keys = self.key(embeddings)
        values = self.value(embeddings)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores /= keys.shape[-1] ** 0.5

        allowed_positions = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            ~allowed_positions,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(attention_weights)

        return attention_weights @ values


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        head_size = (
            config.embedding_size // config.number_of_heads
        )

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    embedding_size=config.embedding_size,
                    head_size=head_size,
                    context_length=config.context_length,
                    dropout=config.dropout,
                )
                for _ in range(config.number_of_heads)
            ]
        )

        self.projection = nn.Linear(
            config.embedding_size,
            config.embedding_size,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        head_outputs = [
            head(embeddings) for head in self.heads
        ]

        combined_output = torch.cat(head_outputs, dim=-1)
        projected_output = self.projection(combined_output)

        return self.dropout(projected_output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        hidden_size = 4 * config.embedding_size

        self.network = nn.Sequential(
            nn.Linear(config.embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.embedding_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(
            config.embedding_size
        )
        self.feed_forward_norm = nn.LayerNorm(
            config.embedding_size
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings + self.attention(
            self.attention_norm(embeddings)
        )

        embeddings = embeddings + self.feed_forward(
            self.feed_forward_norm(embeddings)
        )

        return embeddings


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocabulary_size,
            config.embedding_size,
        )
        self.position_embeddings = nn.Embedding(
            config.context_length,
            config.embedding_size,
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config.number_of_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.embedding_size)
        self.output_layer = nn.Linear(
            config.embedding_size,
            config.vocabulary_size,
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence_length = token_ids.shape[1]

        if sequence_length > self.config.context_length:
            raise ValueError(
                "The sequence is longer than the context length"
            )

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.position_embeddings(
            position_ids
        )

        embeddings = token_embeddings + position_embeddings
        embeddings = self.transformer_blocks(embeddings)
        embeddings = self.final_norm(embeddings)
        logits = self.output_layer(embeddings)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocabulary_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            context = token_ids[
                :, -self.config.context_length:
            ]

            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, next_token_logits.shape[-1])
                cutoff = torch.topk(
                    next_token_logits,
                    top_k,
                ).values[:, -1:]

                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff,
                    float("-inf"),
                )

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token_id = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            token_ids = torch.cat(
                (token_ids, next_token_id),
                dim=1,
            )

        if was_training:
            self.train()

        return token_ids

    def number_of_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: TinyGPT,
    tokenizer: CharacterTokenizer,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    loss: float | None = None,
):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_config": asdict(model.config),
        "vocabulary": tokenizer.vocabulary,
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[TinyGPT, CharacterTokenizer, dict[str, Any]]:
    checkpoint: dict[str, Any] = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    config = ModelConfig(**checkpoint["model_config"])
    tokenizer = CharacterTokenizer(checkpoint["vocabulary"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, tokenizer, checkpoint
