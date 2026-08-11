"""Train, save, and use a GPT-style byte-level BPE tokenizer."""

from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


UNKNOWN_TOKEN = "<unk>"


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        files: list[Path],
        vocabulary_size: int,
        minimum_frequency: int,
    ) -> "BPETokenizer":
        tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = decoders.ByteLevel()

        trainer = BpeTrainer(
            vocab_size=vocabulary_size,
            min_frequency=minimum_frequency,
            special_tokens=[UNKNOWN_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        tokenizer.train(
            files=[str(path) for path in files],
            trainer=trainer,
        )
        return cls(tokenizer)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_string(cls, serialized_tokenizer: str) -> "BPETokenizer":
        return cls(Tokenizer.from_str(serialized_tokenizer))

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path), pretty=True)

    def to_string(self) -> str:
        return self.tokenizer.to_str(pretty=False)

