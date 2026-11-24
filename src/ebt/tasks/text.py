"""Character-level text: corpus, vocabulary, splits, windows and bits per character (method.md 18).

Training corpus: tiny Shakespeare, the `input.txt` of the char-rnn repository
(data/tinyshakespeare). Out-of-distribution text: Pride and Prejudice, Project Gutenberg eBook
1342, reduced to the training vocabulary by the rule of method.md section 18.1. Both files are
read from a local directory and checked by SHA-256; neither is stored in the repository.
"""

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ebt.training.batch import Batch

CORPUS_FILE = "tinyshakespeare.txt"
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
OOD_FILE = "pg1342.txt"
OOD_SHA256 = "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"
TRAIN_FRACTION = 0.9

GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"
FIRST_CHAPTER = "Chapter I.]"
"""The line that ends the illustrated heading of chapter 1 in eBook 1342."""
ILLUSTRATION = re.compile(r"\[Illustration[^\]]*\]")
FOLDING = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)
"""Unicode quotes, dashes and the no-break space mapped to their ASCII forms."""


def read_checked(path: Path, sha256: str) -> str:
    """The UTF-8 text of `path`, after checking its SHA-256; CRLF line ends become LF."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path}: SHA-256 {digest}, expected {sha256}")
    return data.decode("utf-8").replace("\r\n", "\n")


@dataclass(frozen=True)
class Vocabulary:
    """The characters of the training split, sorted by code point; a character's index is its
    rank. There is no unknown token."""

    characters: str

    @classmethod
    def of(cls, text: str) -> "Vocabulary":
        return cls("".join(sorted(set(text))))

    def __len__(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> Tensor:
        """Indices as uint8 (at most 256 characters); raises KeyError for a character outside
        the vocabulary."""
        index = {character: i for i, character in enumerate(self.characters)}
        return torch.tensor([index[character] for character in text], dtype=torch.uint8)

    def decode(self, indices: Tensor) -> str:
        return "".join(self.characters[i] for i in indices.tolist())


def split_text(text: str) -> tuple[str, str, str]:
    """Contiguous training, validation and test blocks: the first floor(0.9 N) characters, then
    the remainder halved (the validation block takes the smaller half)."""
    train = math.floor(TRAIN_FRACTION * len(text))
    validation = (len(text) - train) // 2
    return text[:train], text[train : train + validation], text[train + validation :]


@dataclass(frozen=True)
class OutOfDistributionText:
    """The evaluated out-of-distribution block and the counts of the rule that produced it."""

    block: str
    body_characters: int
    novel_characters: int
    """Characters of the novel after the illustration markers are removed (steps 3 and 4)."""
    dropped: int
    """Characters outside the vocabulary after folding (step 6)."""

    @property
    def dropped_fraction(self) -> float:
        return self.dropped / self.novel_characters


def out_of_distribution_text(
    raw: str, vocabulary: Vocabulary, length: int
) -> OutOfDistributionText:
    """The rule of method.md section 18.1, applied to the text of eBook 1342.

    Body between the Gutenberg markers; novel from the line after the first "Chapter I.]";
    illustration markers removed; quotes, dashes and the no-break space folded to ASCII;
    characters outside `vocabulary` dropped; the first `length` characters kept.
    """
    lines = raw.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(GUTENBERG_START))
    end = next(i for i, line in enumerate(lines) if line.startswith(GUTENBERG_END))
    body = lines[start + 1 : end]
    novel = ILLUSTRATION.sub("", "\n".join(body[body.index(FIRST_CHAPTER) + 1 :]))
    folded = novel.translate(FOLDING)
    allowed = set(vocabulary.characters)
    kept = "".join(character for character in folded if character in allowed)
    if len(kept) < length:
        raise ValueError(f"only {len(kept)} characters remain, {length} requested")
    return OutOfDistributionText(
        block=kept[:length],
        body_characters=len("\n".join(body)),
        novel_characters=len(novel),
        dropped=len(folded) - len(kept),
    )


def training_windows(tokens: Tensor, length: int) -> Batch:
    """Every window of `length` characters (context) with the next character of each position
    (target): one row per start offset 0, ..., N − length − 1."""
    windows = tokens.unfold(0, length + 1, 1)
    return Batch(context=windows[:, :-1], target=windows[:, 1:])


def evaluation_windows(tokens: Tensor, length: int) -> Batch:
    """Non-overlapping windows: rows i hold characters [i·length, (i + 1)·length) as context
    and the following character of each position as target; floor((N − 1) / length) rows.
    Early positions of a window have a short context, the same for every model."""
    rows = (len(tokens) - 1) // length
    context = tokens[: rows * length].reshape(rows, length)
    target = tokens[1 : rows * length + 1].reshape(rows, length)
    return Batch(context=context, target=target)


def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """Mean cross-entropy in nats of logits (..., V) against character indices (...)."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1).long())


def bits_per_character(logits: Tensor, target: Tensor) -> float:
    """Mean cross-entropy in bits over every predicted character."""
    return cross_entropy(logits.double(), target).item() / math.log(2)


@dataclass(frozen=True)
class TextData:
    vocabulary: Vocabulary
    train: Batch
    """Every training window (`training_windows`), uint8."""
    validation: Batch
    test: Batch
    ood: Batch | None
    """The out-of-distribution block in evaluation windows; None without the eBook file."""
    ood_text: OutOfDistributionText | None


def make_data(corpus: str, ood_raw: str | None, length: int) -> TextData:
    """Splits, vocabulary and windows of `corpus`, and the out-of-distribution block of
    `ood_raw` with the length of the test split."""
    train, validation, test = split_text(corpus)
    vocabulary = Vocabulary.of(train)
    ood = None
    if ood_raw is not None:
        ood = out_of_distribution_text(ood_raw, vocabulary, len(test))
    return TextData(
        vocabulary=vocabulary,
        train=training_windows(vocabulary.encode(train), length),
        validation=evaluation_windows(vocabulary.encode(validation), length),
        test=evaluation_windows(vocabulary.encode(test), length),
        ood=None if ood is None else evaluation_windows(vocabulary.encode(ood.block), length),
        ood_text=ood,
    )


def load_data(directory: Path, length: int) -> TextData:
    """`make_data` of the checked files in `directory`; the eBook file is required."""
    corpus = read_checked(directory / CORPUS_FILE, CORPUS_SHA256)
    return make_data(corpus, read_checked(directory / OOD_FILE, OOD_SHA256), length)
