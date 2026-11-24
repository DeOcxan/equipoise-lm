"""Character-level text data: splits, vocabulary, windows, BPC and the OOD rule (method.md 18)."""

import hashlib
import math

import pytest
import torch

from ebt.tasks.text import (
    Vocabulary,
    bits_per_character,
    evaluation_windows,
    make_data,
    out_of_distribution_text,
    read_checked,
    split_text,
    training_windows,
)

CORPUS = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 40


def gutenberg(body: str) -> str:
    return (
        "The Project Gutenberg eBook\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
        "Title page\n[Illustration:\n\n  A PREFACE ]\nPreface text.\n"
        "[Illustration: ·PRIDE AND PREJUDICE·\n\nChapter I.]\n"
        f"{body}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\nLicence\n"
    )


def test_read_checked_rejects_another_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"line one\r\nline two\r\n")
    assert read_checked(path, hashlib.sha256(path.read_bytes()).hexdigest()) == (
        "line one\nline two\n"
    )
    with pytest.raises(ValueError, match="SHA-256"):
        read_checked(path, "0" * 64)


def test_splits_are_contiguous_with_the_documented_sizes():
    """1 115 394 characters give 1 003 854 / 55 770 / 55 770 (method.md section 18.1)."""
    text = "".join(chr(97 + i % 26) for i in range(1_115_394))
    train, validation, test = split_text(text)
    assert (len(train), len(validation), len(test)) == (1_003_854, 55_770, 55_770)
    assert train + validation + test == text


def test_vocabulary_is_sorted_and_round_trips():
    vocabulary = Vocabulary.of(CORPUS)
    assert vocabulary.characters == "".join(sorted(set(CORPUS)))
    encoded = vocabulary.encode(CORPUS)
    assert encoded.dtype == torch.uint8
    assert vocabulary.decode(encoded) == CORPUS
    with pytest.raises(KeyError):
        vocabulary.encode("~")


def test_training_windows_hold_every_offset_and_its_next_characters():
    tokens = torch.arange(10, dtype=torch.uint8)
    windows = training_windows(tokens, 4)
    assert windows.context.shape == (6, 4)
    assert windows.context[2].tolist() == [2, 3, 4, 5]
    assert windows.target[2].tolist() == [3, 4, 5, 6]
    assert windows.target[-1].tolist() == [6, 7, 8, 9]


def test_evaluation_windows_do_not_overlap():
    tokens = torch.arange(11, dtype=torch.uint8)
    windows = evaluation_windows(tokens, 4)
    assert windows.context.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert windows.target.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_bits_per_character_of_uniform_logits_is_log2_of_the_vocabulary():
    logits = torch.zeros(3, 5, 65)
    target = torch.randint(0, 65, (3, 5), dtype=torch.uint8)
    assert bits_per_character(logits, target) == pytest.approx(math.log2(65), abs=1e-12)


def test_ood_rule_strips_front_matter_and_markers_folds_and_drops():
    vocabulary = Vocabulary.of("abc '-\n")
    body = "a’b “c”—a [Illustration]_b_\n(c)"
    ood = out_of_distribution_text(gutenberg(body), vocabulary, 10)
    # Folded: a'b "c"-a _b_\n(c); the quotes, underscores and parentheses are dropped.
    assert ood.block == "a'b c-a b\nc"[:10]
    novel = "a’b “c”—a _b_\n(c)"
    assert ood.novel_characters == len(novel)
    assert ood.dropped == 6
    assert ood.dropped_fraction == 6 / len(novel)
    with pytest.raises(ValueError, match="remain"):
        out_of_distribution_text(gutenberg(body), vocabulary, 100)


def test_make_data_builds_every_split_from_the_training_vocabulary():
    ood = gutenberg("Before we proceed any further, " * 40)
    data = make_data(CORPUS, ood, 8)
    _, validation, test = split_text(CORPUS)
    assert data.validation.context.shape == ((len(validation) - 1) // 8, 8)
    assert len(data.ood_text.block) == len(test)
    assert data.ood.context.shape == data.test.context.shape
