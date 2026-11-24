"""The code fingerprint follows the content of every file it covers, and sees every import.

The fingerprint reads imports with `ast` and follows absolute `ebt` imports only (level 0), so a
relative import would add a dependency that the fingerprint does not cover.
"""

import ast
import importlib.util
import io
import shutil
import tokenize
from pathlib import Path

import pytest

import ebt
from ebt.fingerprint import PACKAGE, code_fingerprint, fingerprint_files

EXPERIMENTS = (
    "text_lm",
    "regression_g2",
    "mnist_denoising",
    "inference_regression",
    "inference_mnist",
    "inference_text",
    "inference_g5",
)


def one_byte_changed(content: bytes) -> bytes:
    """`content` with one letter of its first comment or string swapped in case, so that the
    file still parses (the fingerprint reads every file with `ast`)."""
    text = content.decode()
    lines = text.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            row, column = token.start
            offset = sum(len(line) for line in lines[: row - 1]) + column
            for i, character in enumerate(token.string):
                if character.isascii() and character.isalpha():
                    position = offset + i
                    changed = text[:position] + character.swapcase() + text[position + 1 :]
                    assert len(changed.encode()) == len(content)
                    return changed.encode()
    raise AssertionError("no comment or string with a letter to change")


def copied_fingerprint(root: Path):
    """The fingerprint module of a copy of the package under `root`, loaded from the copy."""
    target = root / "ebt"
    shutil.copytree(Path(ebt.__file__).parent, target, ignore=shutil.ignore_patterns("__pycache__"))
    spec = importlib.util.spec_from_file_location("copied_fingerprint", target / "fingerprint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_changing_one_byte_of_any_covered_file_changes_the_digest(experiment, tmp_path):
    copy = copied_fingerprint(tmp_path)
    original = copy.code_fingerprint(experiment)
    assert original == code_fingerprint(experiment)
    for path in copy.fingerprint_files(experiment):
        content = path.read_bytes()
        changed = one_byte_changed(content)
        assert sum(a != b for a, b in zip(changed, content, strict=True)) == 1
        path.write_bytes(changed)
        assert copy.code_fingerprint(experiment) != original, path.relative_to(tmp_path)
        path.write_bytes(content)
    assert copy.code_fingerprint(experiment) == original


def test_no_fingerprinted_module_uses_a_relative_import():
    files = sorted({path for name in EXPERIMENTS for path in fingerprint_files(name)})
    assert files
    relative = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                where = path.relative_to(PACKAGE).as_posix()
                name = "." * node.level + (node.module or "")
                relative.append(f"{where}:{node.lineno}: from {name}")
    assert not relative, (
        "relative imports are not followed by the code fingerprint; use absolute `ebt` imports:\n"
        + "\n".join(relative)
    )
