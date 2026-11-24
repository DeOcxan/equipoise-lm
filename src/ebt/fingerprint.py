"""Fingerprint of the code that computes an experiment's results.

The fingerprint is the SHA-256 of the source files that can change a result of one experiment:
the experiment's module in `ebt.experiments` and every module of the `ebt` package that it
imports, directly or through other `ebt` modules, together with the `__init__.py` of every
package on the way (Python runs them on import). The imports are read from the source with
`ast`, without running it. Modules the experiment does not import are left out, so adding or
changing another experiment or task does not mark this experiment's results as stale. The files
are hashed in sorted order of their path relative to the package, each as its path, a zero byte,
its contents and a zero byte.
"""

import ast
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def _source(module: str) -> Path | None:
    """The file of `module` (a dotted name inside `ebt`), or None if there is none."""
    path = ROOT.joinpath(*module.split("."))
    if path.is_dir():
        return path / "__init__.py"
    file = path.with_suffix(".py")
    return file if file.exists() else None


def _imported(path: Path) -> set[str]:
    """Dotted names of the `ebt` modules that the file at `path` imports."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.split(".")[0] == "ebt" and _source(name)}


def fingerprint_files(experiment_module: str) -> list[Path]:
    """The files hashed for `experiment_module`, e.g. "regression_g2", sorted by path."""
    pending, seen = [f"ebt.experiments.{experiment_module}"], set()
    while pending:
        module = pending.pop()
        parts = module.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name not in seen:
                seen.add(name)
                pending.extend(_imported(_source(name)))
    files = {_source(name) for name in seen}
    return sorted(files, key=lambda path: path.relative_to(PACKAGE).as_posix())


def code_fingerprint(experiment_module: str) -> str:
    """SHA-256 hex digest of `fingerprint_files(experiment_module)`."""
    digest = hashlib.sha256()
    for path in fingerprint_files(experiment_module):
        digest.update(path.relative_to(PACKAGE).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
