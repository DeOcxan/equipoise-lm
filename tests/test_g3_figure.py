"""The G3 figure tool draws from the committed G3 summaries and per-seed files only."""

from pathlib import Path

from tools import g3_figure

ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_figure_from_committed_summaries(tmp_path):
    data = g3_figure.curves(ROOT / "results")
    assert [entry["trained"] for entry in data["text"]] == [4]
    assert [entry["trained"] for entry in data["mnist"]] == [2]
    assert [entry["trained"] for entry in data["regression"]] == [4, 4, 4, 4]
    for entries in data.values():
        for entry in entries:
            assert entry["steps"] == [1, 2, 4, 8, 16, 32, 64]

    path = tmp_path / "g3_steps.png"
    g3_figure.plot(data, path)
    content = path.read_bytes()
    assert content.startswith(PNG_SIGNATURE)
    assert len(content) > 10_000
