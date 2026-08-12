from pathlib import Path

import numpy as np

from cmvs.utils import l2_normalize, read_jsonl, slugify, write_jsonl


def test_slugify_uses_file_stem_and_safe_characters() -> None:
    assert slugify("My lecture (final).mp4") == "my-lecture-final"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "rows.jsonl"
    rows = [{"frame_idx": 0}, {"frame_idx": 1, "text": "hello"}]

    write_jsonl(destination, rows)

    assert read_jsonl(destination) == rows


def test_l2_normalize_returns_float32_unit_rows() -> None:
    result = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))

    assert result.dtype == np.float32
    assert np.allclose(result[0], np.array([0.6, 0.8], dtype=np.float32))
    assert np.array_equal(result[1], np.zeros(2, dtype=np.float32))
