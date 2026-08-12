from types import SimpleNamespace

import numpy as np
import pytest

from cmvs.group import group_frames_into_moments


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        search=SimpleNamespace(
            score_threshold=0.5,
            merge_gap_s=2.0,
            min_moment_s=1.0,
            max_results=10,
        )
    )


def test_consecutive_qualifying_frames_become_one_moment() -> None:
    timestamps = np.arange(10, dtype=np.float32)
    fused = np.linspace(0.5, 0.9, num=10, dtype=np.float32)

    moments = group_frames_into_moments(timestamps, fused, fused, fused, _cfg())

    assert len(moments) == 1
    assert moments[0]["start"] == 0.0
    assert moments[0]["end"] == 9.0
    assert moments[0]["score"] == pytest.approx(0.9)


def test_gap_larger_than_threshold_splits_moments() -> None:
    timestamps = np.array([0.0, 1.0, 5.0, 6.0], dtype=np.float32)
    fused = np.array([0.8, 0.7, 0.9, 0.6], dtype=np.float32)

    moments = group_frames_into_moments(timestamps, fused, fused, fused, _cfg())

    assert len(moments) == 2
    assert [moment["score"] for moment in moments] == pytest.approx([0.9, 0.8])


def test_single_frame_hit_is_padded_to_minimum_duration() -> None:
    moments = group_frames_into_moments(
        np.array([3.0]),
        np.array([0.8]),
        np.array([0.5]),
        np.array([0.6]),
        _cfg(),
    )

    assert moments[0]["end"] - moments[0]["start"] == 1.0
