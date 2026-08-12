import numpy as np

from cmvs.fuse import align_transcript_to_frames, fuse_scores, minmax_normalize


def test_alignment_leaves_vad_gaps_at_zero() -> None:
    aligned = align_transcript_to_frames(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 3.0]),
        np.array([1.5, 4.0]),
        np.array([0.2, 0.8]),
    )

    assert np.allclose(aligned, [0.2, 0.2, 0.0, 0.8, 0.0])


def test_minmax_and_fusion_calibrate_modalities_independently() -> None:
    visual = minmax_normalize(np.array([0.2, 0.3]))
    transcript = minmax_normalize(np.array([5.0, 15.0]))

    assert np.allclose(visual, [0.0, 1.0])
    assert np.allclose(transcript, [0.0, 1.0])
    assert np.allclose(fuse_scores(visual, transcript, alpha=0.6), [0.0, 1.0])


def test_constant_scores_normalize_to_zeros() -> None:
    assert np.array_equal(minmax_normalize(np.array([4.0, 4.0])), np.zeros(2, dtype=np.float32))
