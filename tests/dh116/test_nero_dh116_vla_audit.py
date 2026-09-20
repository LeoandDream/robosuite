"""Fast audit tests for the saved Nero + DH116 VLA milestone."""

import json

import cv2
import numpy as np

from nero_dh116_vla import (
    REPORT_PATH,
    completion_audit,
    locator_features,
    next_numbered_video_path,
)


def test_locator_features_are_resolution_independent():
    small = np.zeros((32, 32, 3), dtype=np.uint8)
    small[9:14, 19:24, 0] = 255
    large = cv2.resize(small, (128, 128), interpolation=cv2.INTER_NEAREST)

    small_feature = locator_features(small)
    large_feature = locator_features(large)

    np.testing.assert_allclose(small_feature[1:3], large_feature[1:3], atol=0.01)


def test_saved_report_passes_no_cheating_completion_audit():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    audit = completion_audit(report)

    assert audit["passed"]
    assert all(audit["checks"].values())


def test_numbered_video_path_never_reuses_existing_file(tmp_path):
    video_dir = tmp_path / "vla"
    video_dir.mkdir()
    (video_dir / "reach_001.mp4").write_bytes(b"old-1")
    (video_dir / "reach_002.mp4").write_bytes(b"old-2")

    candidate = next_numbered_video_path(video_dir, "reach")

    assert candidate.name == "reach_003.mp4"
    assert (video_dir / "reach_001.mp4").read_bytes() == b"old-1"
