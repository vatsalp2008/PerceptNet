"""Camera-branch smoke test. Marked ``network`` (downloads YOLO weights on first
run) so it is excluded from the default ``make test`` and run with ``-m network``."""

from pathlib import Path

import pytest

FIXTURE_IMG = Path(__file__).parent / "fixtures" / "mini_kitti" / "training" / "image_2" / "000000.png"


@pytest.mark.network
def test_camera_branch_detects_and_extracts_features():
    from perceptnet.models.camera_branch import CameraBranch

    cam = CameraBranch("yolov8n.pt", device="cpu")
    dets, feats = cam(str(FIXTURE_IMG))

    assert set(dets) == {"boxes", "scores", "labels"}
    assert dets["boxes"].ndim == 2 and dets["boxes"].shape[1] == 4
    assert len(dets["scores"]) == len(dets["boxes"]) == len(dets["labels"])

    # FPN maps captured via the Detect hook, ordered finest -> coarsest.
    assert set(feats) == {"p3", "p4", "p5"}
    assert feats["p3"].shape[-1] >= feats["p4"].shape[-1] >= feats["p5"].shape[-1]
    assert cam.fpn_channels is not None and len(cam.fpn_channels) == 3
