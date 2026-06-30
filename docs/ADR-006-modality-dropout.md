# ADR-006: Modality dropout via zero-filled image features

**Status:** Accepted

## Context
A safety-critical perception stack must degrade gracefully when a sensor fails — camera glare/blackout, LiDAR occlusion in rain/fog. The fusion head normally consumes `[LiDAR | image]` features; we need a defined behavior when the image side is missing. Options:
- **Separate inference paths** per available modality — best accuracy per mode, but 2–3× the models to train/maintain/deploy and a branch to switch at runtime.
- **Feature masking** — learn a mask token / attention over present modalities. Flexible, more complex, needs training with masks.
- **Zero-filling** — substitute a zero vector for the missing modality's feature and carry an availability flag.

## Decision
**Zero-fill the projected image feature** (and pass `camera_available=False`) when the camera is unavailable; the same fusion head runs LiDAR-only with no architecture change. Train with random camera dropout so the head learns to lean on LiDAR when the image branch is zero. The robustness study (`perceptnet.evaluation.robustness_study`) quantifies degradation under camera dropout and 50%/90% LiDAR point-drop.

## Consequences
- **+** One model, one code path; trivially testable (`tests/test_fusion.py` asserts the image input is ignored under dropout).
- **+** Matches the LiDAR-primary design (ADR-001): losing the camera degrades to a competent LiDAR-only detector.
- **−** Zero-filling is a blunt signal vs. learned masking; some achievable accuracy under partial failure is left on the table.
- **−** LiDAR-unavailable is the harder case (LiDAR seeds proposals); the documented fallback is monocular depth estimation, a stretch goal.
