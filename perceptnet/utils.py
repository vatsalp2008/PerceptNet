"""Small shared utilities (device selection, seeding)."""

from __future__ import annotations


def get_device(prefer: str = "auto"):
    """Return the best available torch device.

    Preference order for ``"auto"``: CUDA > MPS (Apple) > CPU. Pass an explicit
    string (``"cpu"``, ``"cuda"``, ``"mps"``) to force one. Centralizing this avoids
    scattered ``cuda.is_available()`` checks and makes the MPS fallback explicit.
    """
    import torch

    if prefer not in ("auto",):
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
