import torch
import os


def _resolve_device() -> torch.device:
    requested = os.environ.get("EDG_ACOUSTICS_DEVICE", "auto").strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "EDG_ACOUSTICS_DEVICE=cuda was requested, but CUDA is not available."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(
        "EDG_ACOUSTICS_DEVICE must be one of 'auto', 'cpu', or 'cuda'."
    )


device = _resolve_device()
dtype = torch.float64
