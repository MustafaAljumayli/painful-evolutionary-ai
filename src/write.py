"""Write a dose of a steering vector into model weights, and save or load it."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

_WRITE_KEYS = {"layer", "dose", "vector"}


def write_dose(
    base_model: nn.Module, layer_idx: int, vector: torch.Tensor, dose: float
) -> nn.Module:
    """Add dose * vector as a bias on the MLP output of one layer, in place.

    `dose` multiplies `vector` literally. Converting from working-coefficient
    units is the caller's job. Pass the raw S2 vector, not the unit vector.
    Take `layer_idx` from the frozen per-size setting, not from the vector
    file. Call this on the base model before attaching the adapter.

    Raises TypeError if down_proj is not a plain nn.Linear (for example a LoRA
    wrapper), and ValueError if it already has a bias or the vector has the
    wrong shape. Returns `base_model`.
    """
    block = base_model.model.layers[layer_idx]
    old = block.mlp.down_proj
    if type(old) is not nn.Linear:
        raise TypeError(
            f"down_proj at layer {layer_idx} is {type(old).__name__}, not a plain "
            "nn.Linear; write before attaching an adapter"
        )
    if old.bias is not None:
        raise ValueError(
            f"down_proj at layer {layer_idx} already has a bias: it was already "
            "written, or this is not a Qwen3 model"
        )
    if tuple(vector.shape) != (old.out_features,):
        raise ValueError(f"vector shape {tuple(vector.shape)} does not match ({old.out_features},)")
    new = nn.Linear(
        old.in_features,
        old.out_features,
        bias=True,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight)
        new.bias.copy_((dose * vector).to(new.bias))
    block.mlp.down_proj = new
    return base_model


def save_write(path: str | Path, layer_idx: int, vector: torch.Tensor, dose: float) -> None:
    """Save a write file holding layer, dose and the CPU vector.

    Never use save_pretrained for a written model: reloading drops the write.
    """
    torch.save(
        {
            "layer": int(layer_idx),
            "dose": float(dose),
            "vector": vector.detach().cpu(),
        },
        path,
    )


def load_write(path: str | Path) -> dict:
    """Load a write file and check it has exactly the keys layer, dose, vector."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or set(data) != _WRITE_KEYS:
        raise ValueError(f"write file must have exactly the keys {sorted(_WRITE_KEYS)}")
    return data
