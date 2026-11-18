"""Tiny pure-torch model with the Qwen3 module layout, for CPU tests."""

from __future__ import annotations

import torch
from torch import nn

# Test fixture sizes, arbitrary (not from any source; tiny model only).
HIDDEN: int = 16
INTERMEDIATE: int = 32
N_LAYERS: int = 3
VOCAB: int = 50


class TinyMLP(nn.Module):
    """Gated MLP with the Qwen3MLP shape and no biases."""

    def __init__(self) -> None:
        """Build the three bias-free projections."""
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return down_proj(silu(gate_proj(x)) * up_proj(x))."""
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    """Decoder block with a residual MLP at the end, like Qwen3."""

    def __init__(self, return_tuple: bool = True) -> None:
        """Build the block; return_tuple picks tuple or bare-tensor output."""
        super().__init__()
        self.input_layernorm = nn.LayerNorm(HIDDEN)
        self.self_attn = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(HIDDEN)
        self.mlp = TinyMLP()
        self.return_tuple = return_tuple

    def forward(self, h: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        """Run attention stand-in then MLP, each with a residual."""
        h = h + self.self_attn(self.input_layernorm(h))
        h = h + self.mlp(self.post_attention_layernorm(h))
        return (h,) if self.return_tuple else h


class TinyModel(nn.Module):
    """Stack of blocks and a final norm."""

    def __init__(self, return_tuple: bool = True) -> None:
        """Build N_LAYERS blocks."""
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(return_tuple) for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(HIDDEN)


class TinyForCausalLM(nn.Module):
    """Model plus lm_head, taking hidden states directly."""

    def __init__(self, return_tuple: bool = True) -> None:
        """Build the model and the head."""
        super().__init__()
        self.model = TinyModel(return_tuple)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map hidden states [batch, seq, HIDDEN] to logits [batch, seq, VOCAB]."""
        for block in self.model.layers:
            out = block(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        return self.lm_head(self.model.norm(hidden))


def make_tiny_model(seed: int = 0, return_tuple: bool = True) -> TinyForCausalLM:
    """Build a seeded float32 CPU model in eval mode."""
    torch.manual_seed(seed)
    return TinyForCausalLM(return_tuple).float().cpu().eval()


def add_steering_hook(
    model: nn.Module, layer_idx: int, vector: torch.Tensor, dose: float
) -> torch.utils.hooks.RemovableHandle:
    """Add dose * vector to every position of a block's output via a forward hook.

    Handles tuple output (modifies output[0]) and bare-tensor output.
    """

    def hook(_module: nn.Module, _inputs: tuple, output: object) -> object:
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + (dose * vector).to(hs.dtype)
        if isinstance(output, tuple):
            return (hs, *output[1:])
        return hs

    return model.model.layers[layer_idx].register_forward_hook(hook)
