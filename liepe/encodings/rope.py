from typing import Optional, Tuple
import torch
from torch import Tensor
from .base import PositionalEncoding


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE(PositionalEncoding):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len: int, device: torch.device):
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def apply_rope(self, x: Tensor) -> Tensor:
        seq_len = x.size(-2)
        cos, sin = self._cos_sin(seq_len, x.device)
        return (x * cos) + (rotate_half(x) * sin)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        return self.apply_rope(q), self.apply_rope(k), None


def rope_generator(head_dim: int, base: float = 10000.0) -> Tensor:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    B = torch.zeros(head_dim, head_dim)
    for i, w in enumerate(inv_freq):
        j = 2 * i
        B[j, j + 1] = -w
        B[j + 1, j] = w
    return B
