from typing import Optional, Tuple
import torch
from torch import Tensor
from .base import PositionalEncoding, relative_positions


class ALiBi(PositionalEncoding):
    """Simple ALiBi-style linear distance penalty for causal attention."""

    def __init__(self, n_heads: int, slope_base: float = 2.0):
        super().__init__()
        slopes = torch.tensor([1.0 / (slope_base ** h) for h in range(n_heads)], dtype=torch.float32)
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1), persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        seq_len = q.size(-2)
        rel = relative_positions(seq_len, q.device).abs().float()
        # Penalize distance. Shape: [1, heads, seq, seq]
        bias = -self.slopes.to(q.device) * rel.view(1, 1, seq_len, seq_len)
        return q, k, bias
