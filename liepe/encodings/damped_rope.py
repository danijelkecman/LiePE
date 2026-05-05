from typing import Optional, Tuple
import torch
from torch import Tensor
from .base import PositionalEncoding, relative_positions
from .rope import RoPE


class DampedRoPE(PositionalEncoding):
    def __init__(self, head_dim: int, n_heads: int, base: float = 10000.0, decay: float = 0.03):
        super().__init__()
        self.rope = RoPE(head_dim=head_dim, base=base)
        self.decay = decay
        self.n_heads = n_heads

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        q2, k2, _ = self.rope(q, k)
        seq_len = q.size(-2)
        dist = relative_positions(seq_len, q.device).abs().float()
        bias = -self.decay * dist.view(1, 1, seq_len, seq_len)
        return q2, k2, bias
