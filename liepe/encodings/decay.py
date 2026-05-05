from typing import Optional, Tuple
import torch
from torch import Tensor
from .base import PositionalEncoding, relative_positions


class ExponentialDecay(PositionalEncoding):
    """Distance-based exponential decay implemented as logit bias."""

    def __init__(self, n_heads: int, min_decay: float = 0.01, max_decay: float = 0.2):
        super().__init__()
        rates = torch.linspace(min_decay, max_decay, n_heads)
        self.register_buffer("rates", rates.view(1, n_heads, 1, 1), persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        seq_len = q.size(-2)
        dist = relative_positions(seq_len, q.device).abs().float()
        # Since softmax(logits - rate*distance) multiplies unnormalized scores by exp(-rate*distance).
        bias = -self.rates.to(q.device) * dist.view(1, 1, seq_len, seq_len)
        return q, k, bias


def decay_generator(dim: int, rate: float = 0.1) -> Tensor:
    return -rate * torch.eye(dim)
