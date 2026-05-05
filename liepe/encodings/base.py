from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module, ABC):
    """Base class for query/key positional encodings.

    Implementations may modify q/k directly and/or return an additive attention bias.

    Expected q/k shape: [batch, heads, seq_len, head_dim]
    Expected attention bias shape: [1 or batch, heads or 1, seq_len, seq_len]
    """

    @abstractmethod
    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        raise NotImplementedError


def relative_positions(seq_len: int, device: torch.device) -> Tensor:
    i = torch.arange(seq_len, device=device)
    return i[:, None] - i[None, :]


def causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
