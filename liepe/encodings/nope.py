from typing import Optional, Tuple
import torch
from torch import Tensor
from .base import PositionalEncoding


class NoPE(PositionalEncoding):
    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        return q, k, None
