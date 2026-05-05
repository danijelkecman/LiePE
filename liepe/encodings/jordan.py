from typing import Optional, Tuple
import torch
from torch import Tensor
from .base import PositionalEncoding


def jordan_block(size: int, eigenvalue: float = 0.0) -> Tensor:
    B = torch.eye(size) * eigenvalue
    for i in range(size - 1):
        B[i, i + 1] = 1.0
    return B


def jordan_generator(dim: int, block_size: int = 2, eigenvalue: float = 0.0) -> Tensor:
    if dim % block_size != 0:
        raise ValueError("dim must be divisible by block_size")
    blocks = [jordan_block(block_size, eigenvalue) for _ in range(dim // block_size)]
    return torch.block_diag(*blocks)


class JordanPE(PositionalEncoding):
    """Experimental positional encoding from defective/Jordan-block generators.

    Applies x_t -> exp(tB)x_t to q and k. This is intentionally simple and suitable for
    small experiments, not optimized large-context training.
    """

    def __init__(self, head_dim: int, block_size: int = 2, eigenvalue: float = 0.0, scale: float = 0.02):
        super().__init__()
        B = jordan_generator(head_dim, block_size=block_size, eigenvalue=eigenvalue) * scale
        self.register_buffer("B", B, persistent=True)

    def _transform(self, x: Tensor) -> Tensor:
        seq_len = x.size(-2)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        B = self.B.to(device=x.device, dtype=x.dtype)
        mats = torch.stack([torch.matrix_exp(t * B) for t in positions], dim=0)  # [seq, d, d]
        return torch.einsum("bhsd,sde->bhse", x, mats)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        return self._transform(q), self._transform(k), None
