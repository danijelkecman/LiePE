from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from .base import PositionalEncoding


class LearnedLiePE(PositionalEncoding):
    """Learnable matrix-generator positional encoding.

    Warning: this can become unstable. Keep dimensions and positions small initially.
    """

    def __init__(self, head_dim: int, init_scale: float = 0.005):
        super().__init__()
        self.B = nn.Parameter(torch.randn(head_dim, head_dim) * init_scale)

    def _stable_generator(self) -> Tensor:
        # Small stabilizer: subtract a diagonal term to discourage positive real growth.
        return self.B - 0.01 * torch.eye(self.B.size(0), device=self.B.device, dtype=self.B.dtype)

    def _transform(self, x: Tensor) -> Tensor:
        seq_len = x.size(-2)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        B = self._stable_generator().to(dtype=x.dtype)
        mats = torch.stack([torch.matrix_exp(t * B) for t in positions], dim=0)
        return torch.einsum("bhsd,sde->bhse", x, mats)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        return self._transform(q), self._transform(k), None

    def stability_penalty(self) -> Tensor:
        eig = torch.linalg.eigvals(self._stable_generator())
        return torch.relu(eig.real).pow(2).mean()
