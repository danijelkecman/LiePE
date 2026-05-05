import torch
from torch import Tensor


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: str | torch.device):
    """Next-token language modeling on random tokens with repeated prefix near the end.

    This creates a weak copy-after-delay signal: the final quarter repeats the first quarter.
    """
    x = torch.randint(1, vocab_size, (batch_size, seq_len + 1), device=device)
    span = max(2, seq_len // 4)
    x[:, -span:] = x[:, :span]
    return x[:, :-1], x[:, 1:]
