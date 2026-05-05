import torch


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: str | torch.device):
    """Discretized trend + periodic signal as token prediction.

    This is intentionally synthetic: Jordan-style encodings should have a fairer shot
    because the sequence contains trend-like/polynomial temporal structure.
    """
    t = torch.arange(seq_len + 1, device=device).float().view(1, -1)
    slope = torch.randint(1, 5, (batch_size, 1), device=device).float()
    phase = torch.rand(batch_size, 1, device=device) * 6.28318
    signal = slope * t + 8.0 * torch.sin(0.15 * t + phase)
    x = torch.remainder(signal.round().long(), vocab_size)
    return x[:, :-1], x[:, 1:]
