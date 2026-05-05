import torch


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: str | torch.device):
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    steps = torch.randint(1, min(10, vocab_size), (batch_size, 1), device=device)
    positions = torch.arange(seq_len + 1, device=device).view(1, -1)
    x = (starts + steps * positions) % vocab_size
    return x[:, :-1].long(), x[:, 1:].long()
