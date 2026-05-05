import torch


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: str | torch.device):
    x = torch.randint(1, vocab_size, (batch_size, seq_len + 1), device=device)
    # Inject repeated bigram: a,b ... a,b so model can exploit induction-like pattern.
    a = torch.randint(1, vocab_size, (batch_size,), device=device)
    b = torch.randint(1, vocab_size, (batch_size,), device=device)
    pos1 = seq_len // 4
    pos2 = 3 * seq_len // 4
    x[:, pos1] = a
    x[:, pos1 + 1] = b
    x[:, pos2] = a
    x[:, pos2 + 1] = b
    return x[:, :-1], x[:, 1:]
