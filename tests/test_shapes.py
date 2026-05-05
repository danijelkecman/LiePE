import torch
from liepe.models.transformer import TinyGPT, TransformerConfig


def test_model_forward_shapes_rope():
    cfg = TransformerConfig(vocab_size=32, seq_len=16, d_model=32, n_layers=1, n_heads=4, encoding="rope")
    model = TinyGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, loss = model(x, x)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert loss is not None


def test_model_forward_shapes_jordan():
    cfg = TransformerConfig(vocab_size=32, seq_len=16, d_model=32, n_layers=1, n_heads=4, encoding="jordan")
    model = TinyGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, loss = model(x, x)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert loss is not None
