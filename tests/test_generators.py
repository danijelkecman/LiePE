import torch
from liepe.encodings.jordan import jordan_generator
from liepe.encodings.rope import rope_generator
from liepe.encodings.decay import decay_generator


def check_group_property(B: torch.Tensor):
    t = torch.tensor(0.7)
    s = torch.tensor(1.3)
    A_t = torch.matrix_exp(t * B)
    A_s = torch.matrix_exp(s * B)
    A_ts = torch.matrix_exp((t + s) * B)
    assert torch.allclose(A_ts, A_t @ A_s, atol=1e-5)
    assert torch.allclose(torch.matrix_exp(torch.tensor(0.0) * B), torch.eye(B.size(0)), atol=1e-6)


def test_rope_generator_group_property():
    check_group_property(rope_generator(8))


def test_decay_generator_group_property():
    check_group_property(decay_generator(8))


def test_jordan_generator_group_property():
    check_group_property(jordan_generator(8, block_size=2) * 0.02)
