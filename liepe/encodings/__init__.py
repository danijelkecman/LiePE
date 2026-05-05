from .nope import NoPE
from .rope import RoPE, rope_generator
from .alibi import ALiBi
from .decay import ExponentialDecay, decay_generator
from .damped_rope import DampedRoPE
from .jordan import JordanPE, jordan_block, jordan_generator
from .learned_lie import LearnedLiePE


def build_encoding(name: str, head_dim: int, n_heads: int):
    name = name.lower()
    if name == "nope":
        return NoPE()
    if name == "rope":
        return RoPE(head_dim=head_dim)
    if name == "alibi":
        return ALiBi(n_heads=n_heads)
    if name == "decay":
        return ExponentialDecay(n_heads=n_heads)
    if name == "damped_rope":
        return DampedRoPE(head_dim=head_dim, n_heads=n_heads)
    if name == "jordan":
        return JordanPE(head_dim=head_dim)
    if name == "learned_lie":
        return LearnedLiePE(head_dim=head_dim)
    raise ValueError(f"Unknown encoding: {name}")
