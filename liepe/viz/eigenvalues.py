from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from liepe.encodings.rope import rope_generator
from liepe.encodings.decay import decay_generator
from liepe.encodings.jordan import jordan_generator


def generator(name: str, dim: int) -> torch.Tensor:
    if name == "rope":
        return rope_generator(dim)
    if name == "decay":
        return decay_generator(dim)
    if name == "jordan":
        return jordan_generator(dim) * 0.02
    if name == "damped_rope":
        return rope_generator(dim) - 0.03 * torch.eye(dim)
    if name == "nope":
        return torch.zeros(dim, dim)
    raise ValueError("Eigenvalue plot supports: nope, rope, decay, damped_rope, jordan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", default="rope", choices=["nope", "rope", "decay", "damped_rope", "jordan"])
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--output", default="outputs/eigenvalues.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    B = generator(args.encoding, args.dim)
    eig = torch.linalg.eigvals(B).numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(eig.real, eig.imag)
    plt.axvline(0.0, linewidth=1)
    plt.axhline(0.0, linewidth=1)
    plt.title(f"Generator eigenvalues: {args.encoding}")
    plt.xlabel("real part: decay/growth")
    plt.ylabel("imaginary part: rotation")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
