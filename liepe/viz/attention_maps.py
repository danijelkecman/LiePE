from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from liepe.models.transformer import TinyGPT, TransformerConfig
from liepe.tasks import TASKS
from liepe.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", default="rope", choices=["nope", "rope", "alibi", "decay", "damped_rope", "jordan", "learned_lie"])
    parser.add_argument("--task", default="copy_delay", choices=TASKS.keys())
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--output", default="outputs/attention.png")
    args = parser.parse_args()

    set_seed(42)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        encoding=args.encoding,
    )
    model = TinyGPT(config)
    x, _ = TASKS[args.task](batch_size=1, seq_len=args.seq_len, vocab_size=args.vocab_size, device="cpu")
    model.eval()
    with torch.no_grad():
        model(x)
    attn = model.blocks[0].attn.last_attention[0, 0].cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(attn, aspect="auto")
    plt.title(f"Attention map: {args.encoding}")
    plt.xlabel("key position")
    plt.ylabel("query position")
    plt.colorbar(label="attention")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
