from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import trange

from liepe.models.transformer import TinyGPT, TransformerConfig
from liepe.tasks import TASKS
from liepe.utils.seed import set_seed


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == targets).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS.keys(), default="copy_delay")
    parser.add_argument("--encoding", default="rope", choices=["nope", "rope", "alibi", "decay", "damped_rope", "jordan", "learned_lie"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/model.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    config = TransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        encoding=args.encoding,
    )
    model = TinyGPT(config).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    make_batch = TASKS[args.task]

    pbar = trange(args.steps, desc=f"train {args.task}/{args.encoding}")
    last_loss = None
    last_acc = None
    for step in pbar:
        x, y = make_batch(args.batch_size, args.seq_len, args.vocab_size, args.device)
        logits, loss = model(x, y)
        assert loss is not None

        # Optional regularizer for learned generator.
        reg = torch.tensor(0.0, device=args.device)
        for module in model.modules():
            if hasattr(module, "stability_penalty"):
                reg = reg + module.stability_penalty()
        total_loss = loss + 1e-3 * reg

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == args.steps - 1:
            last_loss = loss.item()
            last_acc = accuracy(logits, y)
            pbar.set_postfix(loss=f"{last_loss:.4f}", acc=f"{last_acc:.3f}")

    torch.save({"model": model.state_dict(), "config": config.__dict__, "args": vars(args)}, args.output)
    print(f"saved={args.output} final_loss={last_loss:.4f} final_acc={last_acc:.3f}")


if __name__ == "__main__":
    main()
