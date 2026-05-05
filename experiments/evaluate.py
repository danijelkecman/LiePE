from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import torch

from liepe.models.transformer import TinyGPT, TransformerConfig
from liepe.tasks import TASKS
from liepe.utils.seed import set_seed


@torch.no_grad()
def eval_once(model, make_batch, batch_size, seq_len, vocab_size, device, batches=10):
    losses = []
    accs = []
    model.eval()
    for _ in range(batches):
        x, y = make_batch(batch_size, seq_len, vocab_size, device)
        logits, loss = model(x, y)
        losses.append(loss.item())
        accs.append((logits.argmax(dim=-1) == y).float().mean().item())
    return sum(losses) / len(losses), sum(accs) / len(accs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS.keys(), default="copy_delay")
    parser.add_argument("--encoding", default="rope", choices=["nope", "rope", "alibi", "decay", "damped_rope", "jordan", "learned_lie"])
    parser.add_argument("--train-seq-len", type=int, default=64)
    parser.add_argument("--eval-seq-lens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="outputs/eval_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.train_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        encoding=args.encoding,
    )
    model = TinyGPT(config).to(args.device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model"])

    rows = []
    make_batch = TASKS[args.task]
    for seq_len in args.eval_seq_lens:
        loss, acc = eval_once(model, make_batch, args.batch_size, seq_len, args.vocab_size, args.device)
        rows.append({"task": args.task, "encoding": args.encoding, "seq_len": seq_len, "loss": loss, "accuracy": acc})
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
