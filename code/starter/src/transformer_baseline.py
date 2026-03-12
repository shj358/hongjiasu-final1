#!/usr/bin/env python3
"""
DATA 37100 — Final Project Starter (Transformer)

A small, practical wrapper around the Week 5 tiny transformer core (week5/src/lab05_transformers_core.py).

Run (from repo root):
  python final/starter/src/transformer_baseline.py --steps 600 --sample-len 300

Controlled experiment (<= 6 runs, exactly 2 knobs):
  python final/starter/src/transformer_baseline.py --grid "temperature=0.3,0.8,1.2;top_p=0.85,0.95"

Outputs:
  ./untrack/outputs/final/transformer/<run_name>/
    - run_args.json
    - train_log.csv
    - sample.txt
    - checkpoint.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch


DEFAULT_TEXT = (
    "Transformers replace recurrence with attention.\n"
    "Self-attention mixes tokens using learned similarity.\n"
    "Causal masking prevents peeking at the future.\n"
    "LayerNorm + residuals stabilize training.\n"
    "Multi-head attention captures relations in parallel.\n"
    "Position encodings inject order information.\n"
)


@dataclass
class CharData:
    stoi: Dict[str, int]
    itos: Dict[int, str]
    train_ids: torch.Tensor
    val_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


def add_week5_src_to_path() -> None:
    # This file: <repo_root>/final/starter/src/transformer_baseline.py
    repo_root = Path(__file__).resolve().parents[3]
    week5_src = repo_root / "week5" / "src"
    if not week5_src.exists():
        raise FileNotFoundError(
            f"Could not find week5/src at: {week5_src}\n"
            "Run this from the course repo root with the expected layout."
        )
    sys.path.insert(0, str(week5_src))


def build_char_dataset(text: str, split: float = 0.9) -> CharData:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(split * len(ids))
    return CharData(stoi=stoi, itos=itos, train_ids=ids[:n], val_ids=ids[n:])


def make_get_batch(data: CharData, block_size: int):
    def get_batch(batch_size: int, split: str):
        ids = data.train_ids if split == "train" else data.val_ids
        ix = torch.randint(0, len(ids) - block_size - 1, (batch_size,))
        x = torch.stack([ids[i : i + block_size] for i in ix])
        y = torch.stack([ids[i + 1 : i + block_size + 1] for i in ix])
        return x, y
    return get_batch


@torch.no_grad()
def decode(data: CharData, ids: torch.Tensor) -> str:
    return "".join(data.itos[int(i)] for i in ids.tolist())


def parse_grid(s: str) -> Dict[str, List[str]]:
    s = (s or "").strip()
    if not s:
        return {}
    out: Dict[str, List[str]] = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Bad grid fragment '{part}'. Expected key=val1,val2,...")
        k, v = part.split("=", 1)
        k = k.strip()
        vals = [x.strip() for x in v.split(",") if x.strip()]
        if not k or not vals:
            raise ValueError(f"Bad grid fragment '{part}'. Expected key=val1,val2,...")
        out[k] = vals
    return out


def format_run_name(args: argparse.Namespace, overrides: Dict[str, str]) -> str:
    bits = [
        f"bsz-{args.block_size}",
        f"steps-{args.steps}",
        f"dm-{overrides.get('d_model', str(args.d_model))}",
        f"nh-{overrides.get('n_heads', str(args.n_heads))}",
        f"nl-{overrides.get('n_layers', str(args.n_layers))}",
        f"temp-{overrides.get('temperature', str(args.temperature))}",
        f"topP-{overrides.get('top_p', str(args.top_p))}",
    ]
    name = "_".join(bits)
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name)


def run_one(args: argparse.Namespace, overrides: Dict[str, str]) -> Path:
    add_week5_src_to_path()
    from lab05_transformers_core import TinyTransformerLM, TrainConfig, train_lm  # noqa: E402

    # merge overrides into args copy
    a = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        if k in {"repeat", "block_size", "steps", "batch_size", "d_model", "n_heads", "n_layers", "seed", "sample_len"}:
            setattr(a, k, int(v))
        elif k in {"lr", "dropout", "temperature", "top_p", "repetition_penalty"}:
            setattr(a, k, float(v))
        else:
            setattr(a, k, v)

    # text
    if a.text_file:
        text = Path(a.text_file).read_text(encoding="utf-8", errors="ignore")
    else:
        text = (DEFAULT_TEXT.strip() + "\n") * max(1, int(a.repeat))

    data = build_char_dataset(text)
    get_batch = make_get_batch(data, int(a.block_size))

    cfg = TrainConfig(
        steps=int(a.steps),
        batch_size=int(a.batch_size),
        lr=float(a.lr),
        eval_every=int(a.eval_every),
        eval_batches=int(a.eval_batches),
        grad_clip=float(a.grad_clip),
        seed=int(a.seed),
    )
    device = cfg.device

    model = TinyTransformerLM(
        vocab_size=data.vocab_size,
        block_size=int(a.block_size),
        d_model=int(a.d_model),
        n_heads=int(a.n_heads),
        n_layers=int(a.n_layers),
        dropout=float(a.dropout),
        use_positional_encoding=bool(a.use_positional_encoding),
        use_causal_mask=bool(a.use_causal_mask),
    )

    out_root = Path(a.out_dir).expanduser()
    run_dir = out_root / format_run_name(a, overrides)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(a), f, indent=2, sort_keys=True)

    # train
    t0 = time.time()
    hist = train_lm(model, get_batch, cfg)
    dt = time.time() - t0

    # save train log
    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        for step, tr, va in hist:
            w.writerow([step, tr, va])

    # checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "cfg": vars(cfg),
        "vocab": {"stoi": data.stoi, "itos": {str(k): v for k, v in data.itos.items()}},
        "args": vars(a),
    }
    torch.save(ckpt, run_dir / "checkpoint.pt")

    # sample
    model = model.to(device).eval()
    start_ch = "T" if "T" in data.stoi else next(iter(data.stoi.keys()))
    prompt = torch.tensor([[data.stoi[start_ch]]], dtype=torch.long, device=device)

    top_k = None if (a.top_k is None or int(a.top_k) <= 0) else int(a.top_k)
    top_p = None if (a.top_p is None or float(a.top_p) <= 0) else float(a.top_p)

    ids = model.generate(
        prompt,
        max_new_tokens=int(a.sample_len),
        temperature=float(a.temperature),
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=float(a.repetition_penalty),
        no_repeat_ngram_size=int(a.no_repeat_ngram_size),
    )[0].cpu()
    sample_text = decode(data, ids)

    (run_dir / "sample.txt").write_text(sample_text, encoding="utf-8")

    summary = {"seconds": round(dt, 2), "device": str(device), "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[Done] {run_dir.name} | {dt/60:.1f} min | device={device} | outputs: {run_dir}")
    return run_dir


def run_grid(args: argparse.Namespace) -> None:
    grid = parse_grid(args.grid)
    if not grid:
        raise ValueError('Empty grid. Example: --grid "temperature=0.3,0.8,1.2;top_p=0.85,0.95"')

    if len(grid) != 2:
        raise ValueError(
            f"Grid must specify exactly two knobs, got {len(grid)}: {list(grid.keys())}\n"
            'Example: --grid "temperature=0.3,0.8,1.2;top_p=0.85,0.95"'
        )

    keys = list(grid.keys())
    combos = list(__import__("itertools").product(*[grid[k] for k in keys]))
    if len(combos) > args.max_runs:
        raise ValueError(f"Grid expands to {len(combos)} runs; max_runs={args.max_runs}. Use fewer values.")

    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / "results.csv"

    rows = []
    for vals in combos:
        overrides = dict(zip(keys, vals))
        run_dir = run_one(args, overrides)
        row = {"run_dir": str(run_dir)}
        row.update(overrides)
        rows.append(row)

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_dir"] + keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[Grid complete] wrote: {results_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final Project Starter — Transformer Baseline")

    p.add_argument("--text-file", type=str, default="", help="Optional: path to a .txt file to train on")
    p.add_argument("--repeat", type=int, default=50, help="Repeat DEFAULT_TEXT this many times (if no --text-file)")

    p.add_argument("--block-size", dest="block_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)

    # model size knobs (keep modest)
    p.add_argument("--d-model", dest="d_model", type=int, default=128, choices=[64, 128, 256])
    p.add_argument("--n-heads", dest="n_heads", type=int, default=4, choices=[2, 4, 8])
    p.add_argument("--n-layers", dest="n_layers", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--use-positional-encoding", dest="use_positional_encoding", action="store_true")
    p.add_argument("--no-positional-encoding", dest="use_positional_encoding", action="store_false")
    p.set_defaults(use_positional_encoding=True)

    p.add_argument("--use-causal-mask", dest="use_causal_mask", action="store_true")
    p.add_argument("--no-causal-mask", dest="use_causal_mask", action="store_false")
    p.set_defaults(use_causal_mask=True)

    # training eval
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # generation knobs (good for analysis)
    p.add_argument("--sample-len", dest="sample_len", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", dest="top_k", type=int, default=40)
    p.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)

    p.add_argument("--out-dir", type=str, default="./untrack/outputs/final/transformer")

    # controlled experiment helper
    p.add_argument("--grid", type=str, default="", help='Example: "temperature=0.3,0.8,1.2;top_p=0.85,0.95"')
    p.add_argument("--max-runs", type=int, default=6)

    args, _ = p.parse_known_args()  # notebook-safe
    # normalize empty text_file
    if isinstance(args.text_file, str) and args.text_file.strip() == "":
        args.text_file = ""
    return args


def main() -> int:
    args = parse_args()
    if args.grid.strip():
        run_grid(args)
        return 0
    run_one(args, overrides={})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
