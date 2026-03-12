#!/usr/bin/env python3
"""
DATA 37100 — Final Project Starter (Diffusion)

A small, practical wrapper around the Week 7 diffusion core (week7/src/lab07_diffusion_core.py).

Run (from repo root):
  python final/starter/src/diffusion_baseline.py --dataset mnist --epochs 1 --T 200

Controlled experiment (<= 6 runs, exactly 2 knobs):
  python final/starter/src/diffusion_baseline.py --dataset mnist --epochs 1 --grid "T=100,200,400;target=eps,x0"

Outputs:
  ./untrack/outputs/final/diffusion/<run_name>/
    - run_args.json
    - train_log.csv
    - samples/ (PNG grids)
    - checkpoint.pt (optional)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

try:
    from torchvision import datasets, transforms
except Exception as e:  # pragma: no cover
    raise ImportError("diffusion_baseline.py requires torchvision. Install with: pip install torchvision") from e


def _resolve_data_root(data_dir: str) -> Path:
    """
    Torchvision MNIST/FashionMNIST expect:
      root/MNIST/raw  and  root/FashionMNIST/raw

    In this course repo, pre-downloaded data may live under:
      ./data/bigdata/MNIST/raw
      ./data/bigdata/FashionMNIST/raw

    So we default data_dir to ./data/bigdata.
    """
    root = Path(data_dir).expanduser()
    return root


def _dataset_exists(root: Path, dataset: str) -> bool:
    if dataset == "mnist":
        return (root / "MNIST" / "raw").exists() or (root / "MNIST" / "processed").exists()
    return (root / "FashionMNIST" / "raw").exists() or (root / "FashionMNIST" / "processed").exists()


def get_local_dataloader(dataset: str, data_dir: str, bs: int, num_workers: int, download: bool) -> "torch.utils.data.DataLoader":
    # Match Week 7 transform: ToTensor then map to [-1,1]
    root = _resolve_data_root(data_dir)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])

    if not download and not _dataset_exists(root, dataset):
        raise FileNotFoundError(
            f"Dataset files not found under: {root}\n"
            f"Expected e.g. {root}/MNIST/raw or {root}/FashionMNIST/raw\n"
            "Fix by either:\n"
            "  (1) pointing --data-dir to the correct folder, or\n"
            "  (2) running once with --download to fetch the dataset."
        )

    if dataset == "mnist":
        ds = datasets.MNIST(root=str(root), train=True, download=download, transform=tfm)
    else:
        ds = datasets.FashionMNIST(root=str(root), train=True, download=download, transform=tfm)

    # macOS worker guardrail (consistent with Week 7 core)
    import platform
    if platform.system() == "Darwin" and num_workers > 0:
        print(f"[DataLoader] macOS detected → forcing num_workers=0 (was {num_workers})")
        num_workers = 0

    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_device(device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto: prefer mps, then cuda, then cpu
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_grid(s: str) -> Dict[str, List[str]]:
    """
    Parse: "T=100,200,400;target=eps,x0" -> {"T":["100","200","400"], "target":["eps","x0"]}
    """
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
        f"ds-{args.dataset}",
        f"T-{overrides.get('T', str(args.T))}",
        f"target-{overrides.get('target', args.target)}",
        f"b2-{overrides.get('beta2', str(args.beta2))}",
        f"ch-{overrides.get('base_ch', str(args.base_ch))}",
    ]
    name = "_".join(bits)
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name)


def add_week7_src_to_path() -> None:
    # This file: <repo_root>/final/starter/src/diffusion_baseline.py
    repo_root = Path(__file__).resolve().parents[3]
    week7_src = repo_root / "code" / "core"
    if not week7_src.exists():
        raise FileNotFoundError(
            f"Could not find week7/src at: {week7_src}\n"
            "Run this from the course repo root with the expected layout."
        )
    sys.path.append(str(week7_src))


def run_single(args: argparse.Namespace, overrides: Dict[str, str]) -> Path:
    add_week7_src_to_path()
    import lab07_diffusion_core as core  # noqa: E402

    # copy args and apply overrides
    a = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        if k in {"T", "epochs", "bs", "num_workers", "seed", "base_ch", "time_emb_dim"}:
            setattr(a, k, int(v))
        elif k in {"lr", "beta1", "beta2"}:
            setattr(a, k, float(v))
        else:
            setattr(a, k, v)

    device = get_device(a.device)

    run_name = format_run_name(a, overrides)
    out_root = Path(a.out_dir).expanduser()
    run_dir = out_root / run_name
    _ensure_dir(run_dir / "samples")
    _ensure_dir(run_dir / "checkpoints")

    with open(run_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(a), f, indent=2, sort_keys=True)

    core.seed_all(a.seed)

    dl = get_local_dataloader(dataset=a.dataset, data_dir=a.data_dir, bs=a.bs, num_workers=a.num_workers, download=getattr(a, "download", False))
    sch = core.make_linear_schedule(T=a.T, beta1=a.beta1, beta2=a.beta2, device=device)

    model = core.TinyEpsModel(base_ch=a.base_ch, time_emb_dim=a.time_emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    step = 0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        step = core.train_one_epoch(
            model=model,
            opt=opt,
            dl=dl,
            sch=sch,
            device=device,
            run_dir=run_dir,
            epoch=epoch,
            step0=step,
            sample_every=a.sample_every,
            save_every=a.save_every,
            sample_n=a.sample_n,
            target=a.target,
        )
    dt = time.time() - t0

    with torch.no_grad():
        x_samp, inter = core.sample(
            model, sch, n=a.sample_n, device=device,
            intermediate_ts=core._default_intermediate_ts(sch.T),
            target=a.target
        )
        core.save_grids(run_dir, x_samp, inter, step)

    summary = {
        "run_name": run_name,
        "seconds": round(dt, 2),
        "device": str(device),
        "epochs": a.epochs,
        "T": a.T,
        "beta2": a.beta2,
        "base_ch": a.base_ch,
        "time_emb_dim": a.time_emb_dim,
        "target": a.target,
        "final_step": step,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[Done] {run_name} | {dt/60:.1f} min | outputs: {run_dir}")
    return run_dir


def run_grid(args: argparse.Namespace) -> None:
    grid = parse_grid(args.grid)
    if not grid:
        raise ValueError('Empty grid. Example: --grid "T=100,200;target=eps,x0"')

    if len(grid) != 2:
        raise ValueError(
            f"Grid must specify exactly two knobs, got {len(grid)}: {list(grid.keys())}\n"
            'Example: --grid "T=100,200,400;target=eps,x0"'
        )

    keys = list(grid.keys())
    combos = list(__import__("itertools").product(*[grid[k] for k in keys]))

    if len(combos) > args.max_runs:
        raise ValueError(
            f"Grid expands to {len(combos)} runs, exceeds max_runs={args.max_runs}. "
            "Use fewer values."
        )

    out_root = Path(args.out_dir).expanduser()
    _ensure_dir(out_root)
    results_csv = out_root / "results.csv"

    rows = []
    for vals in combos:
        overrides = dict(zip(keys, vals))
        run_dir = run_single(args, overrides)
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
    p = argparse.ArgumentParser(description="Final Project Starter — Diffusion Baseline")

    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--download", action="store_true", help="Download dataset if missing (default: off)")
    p.add_argument("--data-dir", type=str, default="./data/bigdata")
    p.add_argument("--out-dir", type=str, default="./untrack/outputs/final/diffusion")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--T", type=int, default=200)
    p.add_argument("--beta1", type=float, default=1e-4)
    p.add_argument("--beta2", type=float, default=0.02)

    p.add_argument("--base-ch", dest="base_ch", type=int, default=64)
    p.add_argument("--time-emb-dim", dest="time_emb_dim", type=int, default=128)

    p.add_argument("--target", type=str, default="eps", choices=["eps", "x0"])

    p.add_argument("--sample-n", type=int, default=64)
    p.add_argument("--sample-every", type=int, default=400)
    p.add_argument("--save-every", type=int, default=400)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--grid", type=str, default="", help='Example: "T=100,200,400;target=eps,x0"')
    p.add_argument("--max-runs", type=int, default=6, help="Hard cap on number of runs from --grid")

    args, _ = p.parse_known_args()  # notebook-safe
    return args


def main() -> int:
    args = parse_args()
    if args.grid.strip():
        run_grid(args)
        return 0
    run_single(args, overrides={})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
