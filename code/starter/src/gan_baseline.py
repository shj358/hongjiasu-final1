#!/usr/bin/env python3
"""
DATA 37100 — Final Project Starter (GAN / DCGAN)

A small, fast DCGAN baseline for MNIST / Fashion-MNIST.

Run (from repo root):
  python final/starter/src/gan_baseline.py --dataset fashionmnist --epochs 1

Controlled experiment (<= 6 runs, exactly 2 knobs):
  python final/starter/src/gan_baseline.py --dataset mnist --epochs 1 --grid "lr=0.0001,0.0002,0.0004;d_steps=1,2"

Outputs:
  ./untrack/outputs/final/gan/<run_name>/
    - run_args.json
    - train_log.csv
    - samples/ (PNG grids)
    - checkpoint.pt
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "torchvision is required for gan_baseline.py. "
        "Install it in your course environment."
    ) from e


# -----------------------------
# Device helper (fast + portable)
# -----------------------------
def pick_device(device_str: str) -> torch.device:
    s = (device_str or "auto").lower().strip()
    if s == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)


# -----------------------------
# DCGAN blocks (28x28, 1-channel)
# -----------------------------
class DCGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, base_ch: int = 64, out_ch: int = 1):
        super().__init__()
        # 1x1 -> 7x7 -> 14x14 -> 28x28
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base_ch * 4, 7, 1, 0, bias=False),  # 7x7
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),  # 14x14
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),  # 28x28
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),

            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, base_ch: int = 64, in_ch: int = 1):
        super().__init__()
        # 28x28 -> 14x14 -> 7x7 -> logits
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(base_ch * 4 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# -----------------------------
# Data
# -----------------------------
def get_dataset(name: str, data_dir: str, download: bool) -> Tuple[torch.utils.data.Dataset, int]:
    name = name.lower().strip()
    base = Path(data_dir).expanduser()

    # Keep outputs in [-1, 1] (DCGAN convention)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if name in ["mnist"]:
        root = base / "MNIST"
        ds = datasets.MNIST(root=str(root), train=True, download=download, transform=tfm)
        return ds, 1
    if name in ["fashion", "fashionmnist", "fmnist"]:
        root = base / "MNIST"
        ds = datasets.FashionMNIST(root=str(root), train=True, download=download, transform=tfm)
        return ds, 1

    raise ValueError("Unsupported dataset. Use: mnist | fashionmnist")


# -----------------------------
# Grid parsing (two knobs)
# -----------------------------
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


def iter_grid(grid: Dict[str, List[str]]) -> List[Dict[str, str]]:
    if not grid:
        return [{}]
    if len(grid) != 2:
        raise ValueError(f"Grid must specify exactly 2 knobs. Got keys={list(grid.keys())}")
    keys = list(grid.keys())
    combos: List[Dict[str, str]] = []
    for a, b in itertools.product(grid[keys[0]], grid[keys[1]]):
        combos.append({keys[0]: a, keys[1]: b})
    return combos


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Expected a float, got '{x}'") from e


def safe_int(x: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Expected an int, got '{x}'") from e


# -----------------------------
# Training
# -----------------------------
@dataclass
class RunPaths:
    run_dir: Path
    samples_dir: Path
    log_csv: Path
    args_json: Path
    ckpt: Path


def make_run_paths(run_name: str) -> RunPaths:
    out_base = Path("./untrack/outputs/final/gan")
    run_dir = out_base / run_name
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        samples_dir=samples_dir,
        log_csv=run_dir / "train_log.csv",
        args_json=run_dir / "run_args.json",
        ckpt=run_dir / "checkpoint.pt",
    )


@torch.no_grad()
def save_samples(G: nn.Module, device: torch.device, paths: RunPaths, step: int, z_dim: int, n: int = 64) -> Path:
    G.eval()
    z = torch.randn(n, z_dim, 1, 1, device=device)
    x = G(z)
    # x in [-1,1] -> [0,1] for saving
    img = (x + 1) / 2
    out = paths.samples_dir / f"grid_step{step:06d}.png"
    save_image(img, str(out), nrow=int(math.sqrt(n)), normalize=False)
    return out


def run_one(args: argparse.Namespace, overrides: Dict[str, str]) -> Path:
    device = pick_device(args.device)

    # knobs (defaults overridden by grid)
    lr = safe_float(overrides.get("lr", str(args.lr)))
    d_steps = safe_int(overrides.get("d_steps", str(args.d_steps)))

    z_dim = safe_int(overrides.get("z_dim", str(args.z_dim)))
    base_ch = safe_int(overrides.get("base_ch", str(args.base_ch)))

    run_name = (
        f"ds-{args.dataset}_ep-{args.epochs}_bs-{args.batch_size}_"
        f"lr-{lr}_dsteps-{d_steps}_z-{z_dim}_ch-{base_ch}"
    )
    paths = make_run_paths(run_name)

    # save args
    run_args = vars(args).copy()
    run_args.update({"lr": lr, "d_steps": d_steps, "z_dim": z_dim, "base_ch": base_ch})
    paths.args_json.write_text(json.dumps(run_args, indent=2))

    ds, in_ch = get_dataset(args.dataset, args.data_dir, args.download)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    G = DCGANGenerator(z_dim=z_dim, base_ch=base_ch, out_ch=in_ch).to(device)
    D = DCGANDiscriminator(base_ch=base_ch, in_ch=in_ch).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(args.beta1, args.beta2))

    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, z_dim, 1, 1, device=device)

    # logging
    with open(paths.log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "lossD", "lossG", "sec_per_step"])

        step = 0
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            for real, _ in dl:
                step += 1
                if args.max_steps > 0 and step > args.max_steps:
                    break

                real = real.to(device)
                bsz = real.size(0)

                # label smoothing (optional)
                real_y = torch.full((bsz,), 1.0 - args.label_smooth, device=device)
                fake_y = torch.zeros((bsz,), device=device)

                # -----------------
                # Train D (maybe multiple steps)
                # -----------------
                lossD_total = 0.0
                for _ in range(d_steps):
                    optD.zero_grad(set_to_none=True)

                    # real
                    logits_real = D(real)
                    loss_real = bce(logits_real, real_y)

                    # fake
                    z = torch.randn(bsz, z_dim, 1, 1, device=device)
                    fake = G(z).detach()
                    logits_fake = D(fake)
                    loss_fake = bce(logits_fake, fake_y)

                    lossD = loss_real + loss_fake
                    lossD.backward()
                    optD.step()
                    lossD_total += float(lossD.detach().cpu())

                lossD_avg = lossD_total / max(1, d_steps)

                # -----------------
                # Train G
                # -----------------
                optG.zero_grad(set_to_none=True)
                z = torch.randn(bsz, z_dim, 1, 1, device=device)
                fake = G(z)
                logits = D(fake)
                lossG = bce(logits, real_y)  # want D(fake)=1
                lossG.backward()
                optG.step()

                # log + sample
                t1 = time.time()
                sec_per_step = (t1 - t0) / max(1, step)
                w.writerow([step, epoch, f"{lossD_avg:.6f}", f"{float(lossG.detach().cpu()):.6f}", f"{sec_per_step:.4f}"])

                if step == 1 or (args.sample_every > 0 and step % args.sample_every == 0):
                    G.eval()
                    with torch.no_grad():
                        x = G(fixed_z)
                        img = (x + 1) / 2
                        out = paths.samples_dir / f"grid_step{step:06d}.png"
                        save_image(img, str(out), nrow=8, normalize=False)

                if args.print_every > 0 and step % args.print_every == 0:
                    print(f"[GAN] ep {epoch:02d} step {step:05d} | lossD={lossD_avg:.3f} lossG={float(lossG):.3f} | {device}")

            if args.max_steps > 0 and step > args.max_steps:
                break

    # checkpoint
    torch.save(
        {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "args": run_args,
        },
        paths.ckpt,
    )

    return paths.run_dir


def run_grid(args: argparse.Namespace) -> None:
    grid = parse_grid(args.grid)
    combos = iter_grid(grid)
    out_base = Path("./untrack/outputs/final/gan")
    out_base.mkdir(parents=True, exist_ok=True)
    results_csv = out_base / "results.csv"

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "run_dir", "knob1", "val1", "knob2", "val2"])
        for overrides in combos:
            run_dir = run_one(args, overrides)
            k1, k2 = list(overrides.keys())
            w.writerow([run_dir.name, str(run_dir), k1, overrides[k1], k2, overrides[k2]])

    print(f"[OK] Wrote grid manifest: {results_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DATA 37100 final project DCGAN baseline (fast).")
    p.add_argument("--dataset", type=str, default="fashionmnist", help="mnist | fashionmnist")
    p.add_argument("--data-dir", type=str, default="./data/bigdata", help="Repo data root (contains MNIST/).")
    p.add_argument("--download", action="store_true", help="Download dataset if missing.")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=400, help="Hard cap for speed (0 = no cap).")
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--d-steps", type=int, default=1, help="Discriminator updates per generator step.")
    p.add_argument("--label-smooth", type=float, default=0.0, help="e.g., 0.1 sets real label to 0.9.")

    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--base-ch", type=int, default=64)

    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=100)

    p.add_argument("--grid", type=str, default="", help='Two-knob grid: "k1=v1,v2; k2=v1,v2"')
    return p


def main() -> int:
    args = build_argparser().parse_args()

    if args.grid.strip():
        run_grid(args)
        return 0

    run_dir = run_one(args, overrides={})
    print(f"[OK] Outputs in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
