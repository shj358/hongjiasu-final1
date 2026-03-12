#!/usr/bin/env python3
#%%
"""
DATA 37100 — Week 7 Lab: Diffusion Models (DDPM-lite)
File: lab07_diffusion_core.py

What this lab is:
- A small, readable DDPM-style diffusion implementation for MNIST/Fashion-MNIST.
- The model predicts noise eps_theta(x_t, t) with an MSE objective.

What this lab is NOT:
- A production U-Net with attention, EMA, fancy schedules, or FID eval.
  (Those are great extensions once the fundamentals are solid.)

Repo-safe, notebook-safe defaults:
- data_dir default: ../../data/bigdata/MNIST
- out_dir  default: ./untrack/outputs/lab07

Run (terminal):
  python lab07_diffusion_core.py --dataset mnist --epochs 1 --T 200 --bs 128
  python lab07_diffusion_core.py --dataset fashion --epochs 2 --T 400 --bs 128

Run (VSCode Interactive / Jupyter):
- Just run the file; argparse will ignore kernel-injected args safely.

Outputs:
- training log + checkpoints
- sample grids and denoising-intermediate grids

"""
from __future__ import annotations

import argparse
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid, save_image
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This lab requires torchvision. Install with: pip install torchvision"
    ) from e


# -----------------------------
# Args / config
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 7: Diffusion (DDPM-lite)")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--data-dir", type=str, default="../../data/bigdata/MNIST")
    p.add_argument("--out-dir", type=str, default="./untrack/outputs/lab07")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    # macOS uses multiprocessing "spawn" by default; locally-defined callables (e.g. lambdas)
    # can fail to pickle when DataLoader num_workers>0. Default to 0 to avoid friction.
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # diffusion
    p.add_argument("--T", type=int, default=200, help="Number of diffusion steps")
    p.add_argument("--beta1", type=float, default=1e-4)
    p.add_argument("--beta2", type=float, default=0.02)

    # model
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--time-emb-dim", type=int, default=128)

    # objective
    p.add_argument("--target", type=str, default="eps", choices=["eps", "x0"],
                   help="Training target: predict noise eps (DDPM default) or predict x0 directly.")

    # sampling
    p.add_argument("--sample-n", type=int, default=64)
    p.add_argument("--sample-every", type=int, default=400, help="iters between sample dumps")
    p.add_argument("--save-every", type=int, default=400, help="iters between ckpt dumps")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    # notebook-safe parse
    args, _ = p.parse_known_args()
    return args


def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Map [0,1] tensor -> [-1,1]. Top-level function so it's pickle-safe."""
    return x * 2 - 1


def get_device(device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto: prefer mps (Mac), then cuda, then cpu
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DiffusionSchedule:
    T: int
    betas: torch.Tensor        # [T]
    alphas: torch.Tensor       # [T]
    alphas_bar: torch.Tensor   # [T]
    sqrt_alphas_bar: torch.Tensor
    sqrt_one_minus_alphas_bar: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


def make_linear_schedule(T: int, beta1: float, beta2: float, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(beta1, beta2, T, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # posterior variance from DDPM paper:
    # Var(q(x_{t-1} | x_t, x0)) = beta_t * (1 - alphabar_{t-1}) / (1 - alphabar_t)
    alphas_bar_prev = torch.cat([torch.ones(1, device=device), alphas_bar[:-1]], dim=0)
    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

    return DiffusionSchedule(
        T=T,
        betas=betas,
        alphas=alphas,
        alphas_bar=alphas_bar,
        sqrt_alphas_bar=sqrt_alphas_bar,
        sqrt_one_minus_alphas_bar=sqrt_one_minus_alphas_bar,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
    )


# -----------------------------
# Model: small CNN with time embedding
# -----------------------------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] integer timesteps. returns [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(0, half, device=t.device).float() / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_proj = nn.Linear(tdim, ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TinyEpsModel(nn.Module):
    """A deliberately small epsilon-predictor CNN for 28x28 grayscale images."""

    def __init__(self, base_ch: int = 64, time_emb_dim: int = 128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        self.rb1 = ResBlock(base_ch, time_emb_dim)
        self.down = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 28->14
        self.rb2 = ResBlock(base_ch, time_emb_dim)
        self.mid = ResBlock(base_ch, time_emb_dim)
        self.up = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)  # 14->28
        self.rb3 = ResBlock(base_ch, time_emb_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.in_conv(x)
        h = self.rb1(h, t_emb)
        h = self.down(h)
        h = self.rb2(h, t_emb)
        h = self.mid(h, t_emb)
        h = self.up(h)
        h = self.rb3(h, t_emb)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# -----------------------------
# Data
# -----------------------------

def get_dataloader(dataset: str, data_dir: str, bs: int, num_workers: int) -> DataLoader:
    # IMPORTANT: DataLoader workers on macOS require all transform callables to be pickleable.
    # Avoid lambdas / nested functions here.
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_to_minus1_1)])  # [-1,1]
    root = Path(data_dir)
    if dataset == "mnist":
        ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    else:
        ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)

    if platform.system() == "Darwin" and num_workers > 0:
        # This keeps CLI behavior predictable even if students copy settings from Linux.
        print(f"[DataLoader] macOS detected → forcing num_workers=0 (was {num_workers})")
        num_workers = 0

    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)


# -----------------------------
# Diffusion helpers
# -----------------------------

def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, sch: DiffusionSchedule) -> torch.Tensor:
    """Sample x_t given x0."""
    # gather per-sample coefficients
    a = sch.sqrt_alphas_bar.gather(0, t).view(-1, 1, 1, 1)
    b = sch.sqrt_one_minus_alphas_bar.gather(0, t).view(-1, 1, 1, 1)
    return a * x0 + b * eps


@torch.no_grad()
def p_sample_step(
    model: nn.Module,
    x_t: torch.Tensor,
    t: int,
    sch: DiffusionSchedule,
    target: str = "eps",
) -> torch.Tensor:
    """One reverse step: sample x_{t-1} from x_t."""
    B = x_t.size(0)
    t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)

    out = model(x_t, t_tensor)

    # By default the model predicts eps_theta(x_t, t).
    # If target=="x0", interpret the output as x0_hat and convert to eps_hat.
    if target == "eps":
        eps_theta = out
    elif target == "x0":
        sqrt_ab = sch.sqrt_alphas_bar[t]
        sqrt_one_minus_ab = sch.sqrt_one_minus_alphas_bar[t]
        eps_theta = (x_t - sqrt_ab * out) / (sqrt_one_minus_ab + 1e-8)
    else:  # pragma: no cover
        raise ValueError(f"Unknown target: {target}")

    beta_t = sch.betas[t]
    sqrt_one_minus_ab = sch.sqrt_one_minus_alphas_bar[t]
    sqrt_recip_alpha = sch.sqrt_recip_alphas[t]
    ab_t = sch.alphas_bar[t]

    # DDPM mean: mu = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1-ab_t))*eps_theta)
    mean = sqrt_recip_alpha * (x_t - (beta_t / sqrt_one_minus_ab) * eps_theta)

    if t == 0:
        return mean

    var = sch.posterior_variance[t]
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(var) * noise


@torch.no_grad()
def sample(
    model: nn.Module,
    sch: DiffusionSchedule,
    n: int,
    device: torch.device,
    save_intermediates: bool = True,
    intermediate_ts: Tuple[int, ...] = (199, 149, 99, 49, 0),
    target: str = "eps",
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    inter: Dict[int, torch.Tensor] = {}

    for t in reversed(range(sch.T)):
        x = p_sample_step(model, x, t, sch, target=target)
        if save_intermediates and t in set(intermediate_ts):
            inter[t] = x.detach().cpu()

    return x.detach().cpu(), inter


# -----------------------------
# Train
# -----------------------------

def _fmt_float_tag(x: float) -> str:
    # Compact, filesystem-friendly float representation.
    s = f"{x:.3g}"
    # Normalize scientific notation like 2e-04 -> 2e-4
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    # Replace dots to avoid awkward folder names
    s = s.replace(".", "p")
    return s


def make_exp_tag(
    dataset: str,
    epochs: int,
    bs: int,
    lr: float,
    T: int,
    betas: Tuple[float, float],
    base_ch: int,
    target: str,
) -> str:
    b0, b1 = betas
    parts = [
        f"ds={dataset}",
        f"ep={epochs}",
        f"bs={bs}",
        f"lr={_fmt_float_tag(lr)}",
        f"T={T}",
        f"b={_fmt_float_tag(b0)}-{_fmt_float_tag(b1)}",
        f"ch={base_ch}",
    ]
    return "__".join(parts)


def make_run_dir(out_dir: str, exp_tag: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Keep runs unique (timestamp) but also searchable (exp_tag).
    folder = f"{exp_tag}__{ts}" if exp_tag else ts
    run_dir = Path(out_dir) / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    return run_dir



def save_grids(run_dir: Path, x: torch.Tensor, inter: Dict[int, torch.Tensor], step: int) -> None:
    # x is in [-1,1]; map to [0,1] for saving
    x01 = (x + 1) * 0.5
    grid = make_grid(x01, nrow=int(math.sqrt(x.size(0))), padding=2)
    save_image(grid, run_dir / "samples" / f"samples_step{step:06d}.png")

    if inter:
        # make a grid-of-grids: one row per timestep snapshot
        rows = []
        for t in sorted(inter.keys(), reverse=True):
            xi = (inter[t] + 1) * 0.5
            rows.append(make_grid(xi, nrow=int(math.sqrt(xi.size(0))), padding=2))
        big = torch.cat(rows, dim=1)  # stack vertically (C,H,W) concatenate along H
        save_image(big, run_dir / "samples" / f"denoise_steps_step{step:06d}.png")


def train_one_epoch(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    dl: DataLoader,
    sch: DiffusionSchedule,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    step0: int,
    sample_every: int,
    save_every: int,
    sample_n: int,
    target: str,
) -> int:
    model.train()
    step = step0
    for x0, _y in dl:
        x0 = x0.to(device)
        B = x0.size(0)

        t = torch.randint(0, sch.T, (B,), device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, sch)

        pred = model(x_t, t)
        if target == "eps":
            loss = F.mse_loss(pred, eps)
        elif target == "x0":
            loss = F.mse_loss(pred, x0)
        else:  # pragma: no cover
            raise ValueError(f"Unknown target: {target}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"epoch {epoch:02d} step {step:06d} | loss={loss.item():.4f}")

        if step % sample_every == 0:
            x_samp, inter = sample(model, sch, n=sample_n, device=device, intermediate_ts=_default_intermediate_ts(sch.T), target=target)
            save_grids(run_dir, x_samp, inter, step)

        if step % save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sch": {"T": sch.T, "beta1": float(sch.betas[0].item()), "beta2": float(sch.betas[-1].item())},
                "step": step,
                "epoch": epoch,
                "target": target,
            }
            torch.save(ckpt, run_dir / "checkpoints" / f"ckpt_step{step:06d}.pt")

        step += 1
    return step


def _default_intermediate_ts(T: int) -> Tuple[int, ...]:
    # choose a few roughly-spaced snapshots, always include 0
    pts = [T - 1, int(T * 0.75), int(T * 0.50), int(T * 0.25), 0]
    pts = [max(0, min(T - 1, p)) for p in pts]
    return tuple(sorted(set(pts), reverse=True))


def main() -> int:
    args = parse_args()
    seed_all(args.seed)
    device = get_device(args.device)
    print(f"[Device] {device}")
    print(f"[Config] dataset={args.dataset} epochs={args.epochs} bs={args.bs} lr={args.lr} T={args.T} betas=({args.beta1},{args.beta2}) base_ch={args.base_ch} target={args.target}")

    dl = get_dataloader(args.dataset, args.data_dir, args.bs, args.num_workers)

    exp_tag = make_exp_tag(args.dataset, args.epochs, args.bs, args.lr, args.T, (args.beta1, args.beta2), args.base_ch, args.target)
    run_dir = make_run_dir(args.out_dir, exp_tag)
    (run_dir / "run_config.txt").write_text(str(vars(args)))

    sch = make_linear_schedule(args.T, args.beta1, args.beta2, device=device)

    model = TinyEpsModel(base_ch=args.base_ch, time_emb_dim=args.time_emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    for epoch in range(1, args.epochs + 1):
        step = train_one_epoch(
            model=model,
            opt=opt,
            dl=dl,
            sch=sch,
            device=device,
            run_dir=run_dir,
            epoch=epoch,
            step0=step,
            sample_every=args.sample_every,
            save_every=args.save_every,
            sample_n=args.sample_n,
            target=args.target,
        )

    # final sample dump
    x_samp, inter = sample(model, sch, n=args.sample_n, device=device, intermediate_ts=_default_intermediate_ts(sch.T))
    save_grids(run_dir, x_samp, inter, step)

    print(f"[Done] outputs in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
