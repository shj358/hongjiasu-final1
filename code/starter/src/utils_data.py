"""
final/starter/src/utils_data.py

Small helper utilities for downloading/loading approved datasets with consistent transforms.

Design goals:
- zero "data hunting"
- predictable storage location
- minimal dependencies (torch + torchvision)

Typical usage:
    from utils_data import get_torchvision_dataset

    train_ds, test_ds, info = get_torchvision_dataset(
        name="mnist",
        root="../../data/bigdata",
        download=True,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from torchvision import datasets, transforms
except Exception as e:  # pragma: no cover
    raise ImportError("utils_data.py requires torchvision. Install with: pip install torchvision") from e


@dataclass
class DatasetInfo:
    name: str
    root: str
    channels: int
    num_classes: int
    image_size: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


def _standard_transform(image_size: int, mean: Tuple[float, ...], std: Tuple[float, ...]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_torchvision_dataset(
    name: str,
    root: str = "./data/bigdata",
    download: bool = True,
    image_size: Optional[int] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, DatasetInfo]:
    """
    Supported names: mnist, fashion, cifar10

    Returns: train_ds, test_ds, info
    """
    name = name.lower().strip()
    base = Path(root).expanduser()

    if name == "mnist":
        ds_root = base / "MNIST"
        info = DatasetInfo(
            name="mnist",
            root=str(ds_root),
            channels=1,
            num_classes=10,
            image_size=28 if image_size is None else int(image_size),
            mean=(0.1307,),
            std=(0.3081,),
        )
        tfm = _standard_transform(info.image_size, info.mean, info.std)
        train_ds = datasets.MNIST(root=str(ds_root), train=True, transform=tfm, download=download)
        test_ds = datasets.MNIST(root=str(ds_root), train=False, transform=tfm, download=download)
        return train_ds, test_ds, info

    if name in {"fashion", "fashion-mnist", "fmnist"}:
        ds_root = base / "FashionMNIST"
        info = DatasetInfo(
            name="fashion-mnist",
            root=str(ds_root),
            channels=1,
            num_classes=10,
            image_size=28 if image_size is None else int(image_size),
            mean=(0.2860,),
            std=(0.3530,),
        )
        tfm = _standard_transform(info.image_size, info.mean, info.std)
        train_ds = datasets.FashionMNIST(root=str(ds_root), train=True, transform=tfm, download=download)
        test_ds = datasets.FashionMNIST(root=str(ds_root), train=False, transform=tfm, download=download)
        return train_ds, test_ds, info

    if name in {"cifar10", "cifar-10"}:
        ds_root = base / "CIFAR10"
        info = DatasetInfo(
            name="cifar10",
            root=str(ds_root),
            channels=3,
            num_classes=10,
            image_size=32 if image_size is None else int(image_size),
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )
        tfm = _standard_transform(info.image_size, info.mean, info.std)
        train_ds = datasets.CIFAR10(root=str(ds_root), train=True, transform=tfm, download=download)
        test_ds = datasets.CIFAR10(root=str(ds_root), train=False, transform=tfm, download=download)
        return train_ds, test_ds, info

    raise ValueError(f"Unknown dataset '{name}'. Supported: mnist, fashion, cifar10")
