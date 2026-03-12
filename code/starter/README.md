# Final Project — Starter Code (Optional)

This folder contains optional starter code to help you run small, reproducible baselines quickly.

You are **not required** to use this code, but it is recommended if you want to stay within the expected project scope (~20 hours).

---

## Folder structure

    final/starter/
    ├── README.md
    └── src/
        ├── gan_baseline.py
        ├── diffusion_baseline.py
        ├── transformer_baseline.py
        └── utils_data.py

Run commands from the **repository root**.

---

## GAN baseline (DCGAN) — Week 4

Baseline run:

```bash
python final/starter/src/gan_baseline.py --dataset fashionmnist --epochs 1
```

Controlled experiment (exactly two knobs, ≤ 6 runs):

```bash
python final/starter/src/gan_baseline.py --dataset mnist --epochs 1 --grid "lr=0.0001,0.0002,0.0004;d_steps=1,2"
```

Outputs:

    ./untrack/outputs/final/gan/

---

## Transformer baseline — Week 5

Baseline run:

```bash
python final/starter/src/transformer_baseline.py --steps 200 --sample-len 200
```

Controlled experiment (exactly two knobs, ≤ 6 runs):

```bash
python final/starter/src/transformer_baseline.py --grid "temperature=0.3,0.8,1.2;top_p=0.85,0.95"
```

Outputs:

    ./untrack/outputs/final/transformer/

---

## Diffusion baseline — Week 7

Baseline run:

```bash
python final/starter/src/diffusion_baseline.py --dataset mnist --epochs 1 --T 200
```

Controlled experiment (exactly two knobs, ≤ 6 runs):

```bash
python final/starter/src/diffusion_baseline.py --dataset mnist --epochs 1 --grid "T=100,200,400;target=eps,x0"
```

Outputs:

    ./untrack/outputs/final/diffusion/

---

## Dataset helper

`utils_data.py` provides standardized loading for approved datasets.

Example:

```python
from utils_data import get_torchvision_dataset
train_ds, test_ds, info = get_torchvision_dataset("mnist", root="./data/bigdata", download=True)
print(info)
```

### Data location note

In this repo, pre-downloaded MNIST/FashionMNIST files may live under:

```
./data/bigdata/MNIST/
  MNIST/raw
  FashionMNIST/raw
```

If your data lives elsewhere, pass `--data-dir`.
