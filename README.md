# DATA 37100 Final Project
Hongjia Su

## Project Description

This project studies how diffusion generation parameters affect the visual structure and stability of generated MNIST images.

The main controlled experiment varies two diffusion parameters:

- diffusion step count T
- prediction target (ε prediction vs x₀ prediction)

Using a small six-run experiment grid, the project compares generated sample quality, artifact patterns, runtime, and a simple diversity proxy.

In addition to the diffusion experiments, baseline runs from two generative model families are included to satisfy the project requirement and to verify that the training setup can produce end-to-end generative results.

## Environment Setup

```bash
conda create -n data371 python=3.10
conda activate data371
pip install -r requirements.txt
```

## Baseline Runs

```bash
python code/starter/src/gan_baseline.py \
    --dataset mnist \
    --epochs 1 \
    --download

python code/starter/src/diffusion_baseline.py \
    --dataset mnist \
    --epochs 1 \
    --download
```

## Controlled Experiment

```bash
python code/starter/src/diffusion_baseline.py \
    --dataset mnist \
    --epochs 1 \
    --download \
    --grid "T=100,200,400;target=eps,x0"
```

## Analysis

After running the experiments, open the analysis notebook:

analysis/final_project_analysis_template.ipynb

The notebook loads experiment outputs from `untrack/outputs/` and reproduces:

- sample comparison grids
- diversity proxy table
- runtime summary

## Notes

The training scripts are adapted from the course starter code.
Minor modifications were made to organize the repository structure and output paths.
The final report is available at writeups/report.pdf.