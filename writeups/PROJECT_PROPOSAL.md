# DATA 37100 — Final Project Proposal

## 1) Model families

Check **at least two**:

☐ GAN (DCGAN)  
☐ Transformer  
☐ Diffusion  

Briefly justify why these are appropriate for your question.

This project focuses on a diffusion model as the main experimental system. A GAN baseline is also included to demonstrate that multiple generative model families can run end-to-end in the project setup. Diffusion models are particularly suitable for controlled experiments because their generation process includes explicit parameters such as the number of denoising steps and the training prediction target.

---

## 2) Dataset

Which approved dataset will you use and why?

We use the MNIST dataset. MNIST is small and fast to train on, which makes it suitable for running multiple controlled experiments within a short time budget. The simplicity of the dataset also makes differences in sample quality and artifacts easier to observe visually.

---

## 3) Your core question (one sentence)

State one clear question about **model behavior** you will investigate.

Examples:
- “How do decoding controls trade off coherence vs. diversity in a tiny transformer?”
- “How does diffusion step count affect sample sharpness vs. artifacts?”
- “When does DCGAN collapse, and what stabilizes it?”

How do diffusion step count and prediction targets affect the visual quality and stability of generated MNIST images?

---

## 4) Controlled variables (two-knob study)

You must vary **exactly two** factors for your controlled experiment.

List:
- Knob 1: diffusion step count T
- Knob 2: prediction target ($\epsilon$ prediction vs $x_0$ prediction)

Planned settings (keep it small; ≤ 6 runs recommended):

T ∈ {100, 200, 400}
target ∈ {$\epsilon$, $x_0$}

---

## 5) Evidence & evaluation plan

Include:
- at least one quantitative signal (e.g., loss curves, simple metric, runtime, diversity proxy)
- qualitative inspection (sample grids / text snippets)

Evaluation will combine qualitative and simple quantitative signals. Generated image grids will be visually compared to analyze digit sharpness, structural clarity, and artifact patterns. Then, a simple diversity proxy based on average pairwise pixel-space distance will be computed to estimate sample diversity across runs. Finally, runtime statistics from each run will be recorded to compare computational cost across different diffusion step counts.

---

## 6) Risks & fallback plan

What might go wrong, and what will you do if it does?

One risk is that diffusion training may produce low-quality samples with limited training time. To mitigate this, the experiment uses a small dataset and a limited parameter grid to ensure all runs complete within the available compute budget. If training instability occurs, the experiment grid can be simplified by reducing the number of diffusion steps or runs while preserving the two-knob comparison.

---

**Instructor approval required before proceeding.**
