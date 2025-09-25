# Fine-Tuning Hyperparameter Guide

This document explains the role of each key hyperparameter and CLI flag when fine-tuning large language models (LLMs) using mlx_lm or similar frameworks.

## Core Hyperparameters

### 1. Iterations (--iters)

Number of optimizer steps (forward + backward + weight update).

Determines total training time and how many times the dataset is seen.

How to choose:
* Small datasets 1,000–3,000 iterations.
* Medium datasets → 5,000–20,000 iterations.
* Large datasets → until validation loss plateaus.

Use early stopping when validation loss stops improving.

### 2. Batch Size (--batch-size)

Number of sequences processed per iteration.

Larger batch stable gradients, faster convergence, higher GPU RAM usage and smaller batch noisier gradients, works on limited hardware.

How to choose:
* Consumer GPUs → 2–8.
* A100/H100 GPUs → 64–256.

### 3. Learning Rate (--learning-rate)

Step size for weight updates.

Too high unstable, divergence and too low slow convergence.

How to choose:
* LoRA/DoRA → 1e-4 to 2e-4.
* Full fine-tuning → 1e-5 to 5e-5.

### 4. Mask Prompt (--mask-prompt)

Excludes prompt tokens from loss calculation.

Prevents model from wasting capacity reproducing the user query.
