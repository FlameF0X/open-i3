# i3-200M (Redherring)

## Overview

**i3-200M (Redherring): Hybrid Linear–Quadratic LLM** ("i3" is a name only; it has no semantic meaning) is an open-source language model architecture designed to bridge the gap between the infinite-context potential of RNNs and the high-precision recall of Transformers.

It implements a **Hybrid RWKV–Attention stack**, leveraging:

* **12 layers of RWKV v4** for efficient state building
* **4 layers of standard multi-head attention** for global context refinement

---

## Key Features

* **Hybrid Architecture**
  RWKV layers provide (O(L)) scaling at the base, while attention layers provide (O(L^2)) precision at the top.

* **JIT-Optimized Kernels**
  Custom PyTorch JIT scripts for the WKV (Weight–Key–Value) computation ensure high-performance execution.

* **Open-Source Ready**
  Modular configuration, environment-based secret management, and HuggingFace-compatible export formats.

* **Efficient Recurrence**
  Infinite context potential via linear state-space dynamics without the memory explosion of full-attention models.

---

## Architecture Detail

### The Hybrid Strategy

Standard Transformers suffer from quadratic memory growth with long sequences. Purely recurrent models (e.g., RWKV, Mamba) scale linearly but can experience *state compression*, losing fine-grained early-sequence details.

i3-200M (Redherring) mitigates this by stacking:

* **Base (Layers 1–12): RWKV Blocks**
  Process the bulk of the sequence, compressing history into a hidden state that acts as fast memory.

* **Top (Layers 13–16): Attention Blocks**
  Perform a final global sweep over refined features, enabling high-fidelity recall before next-token prediction.

---

## Under the Hood: WKV Recurrence

The core of the RWKV layer is the WKV kernel:

<img width="260" height="58" alt="image" src="https://github.com/user-attachments/assets/171822ac-bfe6-4f0f-9045-4c596c834c88" />

This formulation enables stable, linear-time recurrence while preserving token-level influence over long horizons.

---

## Setup & Installation

### Prerequisites

* Python 3.8+
* PyTorch 2.0+
* CUDA-enabled GPU (strongly recommended)

### Installation

```bash
git clone https://github.com/FlameF0X/open-i3.git
cd i3-200m
pip install torch numpy wandb datasets tokenizers
```

### Environment Variables

For secure logging, set WandB credentials:

```bash
export WANDB_API_KEY="your_api_key_here"
export WANDB_PROJECT="i3-200m-redherring"
```

---

## Training

To start training on the default dataset (TinyStories):

```bash
python train.py
```

Hyperparameters such as `d_model`, `lr`, and layer counts can be modified in the `TrainingConfig` class inside `train.py`.

---

## Project Structure

* `train.py` — Main model, tokenizer, and training loop
* `tokenizer.json` — Generated BPE vocabulary (after first run)
* `checkpoints/` — Saved `.pt` model weights

---

## License

MIT License. See the `LICENSE` file for details.
