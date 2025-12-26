# i3-80M

The 82.77M parameter hybrid language model combining RWKV-Mamba blocks with full attention layers.

---

## Model Overview

i3-80M is the second model in the i3 series, featuring a hybrid architecture optimized for efficient training on consumer hardware while maintaining strong performance.

### Architecture Specifications

| Feature                     | Value                          |
| --------------------------- | ------------------------------ |
| Total Parameters            | ~82.77M                        |
| Architecture                | 10 Hybrid + 6 Attention Layers |
| Context Window              | 256 tokens                     |
| Hidden Dimension            | 512                            |
| Attention Heads             | 16                             |
| State Dimension (`d_state`) | 32                             |
| Vocabulary Size             | 35,560 tokens                  |
| Tokenization Strategy       | 2–3 character chunks           |

### Layer Breakdown

```
Layers 1-10:  RWKV-Mamba Hybrid blocks (local context, efficient state management)
Layers 11-16: Full Multi-Head Attention (long-range dependencies, reasoning)
```

---

## Quick Start

### Training from Scratch

1. **Navigate to this directory**
   ```bash
   cd src/80m
   ```

2. **(Optional) Enable Weights & Biases tracking**
   ```bash
   export WANDB_API_KEY="your_key_here"
   ```

3. **Start training**
   ```bash
   python train.py
   ```

The training script will automatically:
- Download and process the datasets
- Build the vocabulary (cached for reuse)
- Train for 5,000 steps with checkpointing
- Save the final model to `i3-model-artifacts/`

### Running Inference

After training (or downloading a pretrained model), generate text using the inference script.

**Basic usage:**
```bash
python inference.py --model_dir "i3-model-artifacts"
```

**Customize generation parameters:**
```bash
python inference.py --model_dir "i3-model-artifacts" --temp 0.6 --length 200 --top_k 50
```

**Available arguments:**
- `--model_dir`: Path to model artifacts folder (required)
- `--temp`: Sampling temperature (default: 0.8)
  - Lower (0.3-0.6): More focused, deterministic
  - Higher (0.9-1.2): More creative, diverse
- `--top_k`: Sample from top K tokens (default: 50)
- `--length`: Number of tokens to generate (default: 100)

**Expected directory structure:**
```
src/80m/
├── train.py
├── inference.py
├── README.md              # This file
└── i3-model-artifacts/    # Created after training
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer.json
```

---

## Training Configuration

### Default Settings

- **Datasets**: 
  - `agentlans/high-quality-english-sentences` (diverse, high-quality English)
  - `roneneldan/TinyStories` (simple narratives, good for learning structure)
  - `starhopp3r/TinyChat` (conversational patterns)
- **Training Steps**: 5,000
- **Batch Size**: 4
- **Learning Rate**: 3e-4 with warmup and cosine decay
- **Optimizer**: AdamW with gradient clipping (norm 1.0)
- **Hardware Used**: NVIDIA P100 (16GB VRAM)
- **Framework**: PyTorch

### Training Efficiency

The model is designed for consumer hardware:
- **GPU Utilization**: 15–20%
- **VRAM Usage**: ~2.2GB allocated (peak ~18% of 16GB)
- **Power Draw**: ~40W
- **Throughput**: 100–550 tokens/second

### VRAM Usage by Context Window

Tested on Tesla P100-PCIE-16GB:

| Context Window | VRAM Usage | Status |
|----------------|------------|--------|
| 128 tokens     | 0.73 GB    | ✅ OK  |
| 256 tokens     | 0.75 GB    | ✅ OK  |
| 512 tokens     | 1.10 GB    | ✅ OK  |
| 1024 tokens    | 2.07 GB    | ✅ OK  |
| 2048 tokens    | 4.70 GB    | ✅ OK  |
| 4096 tokens    | 13.53 GB   | ✅ OK  |
| 8192 tokens    | N/A        | ❌ FAILED |

<img width="873" height="573" alt="image" src="https://github.com/user-attachments/assets/35d15dd6-aa02-4c75-9470-6c5685238f0e" />

**Key Insights:**
- Default 256 token context uses minimal VRAM (~0.75 GB)
- Linear growth up to 2048 tokens
- Exponential growth beyond 2048 tokens
- 16GB VRAM supports up to 4096 token context
- Ideal for consumer GPUs with 8GB+ VRAM

### Expected Training Progress

| Metric        | Initial | Final |
| ------------- | ------- | ----- |
| Training Loss | ~10.0   | ~1.7  |
| Perplexity    | ~4000+  | ~6    |

> **Note:** These metrics are from the published model at [huggingface.co/FlameF0X/i3-80m](https://huggingface.co/FlameF0X/i3-80m). Your results may vary based on hardware and exact dataset versions.

---

## Comparison with i3-22M

i3-80M represents a significant upgrade over the original model:

| Feature          | i3-22M                | i3-80M                                |
| ---------------- | --------------------- | ------------------------------------- |
| Parameters       | 22.6M                 | 82.77M                                |
| Architecture     | 24 Hybrid layers      | 10 Hybrid + 6 Attention               |
| Vocabulary       | 4,466 tokens          | 35,560 tokens                         |
| Training Data    | TinyChat only         | TinyStories + TinyChat + HQ Sentences |
| Training Time    | ~17 hours             | ~2–4 hours                            |
| Attention Layers | None                  | 6 Full Attention                      |

**Key improvements:**
- **Hybrid + Attention**: Better long-range reasoning with full attention layers
- **Larger vocabulary**: 8x more tokens for better text representation
- **Multi-dataset training**: More diverse and robust language understanding
- **Faster training**: More efficient training pipeline despite larger size

---

## Model Capabilities & Limitations

### What i3-80M Does Well
- Short to medium-form text generation
- Conversational responses (influenced by TinyChat)
- Simple narrative generation (from TinyStories)
- Coherent sentence structure (from HQ sentences dataset)

### Current Limitations
- **Language**: English only
- **Context**: 256 tokens (relatively short for complex tasks)
- **Domain**: General language; may need fine-tuning for specialized tasks
- **Style**: Conversational bias from training data

### Recommended Use Cases
- Chatbot prototypes
- Story generation experiments
- Language modeling research
- Testing hybrid architectures
- Educational purposes

---

## Model Files

After training, you'll find these files in `i3-model-artifacts/`:

- **`pytorch_model.bin`**: Model weights (~331 MB)
- **`config.json`**: Architecture configuration
- **`tokenizer.json`**: Vocabulary and tokenization rules

You can also download a pretrained version from [HuggingFace](https://huggingface.co/FlameF0X/i3-80m).

---

## Customization

### Modifying Training

Edit the hyperparameters in `train.py`:
- Change datasets to your own
- Adjust batch size for your hardware
- Modify learning rate schedule
- Increase/decrease training steps
- Enable/disable WandB logging

### Fine-tuning

To fine-tune on your own data:
1. Load the pretrained model weights
2. Replace datasets in training config
3. Use a lower learning rate (e.g., 1e-5)
4. Train for fewer steps

### Architecture Changes

To experiment with the architecture:
- Adjust number of hybrid vs attention layers
- Modify hidden dimension or number of heads
- Change state dimension for Mamba blocks
- Experiment with different context window sizes

---

## Troubleshooting

**Out of Memory (OOM) errors:**
- Reduce batch size in `train.py`
- Decrease context window size
- Enable gradient checkpointing (if implemented)

**Slow training:**
- Check GPU is being used (not CPU)
- Verify CUDA is properly installed
- Try reducing vocab size or dataset size

**Poor generation quality:**
- Train for more steps
- Adjust temperature during inference
- Try fine-tuning on domain-specific data

---

## Citation

If you use i3-80M in your research, please cite:

```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{RWKV,
  title={RWKV: Reinventing RNNs for the Transformer Era},
  author={Peng, Bo and others},
  journal={arXiv preprint arXiv:2305.13048},
  year={2023}
}
```
If you use this code, please cite:

```bibtex
@software{i3bert,
  title={i3-BERT: Hybrid RWKV-Transformer for Efficient Pre-training},
  author={FlameF0X},
  year={2024},
  url={https://github.com/FlameF0X/open-i3}
}
```
