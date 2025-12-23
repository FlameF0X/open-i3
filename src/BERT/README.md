# i3-BERT: Hybrid RWKV-Transformer for Efficient Pre-training

A novel hybrid language model architecture combining the efficiency of RWKV's linear attention with the global reasoning capabilities of standard transformers, designed for BERT-style masked language modeling tasks.

## Architecture Overview

**i3-BERT** implements a two-tier architecture:

- **Bottom Layers (Bi-RWKV)**: Process local context efficiently using bidirectional RWKV blocks with O(T) complexity
- **Top Layers (Full Attention)**: Perform global reasoning and long-range dependencies with O(T²) multi-head attention

This design philosophy leverages the strengths of both approaches: RWKV handles syntactic structure and local patterns efficiently, while attention layers enable global information retrieval and complex reasoning.

## Key Features

- **Bidirectional RWKV**: Novel implementation running RWKV in both forward and backward directions for non-causal tasks
- **JIT-Optimized WKV Kernel**: Compiled linear attention mechanism for faster training
- **Hybrid Layer Stack**: Configurable ratio of RWKV to attention layers
- **Standard BERT Pre-training**: MLM (Masked Language Modeling) + NSP (Next Sentence Prediction)
- **Streaming Data Pipeline**: Handles large datasets without memory issues
- **116M Parameters**: Educational-scale model suitable for consumer GPUs

## Model Configuration

```python
class ModelConfig:
    d_model = 768              # Hidden dimension
    n_rwkv_layers = 4         # Bottom RWKV layers
    n_attn_layers = 4         # Top attention layers
    n_heads = 12              # Attention heads
    seq_len = 128             # Sequence length
    batch_size = 32
    lr = 2e-4
    max_iters = 5000
    vocab_size = 30000
```

## Installation

```bash
pip install torch numpy datasets tokenizers wandb
```

## Quick Start

### 1. Set Environment Variables

```bash
export WANDB_API_KEY="your_wandb_key"
```

### 2. Run Training

```bash
python train.py
```

The script will:
- Train or load a BPE tokenizer (30K vocab)
- Stream data from HuggingFace datasets
- Pre-train the hybrid model
- Log metrics to Weights & Biases
- Save checkpoints to `i3_bert_checkpoints/`

## Architecture Details

### Bi-RWKV Block

The core innovation enabling RWKV for bidirectional tasks:

```python
# Forward pass (left-to-right)
x_fwd = self.fwd_time_mix(x_norm)

# Backward pass (right-to-left)
x_rev = torch.flip(x_norm, [1])
x_bwd_rev = self.bwd_time_mix(x_rev)
x_bwd = torch.flip(x_bwd_rev, [1])

# Fusion
x = x + x_fwd + x_bwd
```

### RWKV Time Mixing

Linear attention mechanism with learned decay:

- **Time Shifting**: Mixes current token with previous token
- **WKV Kernel**: Maintains running state with exponential decay
- **Complexity**: O(T) instead of O(T²)

### Layer Stack

```
Input → Embeddings (Word + Position + Segment)
  ↓
[Bi-RWKV Block] × n_rwkv_layers  ← Efficient local processing
  ↓
[Attention Block] × n_attn_layers ← Global reasoning
  ↓
MLM Head + NSP Head → Outputs
```

## Training Objectives

### Masked Language Modeling (MLM)

- 15% of tokens are selected for masking
  - 80% replaced with `[MASK]`
  - 10% replaced with random token
  - 10% unchanged
- Model predicts original token

### Next Sentence Prediction (NSP)

- 50% real consecutive sentences (IsNext)
- 50% random sentence pairs (NotNext)
- Binary classification task

## Dataset Management

The `BertDatasetManager` streams data efficiently:

```python
# Streams from HuggingFace
datasets = ['HuggingFaceFW/fineweb-edu']

# Buffers sentences
# Generates NSP pairs
# Applies dynamic MLM masking
# Handles padding and segmentation
```

## Tokenizer

BPE tokenizer with special tokens:

- `<UNK>` (0): Unknown
- `<PAD>` (1): Padding
- `<CLS>` (2): Classification token
- `<SEP>` (3): Separator
- `<MASK>` (4): Mask token

## Monitoring

Training metrics logged to W&B:

- Total loss (MLM + NSP)
- MLM loss
- NSP loss
- Sample predictions every 500 iterations

## Model Checkpoints

Saved to `i3_bert_checkpoints/i3_bert_model_116m.pt`

Load for fine-tuning:

```python
model = i3BertModel(vocab_size=30000, ...)
model.load_state_dict(torch.load("i3_bert_checkpoints/i3_bert_model_116m.pt"))
```

## Performance Characteristics

**Memory Efficiency**:
- RWKV layers: O(T) memory
- Attention layers: O(T²) memory
- Overall: Reduced memory vs full transformer

**Speed**:
- JIT-compiled WKV kernel
- Faster than equivalent full-attention model
- Trade-off between efficiency (RWKV) and capability (attention)
- 
## Benchmark

|Metric                    | i3-BERT-v1      | i3-BERT-v2     |
|--------------------------|-----------------|---------------|
|Perplexity                | 89611.9056      | 1031307.4302 |  
|NSP Accuracy              | 49.55%          | 50.64%      |   
|MLM Loss                  | 11.4032         | 13.8463    |    
|NSP Loss                  | 0.7417          | 1.7865    |     
|Throughput (tokens/sec)   | 25173           | 26916    |

## Customization

### Adjust Architecture

```python
# More RWKV-heavy (faster, more efficient)
n_rwkv_layers = 8
n_attn_layers = 2

# More attention-heavy (stronger global reasoning)
n_rwkv_layers = 2
n_attn_layers = 8
```

### Scale Model Size

```python
# Larger model
d_model = 1024
n_heads = 16

# Smaller model
d_model = 512
n_heads = 8
```

## Research Context

This implementation explores hybrid architectures combining:

- **RWKV v4**: Linear attention with learned decay mechanisms
- **Transformers**: Standard multi-head attention
- **BERT**: Bidirectional pre-training objectives

The goal is to find optimal balance between efficiency and capability for language understanding tasks.

## Citation

If you use this code, please cite:

```bibtex
@software{i3bert,
  title={i3-BERT: Hybrid RWKV-Transformer for Efficient Pre-training},
  author={FlameF0X},
  year={2025},
  url={https://github.com/FlameF0X/open-i3}
}
```

## Acknowledgments

- RWKV architecture inspired by [RWKV Language Model](https://github.com/BlinkDL/RWKV-LM)
- BERT pre-training objectives from [Devlin et al. 2018](https://arxiv.org/abs/1810.04805)
- Built with PyTorch, HuggingFace Datasets, and Tokenizers
