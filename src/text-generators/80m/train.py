import os
import time
import json
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from datasets import load_dataset

# ============================= #
# WANDB Configuration & Secrets # 
# ============================= #
# NOTE: If you want to hardcode it for private runs, replace the string below.
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "YOUR_WANDB_API_KEY_HERE")
WANDB_PROJECT = "i3"
WANDB_ENTITY = None 

def init_wandb(config):
    """Initialize WandB with API key and configuration"""
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=config,
        name=f"i3-d{config['d_model']}-l{config['n_layers']}-{time.strftime('%Y%m%d-%H%M%S')}",
        tags=["i3", "hybrid-architecture", "standard-linear", "multidataset"],
        notes="Hybrid i3 model (10 Conv + 6 Attn) trained on TinyStories, TinyChat, and HQ Sentences"
    )
    
    wandb.config.update({
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__
    })
    
    return wandb

# =============================================== #
# Core Architecture: RWKV-Mamba Hybrid Recurrence #
# =============================================== #
class RWKVMambaHybrid(nn.Module):
    """
    Combines RWKV time-mixing with Mamba state-space dynamics.
    
    UNDER THE HOOD:
    1. RWKV Mixing: Linearly interpolates between the current input (x_t) and 
       the running history (h) using a learnable weight (w_mix). This acts as a 
       'decaying' memory of the immediate past.
       
    2. Mamba State (SSM): Maintains a latent state 's'.
       - A: State transition matrix (how much of the old state to keep).
       - B: Input control matrix (how much of the new input enters the state).
       - C: Output projection (how we read the state).
       
    This hybrid approach allows the model to have 'recurrence' (infinite context window 
    potential) without the VRAM cost of a full attention matrix.
    """
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # RWKV mixing parameter
        self.w_mix = nn.Parameter(torch.ones(d_model) * 0.5)
        
        # Mamba SSM parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1) # Passthrough connection
    
    def forward(self, x):
        B, T, C = x.shape
        # Initialize hidden states
        h = torch.zeros(B, C, device=x.device)
        s = torch.zeros(B, self.d_state, device=x.device)
        outputs = []
        
        for t in range(T):
            x_t = x[:, t, :]
            
            # 1. RWKV-style time mixing
            # Interpolate current input with history buffer
            h = self.w_mix * h + (1 - self.w_mix) * x_t
            
            # 2. Mamba-style state update
            # s' = sA + xB
            s = s @ self.A.T + x_t @ self.B.T
            
            # 3. Output projection
            # y = sC + hD
            y_t = s @ self.C.T + h * self.D
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)

# ====================================== #
# Core Architecture: Attention Mechanism #
# ====================================== #
class FullAttention(nn.Module):
    """
    Standard Multi-Head Attention (MHA).
    
    Used in the upper 6 layers of i3 to provide "global" context awareness 
    that recurrent layers sometimes miss. Uses proper mask broadcasting for 
    causal (autoregressive) generation.
    """
    def __init__(self, d_model, n_heads=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Projections
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for heads: [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            # Broadcast mask: [1, 1, T, T] -> [B, n_heads, T, T]
            mask = mask.expand(B, self.n_heads, T, T).bool()
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        # Reassemble heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# =============== #
# i3 Model Blocks # 
# =============== #
class i3HybridBlock(nn.Module):
    """
    Lower Layers: RWKV-Mamba Hybrid Recurrence + FFN.
    These layers handle local context and state maintenance efficiently.
    (No attention mask needed here).
    """
    def __init__(self, d_model, d_state=64, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.hybrid = RWKVMambaHybrid(d_model, d_state)
        
        self.ln2 = nn.LayerNorm(d_model)
        d_ff = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x, mask=None):
        x = x + self.hybrid(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class i3AttentionBlock(nn.Module):
    """
    Upper Layers: Full Attention + FFN.
    These layers refine the output using global context looking back at the 
    entire sequence.
    """
    def __init__(self, d_model, n_heads=16, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FullAttention(d_model, n_heads)
        
        self.ln2 = nn.LayerNorm(d_model)
        d_ff = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

# =================== #
# Main Model: i3Model #
# =================== #
class i3Model(nn.Module):
    """
    Hybrid LLM Architecture
    Structure: 10 Conv/Hybrid blocks -> 6 Attention blocks.
    Total: 16 Layers.
    """
    def __init__(self, vocab_size, d_model=2048, n_heads=16, 
                 max_seq_len=512, d_state=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # --- Layer Construction ---
        # 10 Hybrid layers (Computationally cheap, good for state)
        hybrid_layers = [
            i3HybridBlock(d_model, d_state=d_state, ffn_mult=4)
            for _ in range(10)
        ]
        
        # 6 Attention layers (Computationally expensive, good for recall)
        attention_layers = [
            i3AttentionBlock(d_model, n_heads=n_heads, ffn_mult=4)
            for _ in range(6)
        ]
        
        self.layers = nn.ModuleList(hybrid_layers + attention_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos_embed(pos)
        
        # Causal mask for the attention layers (lower-left triangle)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        for layer in self.layers:
            # Hybrid blocks ignore the mask; Attention blocks use it.
            x = layer(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation loop"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# ==================================================== #
# Tokenizer: Memory-Optimized Variable-Length Chunking #
# ==================================================== #
class ChunkTokenizer:
    """
    A custom tokenizer that avoids the memory overhead of BPE training.
    
    STRATEGY:
    Instead of full BPE, it streams through text and identifies frequent 
    variable-length chunks (2-3 chars). This is critical for training on 
    consumer hardware where loading a massive dataset into RAM to build a 
    vocab would crash the process.
    """
    def __init__(self, vocab_only=False):
        self.chunk_to_idx = {}
        self.idx_to_chunk = {}
        self.vocab_size = 0
        
        self.unk_token = '<UNK>'
        self.unk_idx = 0
        
        # Pre-computed common English trigrams for O(1) lookup speedup
        self.common_trigrams = frozenset({
            'the', 'and', 'ing', 'ion', 'tio', 'for', 'tha', 
            'ter', 'hat', 'his', 'ere', 'ent', 'her', 'was',
            'you', 'are', 'not', 'but', 'can', 'all', 'whi',
            'one', 'our', 'out', 'whe', 'hav', 'thi', 'wit'
        })
    
    def _should_use_3char(self, pos, text):
        """Heuristic to decide if we grab 3 chars or 2"""
        if pos + 3 > len(text):
            return False
        
        chunk_3 = text[pos:pos+3]
        
        # Priority 1: Is it a known common trigram?
        if chunk_3 in self.common_trigrams:
            return True
        
        # Priority 2: Is it surrounded by spaces? (likely a short word)
        if pos > 0 and text[pos-1] == ' ':
            return True
        
        if pos + 3 < len(text) and text[pos+3] == ' ':
            return True
        
        return False
    
    def build_vocab_from_texts(self, all_texts, max_samples=None):
        print("Building vocabulary in streaming mode (memory-efficient)...")
        chunk_freq = {}
        
        for idx, text in enumerate(all_texts):
            if max_samples and idx >= max_samples:
                break
            
            if idx % 1000 == 0:
                print(f"Processed {idx} samples, unique chunks: {len(chunk_freq)}")
                if wandb.run is not None:
                    wandb.log({
                        "vocab_building/samples_processed": idx,
                        "vocab_building/unique_chunks": len(chunk_freq)
                    })
            
            text = text.lower()
            pos = 0
            
            while pos < len(text):
                if self._should_use_3char(pos, text):
                    chunk_len = min(3, len(text) - pos)
                else:
                    chunk_len = min(2, len(text) - pos)
                
                chunk = text[pos:pos+chunk_len]
                chunk_freq[chunk] = chunk_freq.get(chunk, 0) + 1
                pos += chunk_len
        
        print(f"\nTotal unique chunks found: {len(chunk_freq)}")
        
        sorted_chunks = sorted(chunk_freq.items(), key=lambda x: (-x[1], x[0]))
        
        # 0 is UNK
        self.chunk_to_idx = {self.unk_token: self.unk_idx}
        self.idx_to_chunk = {self.unk_idx: self.unk_token}
        
        for idx, (chunk, _) in enumerate(sorted_chunks, start=1):
            self.chunk_to_idx[chunk] = idx
            self.idx_to_chunk[idx] = chunk
        
        self.vocab_size = len(self.chunk_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")
        
        del chunk_freq # Free memory
    
    def encode(self, text):
        """Stream encoding - converts text to ints without storing intermediates"""
        text = text.lower()
        pos = 0
        indices = []
        
        while pos < len(text):
            if self._should_use_3char(pos, text):
                chunk_len = min(3, len(text) - pos)
            else:
                chunk_len = min(2, len(text) - pos)
            
            chunk = text[pos:pos+chunk_len]
            
            if chunk in self.chunk_to_idx:
                indices.append(self.chunk_to_idx[chunk])
                pos += chunk_len
            else:
                # Fallback mechanism: try smaller chunks if the big one fails
                found = False
                for cl in range(chunk_len - 1, 0, -1):
                    sub_chunk = chunk[:cl]
                    if sub_chunk in self.chunk_to_idx:
                        indices.append(self.chunk_to_idx[sub_chunk])
                        pos += cl
                        found = True
                        break
                
                if not found:
                    indices.append(self.unk_idx)
                    pos += 1
        
        return indices
    
    def decode(self, indices):
        return ''.join([self.idx_to_chunk.get(int(i), self.unk_token) for i in indices])
    
    def save(self, path):
        vocab_data = {
            'chunk_to_idx': self.chunk_to_idx,
            'idx_to_chunk': {int(k): v for k, v in self.idx_to_chunk.items()},
            'vocab_size': self.vocab_size,
            'unk_token': self.unk_token,
            'unk_idx': self.unk_idx
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f)
        print(f"Vocabulary saved to {path}")
    
    def load(self, path):
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.chunk_to_idx = vocab_data['chunk_to_idx']
        self.idx_to_chunk = {int(k): v for k, v in vocab_data['idx_to_chunk'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.unk_token = vocab_data.get('unk_token', '<UNK>')
        self.unk_idx = vocab_data.get('unk_idx', 0)
        print(f"Vocabulary loaded from {path} ({self.vocab_size} chunks)")

# =========================== #
# Dataset: Combined HF Loader # 
# =========================== #
class CombinedDataset:
    """
    Handles loading multiple HuggingFace datasets and streaming them into 
    the tokenizer.
    """
    
    # Map dataset name to the column containing the text data
    DATASET_COLUMN_MAP = {
        'agentlans/high-quality-english-sentences': 'text', 
        'roneneldan/TinyStories': 'text',                   
        'starhopp3r/TinyChat': 'text',                         
    }
    
    def __init__(self, dataset_names, seq_len=256, vocab_path=None, max_samples=None):
        self.seq_len = seq_len
        self.dataset_names = dataset_names
        self.all_texts = []
        
        print(f"Loading datasets: {dataset_names}")
        total_samples = 0
        
        for name in self.dataset_names:
            if name not in self.DATASET_COLUMN_MAP:
                print(f"⚠️  Skipping unknown dataset: {name}")
                continue
                
            column_name = self.DATASET_COLUMN_MAP[name]
            print(f"Loading '{name}' (column: '{column_name}')...")
            
            try:
                # Disable streaming here to ensure we get data in memory for vocab building
                # (For truly massive datasets, this logic needs to be fully streaming)
                dataset = load_dataset(name, split='train', streaming=False)
                
                if column_name not in dataset.features:
                    print(f"❌ Column '{column_name}' not found in dataset '{name}'")
                    continue
                
                texts = [item[column_name] for item in dataset]
                texts = [t for t in texts if t and len(t.strip()) > 0]
                
                self.all_texts.extend(texts)
                total_samples += len(texts)
                
                print(f"✓ Loaded {len(texts)} valid samples from {name}.")
                del dataset
                
            except Exception as e:
                print(f"❌ Error loading dataset '{name}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not self.all_texts:
            raise RuntimeError(f"No texts loaded! Check dataset names/mappings.")
        
        print(f"✓ Total merged samples: {total_samples}")
        
        # Setup Tokenizer
        self.tokenizer = ChunkTokenizer(vocab_only=True)
        
        if vocab_path and os.path.exists(vocab_path):
            print(f"Loading pre-built vocabulary from {vocab_path}")
            self.tokenizer.load(vocab_path)
        else:
            self.tokenizer.build_vocab_from_texts(self.all_texts, max_samples=max_samples)
            if vocab_path:
                self.tokenizer.save(vocab_path)
        
        self.vocab_size = self.tokenizer.vocab_size
        
        # Pre-tokenize
        print("Pre-tokenizing dataset (streaming)...")
        self.tokenized_data = []
        
        for idx, text in enumerate(self.all_texts):
            if idx % 1000 == 0 and idx > 0:
                print(f"Tokenized {idx}/{len(self.all_texts)} samples")
            
            tokens = self.tokenizer.encode(text)
            self.tokenized_data.extend(tokens)
        
        del self.all_texts # Cleanup raw text
        
        if not self.tokenized_data:
            raise RuntimeError("Tokenization produced empty dataset.")
        
        self.data = torch.tensor(self.tokenized_data, dtype=torch.long)
        del self.tokenized_data
        
        print(f"✓ Total tokens: {len(self.data):,}")
    
    def get_batch(self, batch_size=2):
        """Random sampling of sequences"""
        data_len = len(self.data)
        if data_len < self.seq_len + 10:
            raise RuntimeError(f"Dataset too short for seq_len {self.seq_len}")
        
        max_start_index = data_len - self.seq_len - 1
        
        ix = torch.randint(max_start_index, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x, y
    
    def decode(self, indices):
        return self.tokenizer.decode(indices)
    
    def encode(self, text):
        return self.tokenizer.encode(text)

# ================ #
# Export Utilities #
# ================ #
def generate_model_files(model, dataset, output_dir="i3-model-artifacts"):
    """Generates HuggingFace-compatible model files (.bin and config.json)"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating model files in: {output_dir}/")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    config = {
        "architectures": ["i3Model"],
        "model_type": "i3",
        "vocab_size": dataset.vocab_size,
        "d_model": model.d_model,
        "n_layers": len(model.layers),
        "n_heads": 16,
        "max_seq_len": model.max_seq_len,
        "conv_layers": 10,
        "attn_layers": 6,
        "d_state": 64,
        "tokenizer_type": "chunk",
        "dataset_sources": dataset.dataset_names,
        "torch_dtype": "float32",
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Model files generated successfully!")

# ============================================================================
# Training Loop
# ============================================================================
def train_i3_on_combined_datasets():
    DATASET_NAMES = [
        'agentlans/high-quality-english-sentences', 
        'roneneldan/TinyStories',
        'starhopp3r/TinyChat'
    ]
    
    # Hyperparameters
    seq_len = 256
    batch_size = 4
    d_model = 512
    n_layers = 16
    n_heads = 16
    d_state = 32
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    log_interval = 10
    gradient_clip = 1.0
    gradient_accumulation_steps = 1
    
    config = {
        "dataset_names": DATASET_NAMES,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "conv_layers": 10,
        "attn_layers": 6,
        "n_heads": n_heads,
        "d_state": d_state,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "optimizer": "AdamW",
        "gradient_clip": gradient_clip
    }
    
    print("=" * 80)
    print(f"TRAINING i3 MODEL on {len(DATASET_NAMES)} DATASETS")
    print("=" * 80)
    
    init_wandb(config)
    
    # MODIFIED: Changed file name to generic HF style
    vocab_path = "tokenizer.json"
    
    # 1. Load Data
    print("\nLoading dataset...")
    dataset = CombinedDataset(
        dataset_names=DATASET_NAMES, 
        seq_len=seq_len, 
        vocab_path=vocab_path
    )
    
    # 2. Init Model
    print("\nInitializing model...")
    model = i3Model(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        d_state=d_state
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    wandb.watch(model, log="all", log_freq=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 3. LR Scheduler
    warmup_iters = 100
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * (it + 1) / warmup_iters
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return learning_rate * (0.1 + 0.9 * coeff)
    
    print("\nStarting training...")
    print("=" * 80)
    
    best_loss = float('inf')
    losses = []
    
    for iter_num in range(max_iters):
        t0 = time.time()
        
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(gradient_accumulation_steps):
            x, y = dataset.get_batch(batch_size)
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            accum_loss += loss.item()
            loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        t1 = time.time()
        dt = t1 - t0
        losses.append(accum_loss)
        
        if iter_num % log_interval == 0:
            wandb.log({
                "train/loss": accum_loss,
                "train/perplexity": np.exp(accum_loss),
                "train/grad_norm": grad_norm.item(),
                "train/learning_rate": lr,
                "train/iteration": iter_num,
            })
            print(f"iter {iter_num:5d} | loss {accum_loss:.4f} | time {dt*1000:.2f}ms")
        
        # --- Evaluation & Checkpointing ---
        if iter_num % eval_interval == 0 and iter_num > 0:
            print("\n" + "-" * 80)
            print(f"Evaluation at iteration {iter_num}")
            model.eval()
            
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"✓ New best loss: {best_loss:.4f}")
                
                # MODIFIED: Changed checkpoint name to HF style (.bin)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config
                }, "pytorch_model_best.bin")
            
            # Simple Generation Test
            prompts = ["hello", "user: tell me", "once upon a time"]
            for prompt in prompts:
                try:
                    ctx = torch.tensor([dataset.encode(prompt)], dtype=torch.long).to(device)
                    gen = model.generate(ctx, max_new_tokens=50)[0].cpu()
                    print(f"Prompt: {prompt} -> {dataset.decode(gen)[:100]}...")
                except:
                    pass
            
            model.train()
            print("-" * 80 + "\n")

    generate_model_files(model, dataset)
    wandb.finish()
    return model, dataset

# ================ #
# Main Entry Point #
# ================ #
if __name__ == "__main__":
    if WANDB_API_KEY == "YOUR_WANDB_API_KEY_HERE":
         print("⚠️  WARNING: Set WANDB_API_KEY env var or update the script.")
    
    train_i3_on_combined_datasets()
