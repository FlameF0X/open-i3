import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import time
import os
import json
import wandb
from collections import deque
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# ========================================== #
# Global Configuration & Environment Secrets # 
# ========================================== #
# STRATEGY: Use environment variables for keys to keep the code open-source safe.
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "i3-rwkv-pro-v1")

class TrainingConfig:
    """Centralized hyperparameters for easy tuning."""
    d_model = 768
    n_rwkv_layers = 12
    n_attn_layers = 4
    n_heads = 12
    seq_len = 512
    batch_size = 4
    accum_steps = 8
    lr = 4e-4
    max_iters = 5000
    vocab_size = 32000
    checkpoint_dir = "checkpoints"

# ================================= #
# Metrics: Performance & Perplexity #
# ================================= #
class AdvancedPerplexityTracker:
    """
    Tracks model convergence using rolling windows and exponential smoothing.
     
    STRATEGY:
    We use a deque for windowed averages to prevent outliers from skewing 
    the "current" performance, while maintaining a smoothed line for 
    cleaner WandB visualizations.
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.best_ppl = float('inf')
        self.best_loss = float('inf')
        self.history = {'loss': [], 'ppl': [], 'smoothed_ppl': []}
     
    def update(self, loss):
        self.losses.append(loss)
        self.history['loss'].append(loss)
        self.history['ppl'].append(np.exp(loss))
     
    def get_metrics(self) -> Dict[str, float]:
        if not self.losses: return {}
        current_loss = self.losses[-1]
         
        # Exponential smoothing for stable tracking
        if len(self.history['smoothed_ppl']) == 0:
            smoothed = current_loss
        else:
            alpha = 0.1
            prev_smooth = np.log(self.history['smoothed_ppl'][-1])
            smoothed = alpha * current_loss + (1 - alpha) * prev_smooth
             
        smoothed_ppl = np.exp(smoothed)
        self.history['smoothed_ppl'].append(smoothed_ppl)
         
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_ppl = np.exp(current_loss)
             
        return {
            'ppl_current': np.exp(current_loss),
            'ppl_smoothed': smoothed_ppl,
            'ppl_windowed': np.exp(np.mean(self.losses)),
            'ppl_best': self.best_ppl
        }

# ========================================== #
# Core Architecture: RWKV Linear Recurrence #
# ========================================== #
@torch.jit.script
def rwkv_linear_attention(B: int, T: int, C: int, 
                          r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          w: torch.Tensor, u: torch.Tensor,
                          state_init: torch.Tensor):
    """
    JIT-Compiled WKV Kernel for RWKV v4.
     
    UNDER THE HOOD:
    This function simulates the attention mechanism as a recurrent state update.
    - 'w' (decay) controls how fast we forget the past.
    - 'u' (bonus) ensures the current token is prioritized.
    """
    y = torch.zeros_like(v)
    state_aa = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_bb = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_pp = state_init.clone()

    for t in range(T):
        rt, kt, vt = r[:, t], k[:, t], v[:, t]
         
        # 1. Output Calculation
        ww = u + state_pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
         
        wkv = (state_aa * e1 + vt * e2) / (state_bb * e1 + e2 + 1e-8)
        y[:, t] = wkv
         
        # 2. State Update
        ww = w + state_pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
         
        state_aa = state_aa * e1 + vt * e2
        state_bb = state_bb * e1 + e2
        state_pp = p
         
    return y

class RWKVTimeMix(nn.Module):
    """
    RWKV Time-Mixing Block.
     
    UNDER THE HOOD:
    1. Time Shift: Mixes current input with the previous time step.
    2. Linear Projection: Maps mixed inputs to Receptance, Key, and Value.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
         
        self.time_decay = nn.Parameter(torch.ones(d_model))
        self.time_first = nn.Parameter(torch.ones(d_model))
         
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
         
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
         
        with torch.no_grad():
            self.time_decay.data.uniform_(-6, -3)

    def forward(self, x):
        B, T, C = x.size()
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device), x[:, :-1]], dim=1)
         
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
         
        k, v = self.key(xk), self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
         
        w = -torch.exp(self.time_decay)
        u = self.time_first
        state_init = torch.full((B, C), -1e30, dtype=torch.float32, device=x.device)
         
        rwkv = rwkv_linear_attention(B, T, C, r, k, v, w, u, state_init)
        return self.output(r * rwkv)

class RWKVChannelMix(nn.Module):
    """RWKV-style Feed-Forward Network (FFN)."""
    def __init__(self, d_model, ffn_mult=4):
        super().__init__()
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
         
        hidden_sz = int(d_model * ffn_mult)
        self.key = nn.Linear(d_model, hidden_sz, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(hidden_sz, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device), x[:, :-1]], dim=1)
         
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
         
        k = torch.square(torch.relu(self.key(xk)))
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
         
        return r * kv

# ====================================== #
# Core Architecture: Attention Mechanism #
# ====================================== #
class FullAttention(nn.Module):
    """Standard Multi-Head Attention (MHA) for global context recall."""
    def __init__(self, d_model, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
         
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
         
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
         
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
             
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# ======================== #
# Main Model: i3HybridModel #
# ======================== #
class i3HybridModel(nn.Module):
    """
    Hybrid RWKV-Transformer Architecture.
     
    STRATEGY:
    - Base Layers: RWKV for efficient long-range state building.
    - Top Layers: Attention for precise information retrieval.
    """
    def __init__(self, vocab_size, d_model=768, n_heads=12, 
                 n_rwkv_layers=12, n_attn_layers=4, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
         
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
         
        self.layers = nn.ModuleList()
        for _ in range(n_rwkv_layers):
            self.layers.append(RWKVBlock(d_model))
        for _ in range(n_attn_layers):
            self.layers.append(StandardAttentionBlock(d_model, n_heads=n_heads))
             
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
        if T > self.max_seq_len:
            idx = idx[:, -self.max_seq_len:]
            T = self.max_seq_len
             
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos_embed(pos)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
         
        for layer in self.layers:
            if isinstance(layer, StandardAttentionBlock):
                x = layer(x, mask)
            else:
                x = layer(x)
             
        x = self.ln_f(x)
        logits = self.head(x)
         
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
             
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

class RWKVBlock(nn.Module):
    def __init__(self, d_model, ffn_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.att, self.ffn = RWKVTimeMix(d_model), RWKVChannelMix(d_model, ffn_mult)
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class StandardAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=12, ffn_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = FullAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Linear(d_model * ffn_mult, d_model))
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

# ============================================================================
# Training Utilities & Entry Point
# ============================================================================
class BPETokenizerManager:
    def __init__(self, vocab_size=32000, model_path="tokenizer.json"):
        self.vocab_size, self.model_path, self.tokenizer = vocab_size, model_path, None

    def train_or_load(self, dataset_names):
        if os.path.exists(self.model_path):
            self.tokenizer = Tokenizer.from_file(self.model_path)
        else:
            print("Training Tokenizer...")
            t = Tokenizer(models.BPE())
            t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            t.decoder = decoders.ByteLevel()
            tr = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"])
            def it():
                for n in dataset_names:
                    ds = load_dataset(n, split='train', streaming=True)
                    for i, item in enumerate(ds):
                        if i > 5000: break
                        yield item.get('text', '')
            t.train_from_iterator(it(), trainer=tr)
            t.save(self.model_path)
            self.tokenizer = t
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids): return self.tokenizer.decode(ids)

class UnifiedDataset:
    def __init__(self, dataset_names, tok_manager, seq_len=512):
        self.tok, self.seq_len, self.buf = tok_manager, seq_len, []
        self.ds = [iter(load_dataset(n, split='train', streaming=True)) for n in dataset_names]
        self.idx = 0

    def fill(self):
        while len(self.buf) < 100000:
            try:
                t = next(self.ds[self.idx]).get('text', '')
                if t.strip(): self.buf.extend(self.tok.encode(t) + [3])
            except StopIteration: self.idx = (self.idx + 1) % len(self.ds)

    def get_batch(self, b_size):
        if len(self.buf) < (b_size * (self.seq_len + 1)): self.fill()
        x, y = [], []
        for _ in range(b_size):
            i = np.random.randint(0, len(self.buf) - self.seq_len - 1)
            c = self.buf[i : i + self.seq_len + 1]
            x.append(torch.tensor(c[:-1], dtype=torch.long))
            y.append(torch.tensor(c[1:], dtype=torch.long))
        return torch.stack(x), torch.stack(y)

def train():
    cfg = TrainingConfig()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
     
    # Initialize WandB safely
    if WANDB_API_KEY:
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        wandb.init(project=WANDB_PROJECT, config=vars(cfg))
     
    tok = BPETokenizerManager(vocab_size=cfg.vocab_size)
    tok.train_or_load(['roneneldan/TinyStories'])
    dataset = UnifiedDataset(['roneneldan/TinyStories'], tok, seq_len=cfg.seq_len)
    tracker = AdvancedPerplexityTracker()

    model = i3HybridModel(tok.vocab_size, cfg.d_model, cfg.n_heads, cfg.n_rwkv_layers, cfg.n_attn_layers, cfg.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # FIXED: Updated GradScaler to new syntax
    scaler = torch.amp.GradScaler(device_type)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    print(f"✓ Model Initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f}M Params")

    for i in range(cfg.max_iters):
        opt.zero_grad()
        loss_val = 0
        for _ in range(cfg.accum_steps):
            bx, by = dataset.get_batch(cfg.batch_size)
            bx, by = bx.to(device), by.to(device)
            
            # FIXED: Updated autocast to new syntax with explicit device
            with torch.amp.autocast(device_type):
                _, loss = model(bx, by)
                loss = loss / cfg.accum_steps
            
            scaler.scale(loss).backward()
            loss_val += loss.item()
         
        scaler.step(opt)
        scaler.update()
        tracker.update(loss_val)

        if i % 10 == 0:
            m = tracker.get_metrics()
            print(f"Iter {i:4d} | Loss {loss_val:.4f} | PPL {m['ppl_current']:.2f}")
            if wandb.run: wandb.log({"loss": loss_val, "ppl": m['ppl_current']})
             
        if i % 1000 == 0 and i > 0:
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f"model_iter_{i}.pt"))

if __name__ == "__main__":
    train()
