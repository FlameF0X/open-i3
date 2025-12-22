import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import time
import os
import json
import wandb
import random
from collections import deque
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# ============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================================
# STRATEGY: Centralized configuration allows for easy experiments with model scaling.
# We aim for a ~116M parameter model suitable for educational pre-training on consumer GPUs.
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "") 
WANDB_PROJECT = "i3-bert-pretrain"
CHECKPOINT_DIR = "i3_bert_checkpoints"

class ModelConfig:
    d_model = 768
    n_rwkv_layers = 4    # Bottom layers: Efficient local context
    n_attn_layers = 4    # Top layers: Global lookup & copying
    n_heads = 12
    seq_len = 128
    batch_size = 32
    lr = 2e-4
    max_iters = 5000
    vocab_size = 30000

# ============================================================================
# 1. RWKV CORE (JIT OPTIMIZED)
# ============================================================================
@torch.jit.script
def rwkv_linear_attention(B: int, T: int, C: int, 
                          r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          w: torch.Tensor, u: torch.Tensor,
                          state_init: torch.Tensor):
    """
    JIT-Compiled WKV Kernel for RWKV v4.
    
    UNDER THE HOOD:
    This performs the 'linear attention' mechanism. Unlike standard attention (N^2),
    this operation is O(T) linear time. It maintains a running state (state_aa, state_bb)
    that decays over time based on 'w'.
    """
    y = torch.zeros_like(v)
    state_aa = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_bb = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_pp = state_init.clone()

    for t in range(T):
        rt, kt, vt = r[:, t], k[:, t], v[:, t]
        
        # 1. Calculate Output (current state + current input)
        ww = u + state_pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        wkv = (state_aa * e1 + vt * e2) / (state_bb * e1 + e2 + 1e-6)
        y[:, t] = wkv
        
        # 2. Update State (decay state + current input)
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
    
    STRATEGY:
    Captures temporal dependencies. 'Time Mixing' implies we are mixing information
    from the current token with the previous token (time-shifting) before processing.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Learnable decay rates and bonus terms
        self.time_decay = nn.Parameter(torch.ones(d_model))
        self.time_first = nn.Parameter(torch.ones(d_model))
        
        # Time-shift mixing factors
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Projections
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        # Init decay to be negative (forgetting mechanism)
        self.time_decay.data.uniform_(-6, -3)

    def forward(self, x):
        B, T, C = x.size()
        
        # 1. Time Shift: Mix current x with x[t-1]
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device), x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        # 2. Projections
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        
        # 3. WKV Kernel (The "Attention")
        w = -torch.exp(self.time_decay)
        u = self.time_first
        state_init = torch.full((B, C), -1e30, dtype=torch.float32, device=x.device)
        
        rwkv = rwkv_linear_attention(B, T, C, r, k, v, w, u, state_init)
        
        return self.output(r * rwkv)

class RWKVChannelMix(nn.Module):
    """
    RWKV Channel-Mixing Block (Feed-Forward Network).
    
    UNDER THE HOOD:
    Equivalent to the FFN in Transformers, but with an added 'Time Shift' 
    on the inputs to give the network slightly more temporal context.
    """
    def __init__(self, d_model, ffn_mult=4):
        super().__init__()
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        hidden_sz = d_model * ffn_mult
        
        self.key = nn.Linear(d_model, hidden_sz, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(hidden_sz, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        # 1. Time Shift
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device), x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        # 2. Gating and Activation
        k = torch.square(torch.relu(self.key(xk))) # Squared ReLU
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        
        return r * kv

# ============================================================================
# 2. HYBRID BLOCKS (Bi-RWKV + ATTENTION)
# ============================================================================
class BiRWKVBlock(nn.Module):
    """
    Bidirectional RWKV Block.
    
    STRATEGY:
    RWKV is inherently causal (left-to-right). To use it in a BERT-style 
    (Masked Language Model) architecture, we need to see the future tokens.
    
    UNDER THE HOOD:
    We run two parallel RWKV instances:
    1. Forward: Processes sequence normally [0 -> T].
    2. Backward: Flips sequence, processes [T -> 0], then flips output back.
    The results are summed to create a bidirectional representation.
    """
    def __init__(self, d_model, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.fwd_time_mix = RWKVTimeMix(d_model)
        self.bwd_time_mix = RWKVTimeMix(d_model) # Separate weights for backward physics
        
        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mix = RWKVChannelMix(d_model, ffn_mult)

    def forward(self, x, mask=None):
        # 1. Bidirectional Time Mixing
        x_norm = self.ln1(x)
        
        # A. Forward Pass
        x_fwd = self.fwd_time_mix(x_norm)
        
        # B. Backward Pass (Flip -> Process -> Flip back)
        x_rev = torch.flip(x_norm, [1])
        x_bwd_rev = self.bwd_time_mix(x_rev)
        x_bwd = torch.flip(x_bwd_rev, [1])
        
        # C. Fusion: Sum contexts into residual stream
        x = x + x_fwd + x_bwd
        
        # 2. Channel Mixing (Standard FFN equivalent)
        x = x + self.channel_mix(self.ln2(x))
        return x

class FullAttention(nn.Module):
    """Standard Multi-Head Attention (O(N^2)) for global recall."""
    def __init__(self, d_model, n_heads=16):
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
        
        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class StandardAttentionBlock(nn.Module):
    """Wrapper for Attention + FFN + Norms."""
    def __init__(self, d_model, n_heads=16, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FullAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

# ============================================================================
# 3. i3-BERT MODEL
# ============================================================================
class i3BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model) # Segment A vs B
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        embeddings = (self.word_embeddings(input_ids) + 
                      self.position_embeddings(pos_ids) + 
                      self.token_type_embeddings(token_type_ids))
        
        return self.dropout(self.ln(embeddings))

class i3BertModel(nn.Module):
    """
    i3-BERT: Hybrid RWKV-Transformer for BERT tasks.
    
    STRATEGY:
    - Bottom Layers (RWKV): Efficiently build local phrases and syntactic structures.
    - Top Layers (Attention): Perform global reasoning and 'copy-paste' operations.
    """
    def __init__(self, vocab_size, d_model=768, n_rwkv_layers=6, n_attn_layers=6, n_heads=12, max_len=512):
        super().__init__()
        self.embeddings = i3BertEmbeddings(vocab_size, d_model, max_len)
        
        self.layers = nn.ModuleList()
        
        # 1. Bi-RWKV Layers (Bottom)
        # These process context efficiently in both directions
        print(f"Building Model: {n_rwkv_layers} Bi-RWKV Layers + {n_attn_layers} Attention Layers")
        for _ in range(n_rwkv_layers):
            self.layers.append(BiRWKVBlock(d_model, ffn_mult=4))
            
        # 2. Standard Attention Layers (Top)
        # These allow for global mixing between distant tokens
        for _ in range(n_attn_layers):
            self.layers.append(StandardAttentionBlock(d_model, n_heads=n_heads))
        
        # --- Pre-training Heads ---
        # MLM Head: Predicts masked tokens
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # NSP Head: Predicts if Sentence B follows Sentence A
        self.pooler_dense = nn.Linear(d_model, d_model)
        self.nsp_head = nn.Linear(d_model, 2)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
        # Special initialization for RWKV parameters usually helps stability
        if isinstance(module, RWKVTimeMix):
            nn.init.orthogonal_(module.receptance.weight, gain=1)
            nn.init.orthogonal_(module.key.weight, gain=1)
            nn.init.orthogonal_(module.value.weight, gain=1)
            nn.init.orthogonal_(module.output.weight, gain=0)

    def forward(self, input_ids, segment_ids, labels=None, nsp_labels=None):
        # Create mask for Attention layers (1 for real, 0 for PAD)
        # RWKV ignores this mask naturally as it just processes 0 vectors
        mask = (input_ids != 1).unsqueeze(1).unsqueeze(2)
        
        x = self.embeddings(input_ids, segment_ids)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        # --- Heads ---
        prediction_scores = self.mlm_head(x)
        
        # Pooler for NSP: Take CLS token (index 0), run through tanh
        cls_token_state = x[:, 0, :]
        pooled_output = torch.tanh(self.pooler_dense(cls_token_state))
        seq_relationship_score = self.nsp_head(pooled_output)
        
        if labels is not None and nsp_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), nsp_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            
            return {
                "loss": total_loss,
                "mlm_loss": masked_lm_loss,
                "nsp_loss": next_sentence_loss,
                "logits": prediction_scores
            }
            
        return prediction_scores, seq_relationship_score

# ============================================================================
# 4. BERT DATASET MANAGER
# ============================================================================
class BertDatasetManager:
    """
    Manages data streaming and dynamic BERT masking.
    
    STRATEGY:
    1. Stream text from HuggingFace to avoid RAM issues.
    2. Buffer sentences.
    3. Generate 'Next Sentence Prediction' (NSP) pairs (50% IsNext, 50% NotNext).
    4. Apply 'Masked Language Model' (MLM) masking (15% of tokens).
    """
    def __init__(self, dataset_names, tokenizer_manager, seq_len=128):
        self.tokenizer = tokenizer_manager
        self.seq_len = seq_len
        self.CLS_TOKEN_ID = 2
        self.SEP_TOKEN_ID = 3
        self.MASK_TOKEN_ID = 4
        self.PAD_TOKEN_ID = 1
        self.VOCAB_SIZE = tokenizer_manager.vocab_size

        self.datasets = []
        for name in dataset_names:
            try:
                ds = load_dataset(name, split='train', streaming=True)
                self.datasets.append(iter(ds))
                print(f"✓ Streaming: {name}")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        
        self.current_ds_idx = 0
        self.buffer = []

    def _get_next_text_chunk(self):
        try:
            dataset_iter = self.datasets[self.current_ds_idx]
            item = next(dataset_iter)
            text = item.get('text', '')
            return text
        except StopIteration:
            self.current_ds_idx = (self.current_ds_idx + 1) % len(self.datasets)
            return self._get_next_text_chunk()
        except Exception:
             self.current_ds_idx = (self.current_ds_idx + 1) % len(self.datasets)
             return ""

    def fill_buffer(self, min_sentences=1000):
        while len(self.buffer) < min_sentences:
            text = self._get_next_text_chunk()
            if not text: continue
            sentences = text.split('.')
            for s in sentences:
                s = s.strip()
                if len(s) > 10:
                    ids = self.tokenizer.encode(s)
                    if len(ids) < self.seq_len - 3:
                        self.buffer.append(ids)

    def create_batch(self, batch_size):
        if len(self.buffer) < batch_size * 2:
            self.fill_buffer()

        batch_input_ids, batch_labels = [], []    
        batch_segment_ids, batch_nsp_labels = [], [] 

        for _ in range(batch_size):
            if len(self.buffer) < 2: self.fill_buffer()
            
            # A. Next Sentence Prediction (NSP) Logic
            is_next = random.random() > 0.5
            idx_a = random.randint(0, len(self.buffer) - 2)
            sent_a = self.buffer[idx_a]
            
            if is_next:
                sent_b = self.buffer[idx_a + 1]
                nsp_label = 1
            else:
                idx_b = random.randint(0, len(self.buffer) - 1)
                sent_b = self.buffer[idx_b]
                nsp_label = 0

            # Truncate to fit sequence length
            max_len_tokens = self.seq_len - 3 # -3 for [CLS], [SEP], [SEP]
            while len(sent_a) + len(sent_b) > max_len_tokens:
                if len(sent_a) > len(sent_b): sent_a.pop()
                else: sent_b.pop()

            input_ids = [self.CLS_TOKEN_ID] + sent_a + [self.SEP_TOKEN_ID] + sent_b + [self.SEP_TOKEN_ID]
            segment_ids = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)
            
            # B. Masked Language Model (MLM) Logic
            labels = [-100] * len(input_ids) # -100 ignored by CrossEntropyLoss
            for i, token in enumerate(input_ids):
                if token in [self.CLS_TOKEN_ID, self.SEP_TOKEN_ID, self.PAD_TOKEN_ID]: continue
                
                # 15% probability to mask a token
                if random.random() < 0.15:
                    labels[i] = token
                    prob = random.random()
                    if prob < 0.8: input_ids[i] = self.MASK_TOKEN_ID # 80% [MASK]
                    elif prob < 0.9: input_ids[i] = random.randint(5, self.VOCAB_SIZE - 1) # 10% Random
                    # 10% Unchanged
            
            # Padding
            pad_len = self.seq_len - len(input_ids)
            input_ids += [self.PAD_TOKEN_ID] * pad_len
            segment_ids += [0] * pad_len
            labels += [-100] * pad_len
            
            batch_nsp_labels.append(torch.tensor(nsp_label))
            batch_input_ids.append(torch.tensor(input_ids))
            batch_segment_ids.append(torch.tensor(segment_ids))
            batch_labels.append(torch.tensor(labels))

        return (torch.stack(batch_input_ids), torch.stack(batch_segment_ids), 
                torch.stack(batch_labels), torch.stack(batch_nsp_labels))

# ============================================================================
# 5. UTILITIES (TOKENIZER)
# ============================================================================
class BPETokenizerManager:
    def __init__(self, vocab_size=30000, model_path="tokenizer_bert.json"):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.tokenizer = None

    def train_or_load(self, dataset_names):
        if os.path.exists(self.model_path):
            print(f"Loading existing tokenizer from {self.model_path}...")
            self.tokenizer = Tokenizer.from_file(self.model_path)
        else:
            print("Training new Tokenizer...")
            self._train(dataset_names)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"Tokenizer Ready. Vocab: {self.vocab_size}")

    def _train(self, dataset_names):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        special_tokens = ["<UNK>", "<PAD>", "<CLS>", "<SEP>", "<MASK>"]
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens, min_frequency=2, show_progress=True)
        
        def batch_iterator():
            for name in dataset_names:
                ds = load_dataset(name, split='train', streaming=True)
                for i, item in enumerate(ds):
                    if i > 10000: break 
                    yield item.get('text', '')

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save(self.model_path)
        self.tokenizer = tokenizer

    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids): return self.tokenizer.decode(ids)

# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
def train():
    cfg = ModelConfig()
    DATASET_NAMES = ['HuggingFaceFW/fineweb-edu']

    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    wandb.init(project=WANDB_PROJECT, config=vars(cfg), name="i3-BERT-116M")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Tokenizer & Dataset
    tokenizer_manager = BPETokenizerManager()
    tokenizer_manager.train_or_load(DATASET_NAMES)
    bert_dataset = BertDatasetManager(DATASET_NAMES, tokenizer_manager, seq_len=cfg.seq_len)

    # 2. Hybrid Model
    model = i3BertModel(
        vocab_size=tokenizer_manager.vocab_size,
        d_model=cfg.d_model,
        n_rwkv_layers=cfg.n_rwkv_layers,
        n_attn_layers=cfg.n_attn_layers,
        n_heads=cfg.n_heads,
        max_len=cfg.seq_len
    ).to(device)

    # Calculate Params
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()

    print("Starting i3-BERT Pre-training...")
    
    for iteration in range(cfg.max_iters):
        t0 = time.time()
        
        input_ids, segment_ids, labels, nsp_labels = bert_dataset.create_batch(cfg.batch_size)
        input_ids, segment_ids = input_ids.to(device), segment_ids.to(device)
        labels, nsp_labels = labels.to(device), nsp_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, segment_ids, labels, nsp_labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"Iter {iteration} | Loss: {loss.item():.4f} | MLM: {outputs['mlm_loss']:.4f} | NSP: {outputs['nsp_loss']:.4f}")
            wandb.log({"total_loss": loss.item(), "mlm_loss": outputs['mlm_loss'].item(), "nsp_loss": outputs['nsp_loss'].item()})

        if iteration % 500 == 0:
            print("\n--- Validation Inference ---")
            logits = outputs['logits'][0]
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().tolist()
            label_id = labels[0].cpu().tolist()
            
            for i, lab in enumerate(label_id):
                if lab != -100:
                    original = tokenizer_manager.decode([lab])
                    predicted = tokenizer_manager.decode([preds[i]])
                    print(f"Masked index {i}: Target='{original}' | Prediction='{predicted}'")
            print("----------------------------\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "i3_bert_model_116m.pt"))
    print("Model Saved.")

if __name__ == "__main__":
    train()
