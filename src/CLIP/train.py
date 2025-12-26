import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import json
import os
import time
from typing import Tuple, Dict, Any
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer

# ============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================================
# STRATEGY: Scaled-up hybrid architecture combining ResNet vision features with
# a dual RWKV-Attention text encoder for CLIP-style contrastive pre-training.
WANDB_PROJECT = "i3-rwkv-clip-hybrid-large"
CHECKPOINT_DIR = "checkpoints"

class Config:
    d_model = 768
    n_rwkv_layers = 12   # Deep recurrent context for text
    n_attn_layers = 4    # Top-level global reasoning for text
    n_heads = 12
    ffn_mult = 4
    max_len = 77         # Standard CLIP context length
    batch_size = 32
    learning_rate = 5e-5
    max_iters = 2000
    image_size = 224

# ============================================================================
# 1. CORE RWKV ENGINE (JIT OPTIMIZED)
# ============================================================================
@torch.jit.script
def rwkv_linear_attention(B: int, T: int, C: int, 
                          r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          w: torch.Tensor, u: torch.Tensor,
                          state_init: torch.Tensor):
    """
    JIT-Compiled WKV Kernel for RWKV v4.
    
    UNDER THE HOOD:
    Implements the linear attention mechanism O(T). It uses a time-decaying 
    state (w) and a 'first-token' bonus (u) to maintain context without
    the quadratic memory cost of standard transformers.
    """
    y = torch.zeros_like(v)
    state_aa = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_bb = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_pp = state_init.clone()

    for t in range(T):
        rt, kt, vt = r[:, t], k[:, t], v[:, t]
        
        # Output calculation with numerical stability
        ww = u + state_pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        wkv = (state_aa * e1 + vt * e2) / (state_bb * e1 + e2 + 1e-6)
        y[:, t] = wkv
        
        # State update (decay existing state + add current input)
        ww = w + state_pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        state_aa = state_aa * e1 + vt * e2
        state_bb = state_bb * e1 + e2
        state_pp = p
    return y

class RWKVTimeMix(nn.Module):
    """Captures temporal dependencies via learnable time-shifting and WKV."""
    def __init__(self, d_model):
        super().__init__()
        self.time_decay = nn.Parameter(torch.uniform_(torch.empty(d_model), -6, -3))
        self.time_first = nn.Parameter(torch.ones(d_model))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        # Time-Shift: Mix token t with t-1
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device), x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        k, v = self.key(xk), self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        
        w, u = -torch.exp(self.time_decay), self.time_first
        state_init = torch.full((B, C), -1e30, device=x.device)
        rwkv = rwkv_linear_attention(B, T, C, r, k, v, w, u, state_init)
        return self.output(r * rwkv)

class RWKVChannelMix(nn.Module):
    """FFN-style block with temporal gating."""
    def __init__(self, d_model, ffn_mult=4):
        super().__init__()
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        self.key = nn.Linear(d_model, d_model * ffn_mult, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model * ffn_mult, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device), x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        k = torch.square(torch.relu(self.key(xk)))
        r = torch.sigmoid(self.receptance(xr))
        return r * self.value(k)

class RWKVBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = RWKVTimeMix(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = RWKVChannelMix(d_model)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ============================================================================
# 2. VISION ENCODER (CNN BACKBONE)
# ============================================================================
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class VisionEncoderLarge(nn.Module):
    """
    Standard ResNet-style Vision Encoder.
    
    STRATEGY:
    Extracts high-level spatial features from images. We use a deep CNN
    to project visual information into the same embedding space as text.
    """
    def __init__(self, d_model=768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, d_model)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [ResBlock(in_c, out_c, stride)]
        for _ in range(1, blocks): layers.append(ResBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))

# ============================================================================
# 3. TEXT ENCODER (HYBRID RWKV-TRANSFORMER)
# ============================================================================
class HybridTextEncoderLarge(nn.Module):
    """
    Hybrid RWKV-Transformer Text Encoder.
    
    STRATEGY:
    - 12 RWKV Layers: Efficiently process long-range sequence context.
    - 4 Attention Layers: Provide dense, global cross-token reasoning.
    
    UNDER THE HOOD:
    The model consumes tokens, builds a hidden state via RWKV, and then 
    refines it using Attention before extracting the final pooled embedding.
    """
    def __init__(self, vocab_size, d_model=768, n_rwkv=12, n_attn=4, max_len=77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        self.rwkv_layers = nn.ModuleList([RWKVBlock(d_model) for _ in range(n_rwkv)])
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=Config.n_heads, 
                dim_feedforward=d_model*4, batch_first=True, activation="gelu"
            ) for _ in range(n_attn)
        ])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        for layer in self.rwkv_layers: x = layer(x)
        for layer in self.attn_layers: x = layer(x)
        # Pooled representation: Take the last token state (common for CLIP)
        return self.ln_final(x[:, -1, :])

# ============================================================================
# 4. i3-CLIP-HYBRID WRAPPER
# ============================================================================
class i3CLIPHybridLarge(nn.Module):
    """
    Dual-Encoder architecture for Contrastive Learning.
    
    STRATEGY:
    Project images and text into a shared hypersphere. Maximize the cosine 
    similarity of paired (Image, Text) while minimizing others in the batch.
    """
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        self.visual = VisionEncoderLarge(d_model=d_model)
        self.textual = HybridTextEncoderLarge(vocab_size=vocab_size, d_model=d_model)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        img_features = F.normalize(self.visual(images), dim=-1)
        txt_features = F.normalize(self.textual(texts), dim=-1)
        
        # Calculate Logits (Contrastive Matrix)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_features @ txt_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

# ============================================================================
# 5. DATASET MANAGER
# ============================================================================
class CLIPDataset:
    """Manages multi-modal data streaming and preprocessing."""
    def __init__(self, tokenizer, split='train', max_len=77):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        ])
        # Using a midjourney descriptive dataset for high quality text-image pairs
        self.ds = load_dataset("MohamedRashad/midjourney-detailed-prompts", split=split, streaming=True)
        self.iterator = iter(self.ds)

    def get_batch(self, batch_size):
        imgs, txts = [], []
        while len(imgs) < batch_size:
            try:
                item = next(self.iterator)
                if not isinstance(item['image'], Image.Image): continue
                
                # Image transformation
                imgs.append(self.transform(item['image'].convert("RGB")))
                
                # Tokenization
                tokens = self.tokenizer(
                    item['image_description'], 
                    padding='max_length', 
                    truncation=True, 
                    max_length=self.max_len, 
                    return_tensors="pt"
                ).input_ids[0]
                txts.append(tokens)
            except StopIteration:
                self.iterator = iter(self.ds)
        return torch.stack(imgs), torch.stack(txts)

# ============================================================================
# 6. TRAINING & UTILITIES
# ============================================================================
def save_checkpoint(model, optimizer, step, loss):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"i3_clip_step_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"--> Saved checkpoint: {path}")

def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = i3CLIPHybridLarge(tokenizer.vocab_size, d_model=Config.d_model).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"i3-CLIP-HYBRID INITIALIZED")
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    dataset = CLIPDataset(tokenizer)
    
    # Optional: Log in via CLI before running or set WANDB_API_KEY env var
    wandb.init(project=WANDB_PROJECT, config=vars(Config))
    
    model.train()
    best_loss = float('inf')
    
    for i in range(Config.max_iters):
        start_time = time.time()
        
        # 1. Fetch Data
        images, texts = dataset.get_batch(Config.batch_size)
        images, texts = images.to(device), texts.to(device)
        
        # 2. Forward Pass
        logits_img, logits_txt = model(images, texts)
        
        # 3. Contrastive Loss (Symmetric Cross Entropy)
        labels = torch.arange(images.size(0), device=device)
        loss_img = F.cross_entropy(logits_img, labels)
        loss_txt = F.cross_entropy(logits_txt, labels)
        loss = (loss_img + loss_txt) / 2
        
        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if i % 10 == 0:
            it_time = time.time() - start_time
            print(f"Step {i:4d} | Loss: {loss.item():.4f} | {it_time:.2f}s/it")
            wandb.log({"train_loss": loss.item(), "step": i, "secs_per_iter": it_time})

        if i % 500 == 0 and i > 0:
            if loss < best_loss:
                best_loss = loss.item()
                save_checkpoint(model, optimizer, i, loss.item())

    print("Training Complete. Saving final model...")
    save_checkpoint(model, optimizer, Config.max_iters, loss.item())

if __name__ == "__main__":
    train()
