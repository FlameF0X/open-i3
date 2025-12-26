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
# STRATEGY: Scaled-up hybrid architecture combining a ResNet-Bottleneck vision 
# backbone with a dual RWKV-Attention text encoder for CLIP-style pre-training.
WANDB_PROJECT = "i3-rwkv-clip-hybrid-large"
SAVE_DIR = "checkpoints"

class Config:
    d_model = 768
    n_rwkv = 12          # Recurrent layers for deep context
    n_attn = 4           # Transformer layers for global reasoning
    n_heads = 12
    ffn_mult = 4
    max_len = 77         # Standard CLIP context window
    batch_size = 32
    learning_rate = 5e-5
    max_iters = 2000
    image_size = 224

# ============================================================================
# 1. RWKV CORE ENGINE (JIT OPTIMIZED)
# ============================================================================
@torch.jit.script
def rwkv_linear_attention(B: int, T: int, C: int, 
                          r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          w: torch.Tensor, u: torch.Tensor,
                          state_init: torch.Tensor):
    """
    JIT-Compiled WKV Kernel for Linear-Time Attention.
    
    UNDER THE HOOD:
    Implements the RWKV v4 WKV logic. By iterating through time O(T) 
    and maintaining a cumulative state, it achieves Transformer-like 
    performance without the quadratic O(T^2) memory bottleneck.
    """
    y = torch.zeros_like(v)
    state_aa = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_bb = torch.zeros(B, C, dtype=torch.float32, device=r.device)
    state_pp = state_init.clone()

    for t in range(T):
        rt, kt, vt = r[:, t], k[:, t], v[:, t]
        
        # Output logic
        ww = u + state_pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        wkv = (state_aa * e1 + vt * e2) / (state_bb * e1 + e2 + 1e-6)
        y[:, t] = wkv
        
        # State update logic
        ww = w + state_pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        state_aa = state_aa * e1 + vt * e2
        state_bb = state_bb * e1 + e2
        state_pp = p
    return y

class RWKVTimeMix(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.time_decay = nn.Parameter(torch.ones(d_model).uniform_(-6, -3))
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
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device), x[:, :-1]], dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = torch.square(torch.relu(self.key(xk)))
        return torch.sigmoid(self.receptance(xr)) * self.value(k)

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
# 2. VISION ENCODER (RESNET-BOTTLENECK)
# ============================================================================
class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck block.
    
    UNDER THE HOOD:
    Uses 1x1 convolutions to compress and expand dimensions, allowing for
    deeper networks with fewer parameters compared to standard residual blocks.
    """
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        return F.relu(out)

class VisionEncoderLarge(nn.Module):
    """
    High-capacity CNN for visual feature extraction.
    
    STRATEGY:
    Maps images to a 768-dimensional latent space. The bottleneck structure
    ensures compatibility with existing 2048-dim pre-pooled features.
    """
    def __init__(self, d_model=768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, d_model)

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        layers = [Bottleneck(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes * 4, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))

# ============================================================================
# 3. HYBRID TEXT ENCODER
# ============================================================================
class HybridTextEncoderLarge(nn.Module):
    """
    The i3-Hybrid Text Encoder.
    
    UNDER THE HOOD:
    Combines RWKV's linear context processing with multi-head attention's
    global refinement. This hybrid approach captures both sequential flow
    and complex inter-token relationships.
    """
    def __init__(self, vocab_size, d_model=768, n_rwkv=12, n_attn=4, max_len=77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.rwkv_layers = nn.ModuleList([RWKVBlock(d_model) for _ in range(n_rwkv)])
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=Config.n_heads, 
                dim_feedforward=d_model*Config.ffn_mult, 
                batch_first=True, activation="gelu"
            ) for _ in range(n_attn)
        ])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        for layer in self.rwkv_layers: x = layer(x)
        for layer in self.attn_layers: x = layer(x)
        return self.ln_final(x[:, -1, :])

# ============================================================================
# 4. i3-CLIP WRAPPER & TRAINING
# ============================================================================
class i3CLIPHybridLarge(nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        self.visual = VisionEncoderLarge(d_model=d_model)
        self.textual = HybridTextEncoderLarge(vocab_size=vocab_size, d_model=d_model)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        img_f = F.normalize(self.visual(images), dim=-1)
        txt_f = F.normalize(self.textual(texts), dim=-1)
        scale = self.logit_scale.exp()
        logits = scale * img_f @ txt_f.t()
        return logits, logits.t()

class CLIPDataset:
    def __init__(self, tokenizer, split='train', max_len=77):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48, 0.45, 0.40), (0.26, 0.26, 0.27))
        ])
        self.ds = load_dataset("MohamedRashad/midjourney-detailed-prompts", split=split, streaming=True)
        self.iterator = iter(self.ds)

    def get_batch(self, batch_size):
        imgs, txts = [], []
        while len(imgs) < batch_size:
            try:
                item = next(self.iterator)
                if not isinstance(item['image'], Image.Image): continue
                imgs.append(self.transform(item['image'].convert("RGB")))
                tokens = self.tokenizer(
                    item['image_description'], padding='max_length', truncation=True, 
                    max_length=self.max_len, return_tensors="pt"
                ).input_ids[0]
                txts.append(tokens)
            except StopIteration:
                self.iterator = iter(self.ds)
        return torch.stack(imgs), torch.stack(txts)

def save_checkpoint(model, optimizer, step, loss):
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"i3_clip_hybrid_step_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': vars(Config)
    }, save_path)
    print(f"--> Saved checkpoint: {save_path}")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = i3CLIPHybridLarge(tokenizer.vocab_size).to(device)
    
    # Checkpoint Recovery
    if os.path.exists("pytorch_model.bin"):
        print("Loading weights from pytorch_model.bin...")
        model.load_state_dict(torch.load("pytorch_model.bin", map_location=device), strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    dataset = CLIPDataset(tokenizer)
    
    wandb.init(project=WANDB_PROJECT, config=vars(Config))
    
    model.train()
    print(f"Starting training on {device}...")
    for i in range(Config.max_iters):
        start = time.time()
        images, texts = dataset.get_batch(Config.batch_size)
        images, texts = images.to(device), texts.to(device)
        
        logits_img, logits_txt = model(images, texts)
        labels = torch.arange(images.size(0), device=device)
        loss = (F.cross_entropy(logits_img, labels) + F.cross_entropy(logits_txt, labels)) / 2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if i % 10 == 0:
            it_time = time.time() - start
            wandb.log({"train_loss": loss.item(), "step": i, "sec_per_it": it_time})
            print(f"Step {i:4d} | Loss: {loss.item():.4f} | {it_time:.2f}s/it")

        if i % 500 == 0 and i > 0:
            save_checkpoint(model, optimizer, i, loss.item())

    save_checkpoint(model, optimizer, Config.max_iters, loss.item())

if __name__ == "__main__":
    train()
