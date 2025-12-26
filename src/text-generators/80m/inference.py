import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# =============================================== #
#        Core Architecture (from training)        #
# =============================================== #

class RWKVMambaHybrid(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.w_mix = nn.Parameter(torch.ones(d_model) * 0.5)
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1) 

    def forward(self, x):
        B, T, C = x.shape
        h = torch.zeros(B, C, device=x.device)
        s = torch.zeros(B, self.d_state, device=x.device)
        outputs = []
        
        for t in range(T):
            x_t = x[:, t, :]
            h = self.w_mix * h + (1 - self.w_mix) * x_t
            s = s @ self.A.T + x_t @ self.B.T
            y_t = s @ self.C.T + h * self.D
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)

class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
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
            mask = mask.expand(B, self.n_heads, T, T).bool()
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class i3HybridBlock(nn.Module):
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

class i3Model(nn.Module):
    def __init__(self, vocab_size, d_model=2048, n_heads=16, max_seq_len=512, d_state=64, conv_layers=10, attn_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        hybrid_layers = [i3HybridBlock(d_model, d_state=d_state, ffn_mult=4) for _ in range(conv_layers)]
        attention_layers = [i3AttentionBlock(d_model, n_heads=n_heads, ffn_mult=4) for _ in range(attn_layers)]
        
        self.layers = nn.ModuleList(hybrid_layers + attention_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Handle sequence length constraint gracefully during inference
        if T > self.max_seq_len:
            idx = idx[:, -self.max_seq_len:]
            T = self.max_seq_len
            
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos_embed(pos)
        
        # Causal mask for attention layers
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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

# =============================================== #
#                  Tokenizer                      #
# =============================================== #

class ChunkTokenizer:
    def __init__(self):
        self.chunk_to_idx = {}
        self.idx_to_chunk = {}
        self.vocab_size = 0
        self.unk_token = '<UNK>'
        self.unk_idx = 0
        self.common_trigrams = frozenset({
            'the', 'and', 'ing', 'ion', 'tio', 'for', 'tha', 
            'ter', 'hat', 'his', 'ere', 'ent', 'her', 'was',
            'you', 'are', 'not', 'but', 'can', 'all', 'whi',
            'one', 'our', 'out', 'whe', 'hav', 'thi', 'wit'
        })

    def _should_use_3char(self, pos, text):
        if pos + 3 > len(text): return False
        chunk_3 = text[pos:pos+3]
        if chunk_3 in self.common_trigrams: return True
        if pos > 0 and text[pos-1] == ' ': return True
        if pos + 3 < len(text) and text[pos+3] == ' ': return True
        return False

    def encode(self, text):
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
        # Handle tensor inputs or list inputs
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        return ''.join([self.idx_to_chunk.get(int(i), self.unk_token) for i in indices])

    def load(self, path):
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.chunk_to_idx = vocab_data['chunk_to_idx']
        self.idx_to_chunk = {int(k): v for k, v in vocab_data['idx_to_chunk'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.unk_token = vocab_data.get('unk_token', '<UNK>')
        self.unk_idx = vocab_data.get('unk_idx', 0)
        print(f"Tokenier loaded: {self.vocab_size} chunks")

# =============================================== #
#                 Inference Logic                 #
# =============================================== #

def load_model(model_dir, device):
    """Loads configuration and weights"""
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "pytorch_model.bin") # HF Style
    # Check for alternate name from training script
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "pytorch_model_best.bin")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find config.json or pytorch_model.bin in {model_dir}")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 2. Initialize Model
    print(f"Initializing i3 Model with d_model={config['d_model']}, layers={config['n_layers']}...")
    model = i3Model(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        d_state=config['d_state'],
        conv_layers=config.get('conv_layers', 10),
        attn_layers=config.get('attn_layers', 6)
    )

    # 3. Load Weights
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle if checkpoint saves a dict with 'model_state_dict' key or just the state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model, config

def main():
    parser = argparse.ArgumentParser(description="i3 Hybrid Model Inference")
    parser.add_argument("--model_dir", type=str, default="i3-model-artifacts", help="Path to model directory")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--length", type=int, default=100, help="Maximum new tokens to generate")
    args = parser.parse_args()

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Load Tokenizer
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json") # HF style
    if not os.path.exists(tokenizer_path):
         # Try config name if default fails
         tokenizer_path = os.path.join(args.model_dir, "vocab.json")

    tokenizer = ChunkTokenizer()
    try:
        tokenizer.load(tokenizer_path)
    except FileNotFoundError:
        print(f"Error: Could not find tokenizer.json in {args.model_dir}")
        return

    # Load Model
    try:
        model, config = load_model(args.model_dir, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "="*50)
    print("i3 INFERENCE MODE")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            prompt = input("User: ")
            if prompt.lower() in ['quit', 'exit']:
                break
            
            if not prompt.strip():
                continue

            # Encode
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

            # Generate
            start_time = time.time()
            output_ids = model.generate(
                input_tensor, 
                max_new_tokens=args.length, 
                temperature=args.temp, 
                top_k=args.top_k
            )
            end_time = time.time()

            # Decode
            # Only decode the new tokens
            generated_ids = output_ids[0].tolist()
            full_text = tokenizer.decode(generated_ids)
            
            # Print result
            print(f"\ni3: {full_text}")
            
            tokens_gen = len(generated_ids) - len(input_ids)
            print(f"\n[Stats: {tokens_gen} tokens in {end_time - start_time:.2f}s]")
            print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
