import torch
import torch.nn as nn
import math
import time
import os
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

print("🚀 Starting adjusted large-scale length extrapolation experiment...")
print("Estimated runtime: 48-72 hours")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Create save directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)

#  Global start time - fix: define before training starts
experiment_start_time = time.time()

# Step 1: Load dataset (use partial data to reduce memory)
print("\n" + "="*80)
print("📥 Loading WikiText-103 dataset (using partial data)...")
print("="*80)

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

def filter_and_clean(texts, min_length=100):
    """Filter and clean text"""
    cleaned = []
    for text in texts:
        if not text.strip():
            continue
        if len(text.strip()) < min_length:
            continue
        if text.startswith(' = ') and text.endswith(' = '):
            continue
        cleaned.append(text.strip())
    return cleaned

print("Filtering training data...")
train_texts = filter_and_clean(dataset["train"]["text"])
print("Filtering validation data...")
valid_texts = filter_and_clean(dataset["validation"]["text"])
print("Filtering test data...")
test_texts = filter_and_clean(dataset["test"]["text"])

# Use partial data to reduce memory usage
train_texts = train_texts[:2000]  # Use 2000 training documents
valid_texts = valid_texts[:500]   # Use 500 validation documents
test_texts = test_texts[:500]     # Use 500 test documents

train_text = "\n".join(train_texts)
valid_text = "\n".join(valid_texts)
test_text = "\n".join(test_texts)

print(f"\nProcessed data statistics:")
print(f"  Training documents: {len(train_texts):,}")
print(f"  Validation documents: {len(valid_texts):,}")
print(f"  Test documents: {len(test_texts):,}")
print(f"  Training text length: {len(train_text):,} characters")
print(f"  Validation text length: {len(valid_text):,} characters")

#  Step 2: Initialize Tokenizer
print("\n" + "="*80)
print("🔤 Initializing Tokenizer...")
print("="*80)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

#  Step 3: Efficient dataset construction
print("\n" + "="*80)
print("📊 Building dataset...")
print("="*80)

def build_efficient_dataset(text, seq_len=512, max_samples=20000):
    """Build efficient dataset with reduced memory usage"""
    print(f"Building dataset: seq_len={seq_len}, max_samples={max_samples}")

    # Process in chunks
    chunk_size = 200000  # 200K characters per chunk
    all_X, all_Y = [], []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if len(chunk) < seq_len + 100:
            continue

        encoded = tokenizer(chunk, return_tensors="pt", add_special_tokens=False, truncation=False)
        if encoded["input_ids"].numel() == 0:
            continue

        ids = encoded["input_ids"][0]

        if len(ids) < seq_len + 1:
            continue

        # Use sliding window
        stride = seq_len // 2
        X, Y = [], []
        for j in range(0, len(ids) - seq_len, stride):
            x = ids[j:j+seq_len]
            y = ids[j+1:j+seq_len+1]
            X.append(x)
            Y.append(y)

            if len(X) >= max_samples // 10:  # Limit samples per chunk
                break

        all_X.extend(X)
        all_Y.extend(Y)

        if len(all_X) >= max_samples:
            break

    if not all_X:
        return torch.tensor([]), torch.tensor([])

    all_X = all_X[:max_samples]
    all_Y = all_Y[:max_samples]

    print(f"Generated {len(all_X)} samples")
    return torch.stack(all_X), torch.stack(all_Y)

# Build training dataset
print("\nBuilding training dataset (seq_len=512)...")
train_X, train_Y = build_efficient_dataset(train_text, seq_len=512, max_samples=30000)

print("\nBuilding validation dataset (seq_len=512)...")
val_X, val_Y = build_efficient_dataset(valid_text, seq_len=512, max_samples=5000)

print("\nBuilding test dataset (seq_len=512)...")
test_X, test_Y = build_efficient_dataset(test_text, seq_len=512, max_samples=5000)

if len(train_X) == 0:
    print(" Training dataset is empty, exiting!")
    exit()

print(f"\n📈 Dataset statistics:")
print(f"  Training set: {len(train_X):,} samples")
print(f"  Validation set: {len(val_X):,} samples")
print(f"  Test set: {len(test_X):,} samples")
print(f"  Tokens per sample: 512")

#  Step 4: Create data loaders
train_loader = DataLoader(
    TensorDataset(train_X, train_Y),
    batch_size=8,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    TensorDataset(val_X, val_Y),
    batch_size=8,
    shuffle=False,
    pin_memory=True
)

test_loader = DataLoader(
    TensorDataset(test_X, test_Y),
    batch_size=8,
    shuffle=False,
    pin_memory=True
)

#  Step 5: Adjusted model architecture (reduced memory usage)
print("\n" + "="*80)
print("🤖 Defining adjusted model architecture...")
print("="*80)

class EfficientAttention(nn.Module):
    """Efficient attention mechanism with reduced memory usage"""
    def __init__(self, d_model=512, n_heads=8, rope=False, alibi=False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.alibi = alibi
        self.dropout = dropout

        assert d_model % n_heads == 0

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # ALiBi slopes
        if alibi:
            slopes = self._get_alibi_slopes(n_heads)
            self.register_buffer("alibi_slopes", torch.tensor(slopes, dtype=torch.float32))

    def _get_alibi_slopes(self, n_heads):
        """Generate ALiBi slopes"""
        return [2**(-8 * (i+1) / n_heads) for i in range(n_heads)]

    def apply_rope(self, x):
        """RoPE implementation"""
        batch_size, seq_len, dim = x.shape

        if dim % 2 != 0:
            dim = dim - 1
            x = x[..., :dim]

        half_dim = dim // 2

        # Position encoding
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        positions = positions.unsqueeze(1)

        # Frequencies
        freqs = torch.arange(0, half_dim, device=x.device, dtype=torch.float32)
        freqs = 1.0 / (10000 ** (freqs / half_dim))

        # Angles
        angles = positions * freqs.unsqueeze(0)

        # Sine and cosine
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # Split tensor
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:2*half_dim]

        # Apply rotation
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        # Combine
        rotated = torch.cat([x1_rotated, x2_rotated], dim=-1)

        if x.shape[-1] > dim:
            rotated = torch.cat([rotated, x[..., dim:]], dim=-1)

        return rotated

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Projections
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rope:
            q_flat = q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            k_flat = k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

            q_rotated = self.apply_rope(q_flat)
            k_rotated = self.apply_rope(k_flat)

            q = q_rotated.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k_rotated.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply ALiBi
        if self.alibi:
            # Create relative position bias
            context_pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)[:, None]
            memory_pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)[None, :]
            relative_pos = memory_pos - context_pos

            alibi_bias = relative_pos.unsqueeze(0).unsqueeze(0)
            alibi_bias = alibi_bias * self.alibi_slopes.view(1, self.n_heads, 1, 1)
            alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1)

            scores = scores + alibi_bias

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Attention calculation
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # Combine multi-head
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out(attn_output)

        return attn_output

class EfficientTransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, rope=False, alibi=False, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = EfficientAttention(d_model, n_heads, rope, alibi, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention + residual connection
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out

        # Feed-forward network + residual connection
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out

        return x

class EfficientTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12,
                 rope=False, alibi=False, dropout=0.1):
        super().__init__()
        assert not (rope and alibi), "Cannot use both RoPE and ALiBi"

        self.d_model = d_model
        self.n_layers = n_layers

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(d_model, n_heads, rope, alibi, dropout)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Output layer
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x, y=None):
        # Token embedding
        h = self.embed(x)
        h = self.dropout(h)

        # Pass through all layers
        for layer in self.layers:
            h = layer(h)

        # Final normalization
        h = self.ln_f(h)

        # Language model head
        logits = self.lm_head(h)

        if y is None:
            return logits

        # Calculate loss
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    def get_num_params(self):
        """Return number of model parameters"""
        return sum(p.numel() for p in self.parameters())

#  Step 6: Initialize models
print("\n" + "="*80)
print("🚀 Initializing models...")
print("="*80)

# Create RoPE model
rope_model = EfficientTransformer(
    vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=12,
    rope=True,
    dropout=0.1
).cuda()

# Create ALiBi model
alibi_model = EfficientTransformer(
    vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=12,
    alibi=True,
    dropout=0.1
).cuda()

print(f"📊 Model statistics:")
print(f"  RoPE model parameters: {rope_model.get_num_params():,}")
print(f"  ALiBi model parameters: {alibi_model.get_num_params():,}")
print(f"  Each model size: {rope_model.get_num_params() * 4 / (1024**3):.2f} GB (float32)")

#  Step 7: Long-term training function
print("\n" + "="*80)
print("🏋️ Starting long-term training (20 epochs)...")
print("="*80)

def train_model(model, train_loader, val_loader, epochs=20, model_name="Model"):
    """Train model"""
    print(f"\n🎯 Starting training for {model_name}, {epochs} epochs")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=1e-6
    )

    # Training statistics
    train_losses = []
    val_ppls = []
    best_val_ppl = float('inf')
    model_start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        # Show progress with tqdm
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}")

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.cuda(), y.cuda()

            # Forward pass
            loss = model(x, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # Record every 100 batches
            if batch_idx % 100 == 0:
                train_losses.append(loss.item())

        # Calculate epoch average loss
        avg_epoch_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        print(f"\n{model_name} - Epoch {epoch+1} completed, took {epoch_time/60:.2f} minutes")
        print(f"  Average loss: {avg_epoch_loss:.4f}")

        # Validation
        if val_loader is not None:
            val_ppl = evaluate_model_ppl(model, val_loader)
            val_ppls.append(val_ppl)
            print(f"  Validation PPL: {val_ppl:.2f}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                print(f"  🎉 New best validation PPL: {best_val_ppl:.2f}")

                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_ppl': best_val_ppl,
                }, f'checkpoints/best_{model_name}.pt')

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'train_losses': train_losses,
                'val_ppls': val_ppls
            }
            torch.save(checkpoint, f'checkpoints/checkpoint_{model_name}_epoch_{epoch+1}.pt')
            print(f"  Checkpoint saved: checkpoints/checkpoint_{model_name}_epoch_{epoch+1}.pt")

        # Print detailed statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - model_start_time
            print(f"\n📊 Training progress report ({model_name}):")
            print(f"  Elapsed time: {elapsed_time/3600:.2f} hours")
            print(f"  Estimated remaining time: {elapsed_time/(epoch+1) * (epochs-epoch-1)/3600:.2f} hours")

    total_time = time.time() - model_start_time
    print(f"\n {model_name} training completed!")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Best validation PPL: {best_val_ppl:.2f}")

    return train_losses, val_ppls, best_val_ppl

def evaluate_model_ppl(model, loader):
    """Calculate perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.cuda(), y.cuda()

            logits = model(x)
            loss = nn.CrossEntropyLoss(reduction='sum')(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    model.train()
    return ppl

#  Step 8: Train both models
print("\n" + "="*80)
print(" Starting RoPE model training...")
print("="*80)

rope_losses, rope_val_ppls, rope_best_ppl = train_model(
    rope_model, train_loader, val_loader,
    epochs=20, model_name="RoPE_Model"
)

print("\n" + "="*80)
print("🔥 Starting ALiBi model training...")
print("="*80)

alibi_losses, alibi_val_ppls, alibi_best_ppl = train_model(
    alibi_model, train_loader, val_loader,
    epochs=20, model_name="ALiBi_Model"
)

#  Step 9: Length extrapolation testing
print("\n" + "="*80)
print("🌌 Starting length extrapolation testing...")
print("="*80)

def test_extrapolation(model, text, model_name, max_length=8192):
    """Test length extrapolation capability"""
    print(f"\n🔬 Testing length extrapolation for {model_name}...")

    test_lengths = [256, 512, 1024, 2048, 4096, 8192]
    results = {}

    for length in test_lengths:
        if length > max_length:
            print(f"  Skipping length {length} (exceeds max limit {max_length})")
            results[length] = float('inf')
            continue

        try:
            print(f"\n  Testing length: {length}")

            # Create test dataset
            X, Y = build_efficient_dataset(
                text,
                seq_len=length,
                max_samples=100
            )

            if len(X) == 0:
                print(f"    ❌ No valid data for length {length}")
                results[length] = float('inf')
                continue

            # Create data loader
            loader = DataLoader(
                TensorDataset(X, Y),
                batch_size=2,  # Use small batch size for long sequences
                shuffle=False
            )

            # Evaluate
            ppl = evaluate_model_ppl(model, loader)
            results[length] = ppl

            print(f"    Length {length}: PPL = {ppl:.2f}")

            # Save intermediate results
            interim_results = {
                'model': model_name,
                'length': length,
                'ppl': ppl,
                'timestamp': time.time()
            }
            torch.save(interim_results, f'results/{model_name}_length_{length}_result.pt')

        except torch.cuda.OutOfMemoryError:
            print(f"     Length {length} exceeds GPU memory")
            results[length] = float('inf')
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"     Error at length {length}: {str(e)}")
            results[length] = float('inf')

    return results

# Perform extrapolation testing on test set
print("\nTesting RoPE model length extrapolation...")
rope_extreme_results = test_extrapolation(rope_model, test_text, "RoPE", max_length=4096)

print("\nTesting ALiBi model length extrapolation...")
alibi_extreme_results = test_extrapolation(alibi_model, test_text, "ALiBi", max_length=4096)

#  Step 10: Result analysis and saving
print("\n" + "="*80)
print("📊 Final result analysis and saving")
print("="*80)

# Aggregate all results
final_results = {
    'training': {
        'rope_train_losses': rope_losses,
        'rope_val_ppls': rope_val_ppls,
        'rope_best_ppl': rope_best_ppl,
        'alibi_train_losses': alibi_losses,
        'alibi_val_ppls': alibi_val_ppls,
        'alibi_best_ppl': alibi_best_ppl,
    },
    'extrapolation': {
        'rope_results': rope_extreme_results,
        'alibi_results': alibi_extreme_results,
    },
    'config': {
        'model_size': '512x12x8',
        'train_seq_len': 512,
        'train_epochs': 20,
        'train_samples': len(train_X),
        'val_samples': len(val_X),
        'test_samples': len(test_X),
        'rope_params': rope_model.get_num_params(),
        'alibi_params': alibi_model.get_num_params(),
    },
    'timing': {
        'experiment_start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(experiment_start_time)),
        'experiment_end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_hours': (time.time() - experiment_start_time) / 3600,
    }
}

# Save final results
results_file = 'results/final_experiment_results.pt'
torch.save(final_results, results_file)

# Save as JSON for readability
json_file = 'results/final_experiment_results.json'
with open(json_file, 'w') as f:
    # Convert tensors to Python scalars
    json_results = {}
    for key, value in final_results.items():
        if key == 'training':
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    json_results[key][subkey] = [float(v) if torch.is_tensor(v) else v for v in subvalue]
                else:
                    json_results[key][subkey] = float(subvalue) if torch.is_tensor(subvalue) else subvalue
        elif key == 'extrapolation':
            json_results[key] = {}
            for subkey, subvalue in value.items():
                json_results[key][subkey] = {}
                for length, ppl in subvalue.items():
                    json_results[key][subkey][length] = float(ppl) if torch.is_tensor(ppl) else ppl
        else:
            json_results[key] = value

    json.dump(json_results, f, indent=2)

print(f"\n All results saved!")
print(f"  PyTorch format: {results_file}")
print(f"  JSON format: {json_file}")

# Print key findings
print("\n" + "="*80)
print("🔍 Key findings summary")
print("="*80)

# Analyze extrapolation results
valid_lengths = [l for l in rope_extreme_results.keys()
                if rope_extreme_results[l] < float('inf') and alibi_extreme_results[l] < float('inf')]

if valid_lengths:
    print(f"\n📈 Valid test lengths: {valid_lengths}")

    for length in valid_lengths:
        rope_ppl = rope_extreme_results[length]
        alibi_ppl = alibi_extreme_results[length]

        if rope_ppl > 0 and alibi_ppl > 0:
            advantage = ((rope_ppl - alibi_ppl) / rope_ppl) * 100
        else:
            advantage = 0

        print(f"\n  Length {length}:")
        print(f"    RoPE PPL: {rope_ppl:,.2f}")
        print(f"    ALiBi PPL: {alibi_ppl:,.2f}")
        print(f"    ALiBi advantage: {advantage:+.1f}%")

    # Find length with maximum ALiBi advantage
    if len(valid_lengths) > 1:
        valid_for_advantage = [l for l in valid_lengths if rope_extreme_results[l] > 0]
        if valid_for_advantage:
            best_adv_length = max(valid_for_advantage,
                                key=lambda l: ((rope_extreme_results[l] - alibi_extreme_results[l]) /
                                              rope_extreme_results[l]))
            best_advantage = ((rope_extreme_results[best_adv_length] - alibi_extreme_results[best_adv_length]) /
                             rope_extreme_results[best_adv_length]) * 100

            print(f"\n🏆 Maximum ALiBi advantage at length {best_adv_length}: {best_advantage:+.1f}%")

            # Trend analysis
            if len(valid_for_advantage) >= 3:
                lengths_sorted = sorted(valid_for_advantage)
                advantages = []
                for l in lengths_sorted:
                    if rope_extreme_results[l] > 0:
                        adv = ((rope_extreme_results[l] - alibi_extreme_results[l]) / rope_extreme_results[l]) * 100
                        advantages.append(adv)

                if len(advantages) >= 3:
                    # Calculate trend (simple linear fit)
                    x = np.array(lengths_sorted[:len(advantages)])
                    y = np.array(advantages)
                    coeff = np.polyfit(x, y, 1)

                    print(f"📈 Advantage trend: For every 1000 length increase, ALiBi advantage increases by {coeff[0]*1000:.1f}%")
                    if coeff[0] > 0:
                        print("   Trend: ALiBi advantage increases with length ✅")
                    else:
                        print("   Trend: ALiBi advantage decreases with length ❌")
else:
    print("\n⚠️ No valid length extrapolation results")

print(f"\n🎉 Experiment completed!")
print(f"  Total runtime: {(time.time() - experiment_start_time) / 3600:.2f} hours")
print(f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Create result visualizations
try:
    print("\n📊 Generating result visualizations...")

    # ===============================
    # Visualization of results
    # ===============================
    plt.figure(figsize=(15, 5))

    # ----------------------
    # 1. Training Loss Curve
    # ----------------------
    plt.subplot(1, 3, 1)
    plt.plot(rope_losses, label='RoPE Training Loss', alpha=0.8)
    plt.plot(alibi_losses, label='ALiBi Training Loss', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ----------------------
    # 2. Validation PPL Curve
    # ----------------------
    rope_val_curve = rope_val_ppls
    alibi_val_curve = alibi_val_ppls

    plt.subplot(1, 3, 2)
    plt.plot(rope_val_curve, 'o-', label='RoPE Validation PPL')
    plt.plot(alibi_val_curve, 's-', label='ALiBi Validation PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity (PPL)')
    plt.title('Validation PPL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # ----------------------
    # 3. Length Extrapolation Performance
    # ----------------------
    plt.subplot(1, 3, 3)

    # Filter valid lengths
    valid_lengths = [
        l for l in rope_extreme_results.keys()
        if rope_extreme_results[l] < float('inf') and alibi_extreme_results[l] < float('inf')
    ]

    if valid_lengths:
        rope_ppls = [rope_extreme_results[l] for l in valid_lengths]
        alibi_ppls = [alibi_extreme_results[l] for l in valid_lengths]

        plt.plot(valid_lengths, rope_ppls, 'o-', label='RoPE', linewidth=2)
        plt.plot(valid_lengths, alibi_ppls, 's-', label='ALiBi', linewidth=2)
        plt.xlabel('Sequence Length')
        plt.ylabel('Perplexity (PPL)')
        plt.title('Length Extrapolation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig('results/extrapolation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(" Visualization charts saved to results/extrapolation_results.png")

except Exception as e:
    print(f" Visualization generation failed: {e}")


