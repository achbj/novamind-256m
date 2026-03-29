# Prompt for Next Chat: NovaMind-256M Implementation

> Copy and paste everything below the horizontal rule into a new chat.

---

I want to build from scratch a **novel 256M parameter Decoder-Only Language Model** in PyTorch called **NovaMind-256M**. This is a research project targeting a paper publication. Please act as an **elite AI research scientist and PyTorch engineer**.

---

## ⚠️ CRITICAL CONTEXT — Why We Are Here

We previously built a 60M parameter model (`PiGPT-60M`) that **completely failed** — it could not predict even the second word of a sentence correctly. The root causes were:
1. **Too small** — 60M params cannot hold enough language knowledge for conversation
2. **Underfitted** — only trained on TinyStories, which is too simple
3. **No reasoning grounding** — no CoT training
4. **Tokenizer mismatch** — GPT-2 tokenizer was inefficient for conversation
5. **Training instability** — FP16 on T4 was unstable, no proper monitoring

This time we fix **every single one of these problems.**

---

## Architecture Requirements (Exact Specs)

### Core Hyperparameters
```python
vocab_size  = 32000      # LLaMA-2 tokenizer (NOT GPT-2)
d_model     = 1024
n_heads     = 16         # query heads
n_kv_heads  = 4          # key/value heads (GQA, ratio 4:1)
n_layers    = 24
head_dim    = 64         # d_model // n_heads
ff_dim      = 2816       # ≈ 2.75 × d_model, divisible by 128
max_seq_len = 2048       # training context
dropout     = 0.0        # no dropout (modern LLM best practice)
# Target: ~252M parameters after weight tying
```

### Core Components (Non-negotiable)
- **Pre-RMSNorm** before every attention and FFN sub-layer (NOT post-norm)
- **SwiGLU FFN** with 3 matrices (gate, up, down), no bias
- **GQA (16Q / 4KV)** with `F.scaled_dot_product_attention` (Flash Attention path)
- **Weight Tying** — `lm_head.weight = token_embedding.weight` (saves 33M params)
- **No bias** in any Linear layer
- **BF16 training** (H100 native — more stable than FP16)
- **Gradient checkpointing** support (to fit on H100 40GB)
- **Scaled residual initialization**: `std = 0.02 / sqrt(2 * n_layers)` for output projections

---

## 🌟 Novel Contribution 1: HierRoPE (Hierarchical Rotary Position Embedding)

This is our PRIMARY novel architectural contribution. Implement it exactly as follows:

**Core idea:** Split each attention head's `head_dim=64` dimensions into two halves:
- **Dims 0–31 (first half): Local RoPE** with `base=10,000`
  - Encodes fine-grained position WITHIN a single conversation turn
- **Dims 32–63 (second half): Global RoPE** with `base=500,000`
  - Encodes coarse position ACROSS conversation turns (much slower rotation = longer range)

**Implementation:**
```python
def build_hier_rope_cache(seq_len, head_dim, device, local_base=10000, global_base=500000):
    half = head_dim // 2
    # Local frequencies (first half of head dim)
    local_theta = 1.0 / (local_base ** (torch.arange(0, half, 2, device=device).float() / half))
    # Global frequencies (second half of head dim)
    global_theta = 1.0 / (global_base ** (torch.arange(0, half, 2, device=device).float() / half))
    
    positions = torch.arange(seq_len, device=device).float()
    
    local_freqs  = torch.outer(positions, local_theta)
    global_freqs = torch.outer(positions, global_theta)
    
    # cat to full head_dim
    local_freqs  = torch.cat([local_freqs, local_freqs], dim=-1)    # shape: [seq, half]
    global_freqs = torch.cat([global_freqs, global_freqs], dim=-1)  # shape: [seq, half]
    
    # Combined: [local_cos, global_cos], [local_sin, global_sin]
    cos = torch.cat([local_freqs.cos(), global_freqs.cos()], dim=-1)  # shape: [seq, head_dim]
    sin = torch.cat([local_freqs.sin(), global_freqs.sin()], dim=-1)  # shape: [seq, head_dim]
    return cos, sin
```

Apply RoPE identically to Q and K before attention. The split happens implicitly — since dims 0-31 use local_theta and dims 32-63 use global_theta, the rotation naturally encodes hierarchical position.

**This costs ZERO extra parameters.** It is purely a change in the frequency basis.

---

## 🌟 Novel Contribution 2: Tag-Aware Loss Curriculum

This is our SECONDARY novel contribution — a training methodology (not architecture).

Our training data uses XML-style tags:
```xml
<human>What is 2+2?</human>
<think>The user is asking a basic arithmetic question. 2+2 equals 4.</think>
<assistant>2+2 = 4.</assistant>
```

Token categories:
- **THINK tokens** — inside `<think>...</think>` — these are the model's reasoning
- **RESPONSE tokens** — inside `<assistant>...</assistant>` — the actual output
- **USER/SYSTEM tokens** — inside `<human>...</human>` or system — always masked (weight=0)

The loss function must support **per-token loss weights** that change by training phase:

```python
# Phase weights for <think> tokens:
THINK_WEIGHTS = {
    "pretrain":  0.0,   # Phase 1: don't train on think tokens yet
    "sft":       0.5,   # Phase 2: soft introduction to reasoning structure
    "cot_sft":   1.5,   # Phase 3: BOOST think tokens — force careful reasoning
}
# Response tokens are always 1.0
# User/system tokens are always 0.0 (masked via ignore_index or weight=0)
```

The loss function signature:
```python
def tag_aware_loss(logits, targets, token_weights):
    """
    logits:        [B, T, V]
    targets:       [B, T]      — -1 for masked positions
    token_weights: [B, T]      — per-token float weights
    """
    # Use weighted cross-entropy, ignoring index=-1 positions
```

---

## Full Training Pipeline (3 Phases)

### Phase 1: Language Pretraining
- **Data:** FineWeb-Edu (1.5B tokens) + TinyStories (1B tokens)
- **Loss weights:** response=1.0, think=0.0 (no CoT data yet), user=0.0
- **Tokens:** ~3B total
- **Batch:** 32 sequences × 2048 tokens = 65K tokens/batch
- **Grad accumulation:** 8 steps → effective batch = 512K tokens
- **LR:** 3e-4, cosine decay to 3e-5, 2000-step warmup
- **Duration:** ~14 hours on H100 40GB

### Phase 2: Instruction SFT
- **Data:** ShareGPT (filtered, 4-turn), Alpaca (cleaned), DailyDialog
- **Loss weights:** response=1.0, think=0.5, user=0.0
- **Total tokens:** ~300M
- **Batch:** 16 × 2048
- **LR:** 1e-4, cosine decay to 1e-5
- **Duration:** ~2.5 hours on H100

### Phase 3: CoT SFT (Reasoning Boost)
- **Data:** OpenHermes 2.5, Orca Math, GSM8K (all with `<think>` tags)
- **Loss weights:** response=1.0, think=1.5 (BOOSTED), user=0.0
- **Total tokens:** ~150M
- **Batch:** 8 × 2048
- **LR:** 5e-5, cosine decay to 5e-6
- **Duration:** ~1.5 hours on H100

---

## 📊 Logging Requirements (CRITICAL — We Cannot Afford to Fly Blind Again)

The H100 sessions on Lightning.ai are 1 hour each. We MUST log and checkpoint aggressively.

### Step-level logging (every optimizer step — use `wandb` AND a local JSON file):
```python
log = {
    "step": int,
    "epoch_frac": float,            # e.g., 0.47 (what fraction of epoch)
    "phase": str,                   # "pretrain" | "sft" | "cot_sft"
    "loss/train": float,
    "loss/think_component": float,  # loss contribution from <think> tokens only
    "loss/response_component": float,
    "lr": float,
    "grad_norm": float,             # CRITICAL for stability monitoring
    "tokens_seen_total": int,
    "tokens_per_sec": float,
    "gpu_mem_gb": float,
    "loss_weight_think": float,     # which curriculum we're in
    "time_elapsed_min": float,
}
```

### Every 500 steps:
- Log `val_loss` on held-out validation set
- Log example generation from 3 fixed test prompts:
  1. `"What is the capital of France?"`
  2. `"What is 15 × 17?"`
  3. `"Tell me a joke."`
- Save rolling checkpoint

### Checkpoint naming:
```
checkpoints/
  novamind_step{step:07d}_loss{loss:.4f}.pt   ← rolling (overwrite)
  novamind_milestone_{step:07d}.pt             ← every 2000 steps (keep all)
  novamind_phase1_final.pt
  novamind_phase2_final.pt
  novamind_phase3_final.pt
```

### Checkpoint content (must be complete for resuming after session end):
```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,       # for BF16 grad scaler
    "step": int,
    "phase": str,
    "tokens_seen": int,
    "config": dict,                  # full hyperparams
    "train_loss_history": list,
    "val_loss_history": list,
    "timestamp": str,
}
```

---

## What to Generate

Please produce **two files**:

### File 1: `model.py`
Complete PyTorch model implementation including:
1. `RMSNorm` class
2. `build_hier_rope_cache()` function (HierRoPE — our novel contribution)
3. `apply_hier_rope()` function
4. `GQACausalAttention` class (using HierRoPE for Q and K)
5. `SwiGLUFFN` class
6. `TransformerBlock` class (Pre-norm, attention, FFN, residuals)
7. `NovaMind256M` main model class with:
   - `gradient_checkpointing_enable/disable`
   - Scaled residual init
   - `forward(input_ids, targets, token_weights)` — where `token_weights` is optional
   - `tag_aware_loss()` static method
   - `generate()` method with temperature, top-p, top-k, repetition penalty
   - `count_parameters()` method that prints a breakdown table
8. A `__main__` block that verifies parameter count (must be ~252M)

### File 2: `train.py`
Complete training script including:
1. Full argument parser (phase, resume checkpoint, LR, batch size, etc.)
2. Dataset loading for all 3 phases
3. `TagAwareDataset` class that returns `(input_ids, targets, token_weights)` triplets
4. Training loop with:
   - BF16 `torch.autocast`
   - `GradScaler`
   - Gradient accumulation
   - Grad norm clipping (max=1.0) and logging
   - All logging as specified above
   - Checkpoint saving (every 500 steps rolling + every 2000 milestone)
   - Resume from checkpoint (detect `tokens_seen`, `step`, optimizer state)
5. Validation loop (every 500 steps)
6. Sample generation at validation checkpoints
7. `wandb` integration (with graceful fallback to local JSON if wandb not available)
8. Cosine LR schedule with linear warmup

### Critical constraints:
- All code must be **production quality** — not prototype/demo quality
- Every non-obvious line must be commented with WHY, not just WHAT
- The HierRoPE implementation must be clearly labeled as a novel contribution with a docstring
- The tag-aware loss must clearly document the curriculum phases
- The training script must be **resumable mid-epoch** — Lightning.ai sessions end at 1 hour
- Use `torch.compile` where beneficial (PyTorch 2.0+)
- Print a full parameter count table at model init showing each component's contribution
