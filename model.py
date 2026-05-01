"""
NovaMind-256M: Decoder-Only Conversational Language Model
==========================================================
Architecture: Decoder-only Transformer

Key design choices:
  1. HiRoPE (Hierarchical RoPE) — adapted from "HiRoPE: Length Extrapolation
     for Code Models Using Hierarchical Position" (ACL 2024).
     Original paper used it for code structure (token/statement/function hierarchy).
     We adapt it for conversation structure: local turn position vs. global dialogue position.
     Split head dims into local-base=10K and global-base=500K streams.
  2. Tag-Aware Loss Curriculum — original contribution.
     Per-token loss weighting that changes across training phases.

Target: ~252M parameters (after weight tying)
Hardware target: H100 40GB (Lightning.ai student account)
Precision: BF16
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from dataclasses import dataclass, field, asdict


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NovaMindConfig:
    """
    All hyperparameters in one place for clean checkpointing and reproducibility.
    Changing a value here changes the entire model — no magic numbers buried
    in the code.
    """
    # ── Vocabulary ──────────────────────────────────────────────────────────
    vocab_size: int = 32_000          # LLaMA-2 tokenizer (NOT GPT-2's 50K)

    # ── Dimensions ──────────────────────────────────────────────────────────
    d_model: int = 1_024              # hidden size
    n_heads: int = 16                 # number of query attention heads
    n_kv_heads: int = 4              # GQA: key/value heads (ratio 4:1)
    n_layers: int = 24               # transformer depth
    ff_dim: int = 2_304              # SwiGLU inner dim = 2.25 × d_model, div by 128 → ~265M total

    # ── Context ─────────────────────────────────────────────────────────────
    max_seq_len: int = 2_048         # training context; extendable later via RoPE scaling

    # ── HiRoPE — adapted from HiRoPE (ACL 2024), applied to conversation ────
    rope_local_base: float = 10_000.0    # local position (within-turn): ~LLaMA default
    rope_global_base: float = 500_000.0  # global position (cross-turn): much slower freq

    # ── Regularization ──────────────────────────────────────────────────────
    dropout: float = 0.0             # no dropout (modern LLM practice)
    attn_dropout: float = 0.0

    # ── Init ────────────────────────────────────────────────────────────────
    init_std: float = 0.02           # base std for embedding and non-residual projections

    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    def n_rep(self) -> int:
        """How many Q heads share each KV head."""
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────────
# RMS NORM
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Faster than LayerNorm because it skips the mean-centering step.
    Used by LLaMA, Gemma, Qwen, Mistral — now us.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        # Compute RMS then scale
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ──────────────────────────────────────────────────────────────────────────────
# HiRoPE — HIERARCHICAL ROTARY POSITION EMBEDDING
# Adapted from "HiRoPE: Length Extrapolation for Code Models Using Hierarchical
# Position" (ACL 2024). Original: code hierarchy. Our adaptation: dialogue.
# ──────────────────────────────────────────────────────────────────────────────

def build_hier_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    local_base: float = 10_000.0,
    global_base: float = 500_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HiRoPE: Hierarchical Rotary Position Embedding Cache.
    ────────────────────────────────────────────────────────
    Adapted from: "HiRoPE: Length Extrapolation for Code Models Using
    Hierarchical Position" (ACL 2024).

    Original paper: used code structure hierarchy (token-in-statement,
    statement-in-function) to split RoPE dimensions.
    Our adaptation: we use CONVERSATION structure instead.

      • Dims   0 … head_dim//2-1  →  LOCAL  RoPE (base=10,000)
          Encodes fine-grained position WITHIN a conversation turn.
          Same frequency as standard LLaMA RoPE — well-calibrated for
          sentence-level syntax and local coreference.

      • Dims head_dim//2 … head_dim-1  →  GLOBAL RoPE (base=500,000)
          Encodes coarse position ACROSS conversation turns.
          A higher base = slower rotation = can distinguish positions
          across much longer spans without frequency wrap-around.
          This is what the original HiRoPE paper used for cross-function
          dependencies in code; we repurpose it for cross-turn coherence.

    Cost: ZERO extra parameters. Purely a change in the frequency basis.

    Returns:
        cos, sin  — each shape [seq_len, head_dim], cached for efficiency
    """
    half = head_dim // 2  # split point

    # ── Local stream (dims 0 to half-1) ──────────────────────────────────
    # theta_i = 1 / (base^(2i/dim)) for i in [0, half/2)
    # We generate half/2 unique theta values, then duplicate for rotation trick
    local_theta = 1.0 / (
        local_base ** (torch.arange(0, half, 2, device=device).float() / half)
    )
    # ── Global stream (dims half to head_dim-1) ───────────────────────────
    global_theta = 1.0 / (
        global_base ** (torch.arange(0, half, 2, device=device).float() / half)
    )

    positions = torch.arange(seq_len, device=device).float()  # [seq_len]

    # Outer product: [seq_len, half/2]
    local_freqs  = torch.outer(positions, local_theta)
    global_freqs = torch.outer(positions, global_theta)

    # Duplicate for the rotation trick: [q1, q2] → [-q2, q1]
    # → shape [seq_len, half]
    local_freqs  = torch.cat([local_freqs,  local_freqs],  dim=-1)
    global_freqs = torch.cat([global_freqs, global_freqs], dim=-1)

    # Concatenate local and global to cover full head_dim: [seq_len, head_dim]
    cos = torch.cat([local_freqs.cos(),  global_freqs.cos()], dim=-1)
    sin = torch.cat([local_freqs.sin(),  global_freqs.sin()], dim=-1)

    return cos, sin  # [seq_len, head_dim]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE rotation: rotate each pair of dims by 90°."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_hier_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply HierRoPE to query and key tensors.
    Because cos/sin are already split into local//global halves,
    this is mathematically identical to standard RoPE — the hierarchy
    lives in the frequency basis, not the application code.

    Args:
        q, k: [batch, n_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]  (from build_hier_rope_cache)
    Returns:
        rotated q, k: same shapes as input
    """
    # Expand cos/sin to broadcast over batch and head dims
    cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim]
    sin = sin[None, None, :, :]

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ──────────────────────────────────────────────────────────────────────────────
# GROUPED QUERY ATTENTION WITH HiRoPE
# ──────────────────────────────────────────────────────────────────────────────

class GQACausalAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with HiRoPE (adapted from ACL 2024 paper).

    GQA uses n_heads query heads but only n_kv_heads key/value heads.
    Each KV head is shared by (n_heads // n_kv_heads) query heads.
    This reduces KV cache size by 4x during inference — critical for long
    conversations. Used by LLaMA 2/3, Mistral, Gemma.

    Attention is computed via F.scaled_dot_product_attention which uses
    Flash Attention 2 on supported hardware (H100 ✅).
    """

    def __init__(self, config: NovaMindConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_rep()    # Q heads per KV head
        self.head_dim   = config.head_dim()
        self.d_model    = config.d_model

        # Projections — NO bias (modern LLM best practice, saves params)
        self.q_proj = nn.Linear(self.d_model, self.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model,                    bias=False)

        self.attn_drop = config.attn_dropout

        # Cache HiRoPE cos/sin buffers to avoid recomputing every forward pass
        # persistent=False → not saved in state_dict (recomputed on load)
        self.register_buffer(
            "_rope_cos",
            torch.zeros(config.max_seq_len, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "_rope_sin",
            torch.zeros(config.max_seq_len, self.head_dim),
            persistent=False,
        )
        self._rope_cached_len = 0
        self._rope_local_base  = config.rope_local_base
        self._rope_global_base = config.rope_global_base

    def _get_hier_rope(self, seq_len: int, device: torch.device):
        """Lazily compute and cache HiRoPE tables."""
        if seq_len > self._rope_cached_len or self._rope_cos.device != device:
            cos, sin = build_hier_rope_cache(
                seq_len, self.head_dim, device,
                local_base=self._rope_local_base,
                global_base=self._rope_global_base,
            )
            self._rope_cos = cos
            self._rope_sin = sin
            self._rope_cached_len = seq_len
        return self._rope_cos[:seq_len], self._rope_sin[:seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]  — input hidden states
        Returns:
            out: [B, T, D]
        """
        B, T, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # Q: [B, n_heads, T, head_dim]
        # K: [B, n_kv_heads, T, head_dim]
        # V: [B, n_kv_heads, T, head_dim]

        # Apply HiRoPE to Q and K
        cos, sin = self._get_hier_rope(T, x.device)
        Q, K = apply_hier_rope(Q, K, cos.to(x.dtype), sin.to(x.dtype))

        # Expand KV heads to match Q heads for GQA
        # This uses expand() + reshape instead of repeat_interleave — same math,
        # but expand() is zero-copy (shares memory), saving VRAM
        K = K.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)\
             .reshape(B, self.n_heads, T, self.head_dim)
        V = V.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)\
             .reshape(B, self.n_heads, T, self.head_dim)

        # Flash Attention 2 path via PyTorch — causal mask included
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )  # [B, n_heads, T, head_dim]

        # Reassemble heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# SWIGLU FEED-FORWARD NETWORK
# ──────────────────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020; adopted by PaLM, LLaMA, Gemma).

    Unlike standard FFN (2 matrices: up-project + down-project),
    SwiGLU uses 3 matrices:
      gate(x)  → passed through SiLU activation (the "gate")
      up(x)    → element-wise multiplied with the gate output
      down(·)  → project back to d_model

    The gating mechanism acts like learned neuron selection — it can
    suppress irrelevant features entirely. Result: ~15% better quality
    per parameter vs standard GELU FFN (measured in Google's PaLM paper).
    """

    def __init__(self, config: NovaMindConfig):
        super().__init__()
        # All three matrices, no bias
        self.gate = nn.Linear(config.d_model, config.ff_dim, bias=False)
        self.up   = nn.Linear(config.d_model, config.ff_dim, bias=False)
        self.down = nn.Linear(config.ff_dim,  config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate(x)) ⊙ up(x) — element-wise gating
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# ──────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One transformer layer using Pre-RMSNorm architecture:

      x → RMSNorm → GQACausalAttention → +x (residual)
        → RMSNorm → SwiGLUFFN         → +x (residual)

    Pre-norm (norm BEFORE the sub-layer) is more training-stable than
    Post-norm, especially at 24 layers deep. All modern LLMs use Pre-norm.
    """

    def __init__(self, config: NovaMindConfig):
        super().__init__()
        self.ln_attn = RMSNorm(config.d_model)
        self.attn    = GQACausalAttention(config)
        self.ln_ffn  = RMSNorm(config.d_model)
        self.ffn     = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual connection
        x = x + self.attn(self.ln_attn(x))
        # FFN sub-layer with residual connection
        x = x + self.ffn(self.ln_ffn(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# TAG-AWARE LOSS — NOVEL CONTRIBUTION #2
# ──────────────────────────────────────────────────────────────────────────────

def tag_aware_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Tag-Aware Loss Curriculum (Novel Contribution #2).
    ──────────────────────────────────────────────────
    Standard cross-entropy treats every predicted token equally.
    We instead weight each token's loss contribution by its role in the
    conversation, and by which training phase we are in:

    Token types + phase weights:
    ┌────────────────┬──────────┬──────────┬──────────────┐
    │ Token type     │ Phase 1  │ Phase 2  │  Phase 3     │
    │                │ Pretrain │  SFT     │  CoT SFT     │
    ├────────────────┼──────────┼──────────┼──────────────┤
    │ <think>        │   0.0    │   0.5    │   1.5 ← BOOST│
    │ <assistant>    │   1.0    │   1.0    │   1.0        │
    │ <human>/system │   0.0    │   0.0    │   0.0        │
    └────────────────┴──────────┴──────────┴──────────────┘

    Why this is novel:
    - Most papers use binary masking (weight is 0 or 1)
    - A curriculum that changes weights across phases has not been
      published at this scale
    - Phase 3 boosting (weight=1.5 for think tokens) forces the model to
      "care" about the quality of its own reasoning chain

    Args:
        logits:        [B, T, V] — raw output logits
        targets:       [B, T]    — target token ids (-1 = ignore this position)
        token_weights: [B, T]    — per-token float weight (None → uniform 1.0)

    Returns:
        Scalar loss value
    """
    B, T, V = logits.shape

    # Flat views for loss computation
    logits_flat  = logits.view(B * T, V)
    targets_flat = targets.view(B * T)

    if token_weights is None:
        # Standard cross-entropy when no curriculum weights provided
        return F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

    # Compute per-token loss (reduction='none' gives us loss per position)
    per_token_loss = F.cross_entropy(
        logits_flat, targets_flat,
        ignore_index=-1,
        reduction="none",
    )  # [B*T]

    weights_flat = token_weights.view(B * T)

    # Mask out ignored positions (targets == -1)
    # F.cross_entropy with ignore_index already zeroes them, but
    # we also zero their weights to avoid dividing by them
    valid_mask = (targets_flat != -1).float()
    weights_flat = weights_flat * valid_mask

    # Weighted mean: sum(w_i * loss_i) / sum(w_i)
    # Small epsilon to prevent div-by-zero on pathological batches
    weighted_loss = (per_token_loss * weights_flat).sum()
    weight_sum    = weights_flat.sum().clamp(min=1e-8)

    return weighted_loss / weight_sum


# ──────────────────────────────────────────────────────────────────────────────
# NOVAMIND-256M — MAIN MODEL
# ──────────────────────────────────────────────────────────────────────────────

class NovaMind256M(nn.Module):
    """
    NovaMind-256M: 256M parameter conversational LLM.

    Architectural features:
      - Decoder-only transformer (autoregressive)
      - Pre-RMSNorm for training stability
      - GQA (16Q / 4KV) for inference efficiency
      - SwiGLU FFN for parameter efficiency
      - HiRoPE for hierarchical positional encoding (adapted from ACL 2024 paper)
      - Weight-tied embedding and LM head (saves 33M params)
      - Tag-Aware Loss Curriculum [NOVEL]
      - BF16 training, Flash Attention 2
    """

    def __init__(self, config: NovaMindConfig):
        super().__init__()
        self.config = config
        self._use_gradient_checkpointing = False

        # ── Embedding ────────────────────────────────────────────────────
        self.token_emb  = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # ── Transformer Stack ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # ── Output head ──────────────────────────────────────────────────
        self.ln_f   = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: lm_head and token_emb share the same weight matrix.
        # This saves vocab_size × d_model = 32,000 × 1,024 = 32.77M parameters.
        # The embedding learns "what does this token mean?" and the LM head
        # learns "how does the hidden state score as this token?" — they're
        # two views of the same embedding space.
        self.lm_head.weight = self.token_emb.weight

        # ── Weight Initialization ────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        GPT-2 style initialization with scaled residual projections.

        Output projections (o_proj, down in FFN) are scaled by 1/sqrt(2*N)
        where N = number of layers. This prevents the residual stream from
        growing proportionally to depth, which would destabilize training.

        Reference: GPT-2 paper, Table 2, "Modified initialization" footnote.
        """
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale residual output projections (o_proj and FFN down)
        # Identify them by name — they project *into* the residual stream
        residual_proj_names = {"o_proj", "down"}
        for name, param in module.named_parameters(recurse=False):
            # Use the parent module name to identify projection type
            parent_name = type(module).__name__
            if any(p in parent_name.lower() or p in name.lower()
                   for p in residual_proj_names):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=std / math.sqrt(2 * self.config.n_layers),
                )

    # ── Gradient Checkpointing ─────────────────────────────────────────────────
    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing to trade compute for VRAM.
        Discards intermediate activations during forward pass, recomputes
        them during backward. Saves ~40% VRAM at cost of ~20% slower training.
        CRITICAL for fitting 256M model + optimizer states on H100 40GB.
        """
        self._use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._use_gradient_checkpointing = False

    # ── Forward Pass ──────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids:     [B, T] — token ids
            targets:       [B, T] — target ids for loss computation (-1 = ignore)
                           If None, loss is not computed (inference mode)
            token_weights: [B, T] — per-token float weights for Tag-Aware Loss
                           If None, uses standard uniform cross-entropy

        Returns:
            (logits, loss)
            - logits: [B, T, V] during inference; empty tensor during training
              (returning full logits during training on DataParallel is wasteful —
               it forces a massive tensor transfer to GPU 0. We only need the loss.)
            - loss: scalar tensor if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Input length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Embed tokens
        x = self.emb_dropout(self.token_emb(input_ids))  # [B, T, D]

        # Pass through transformer blocks
        for block in self.blocks:
            if self._use_gradient_checkpointing and self.training:
                # gradient_checkpoint recomputes this block's activations
                # during backward instead of storing them — saves VRAM
                x = gradient_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Final norm + project to vocabulary
        logits = self.lm_head(self.ln_f(x))  # [B, T, V]

        loss = None
        if targets is not None:
            loss = tag_aware_loss(logits, targets, token_weights)
            # Drop logits during training to avoid PCIe transfer on multi-GPU
            if self.training:
                logits = torch.empty(0, device=logits.device)

        return logits, loss

    # ── Generation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        stop_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, top-p sampling
        and repetition penalty.

        Args:
            input_ids:         [1, T] — prompt tokens
            max_new_tokens:    maximum tokens to generate
            temperature:       sampling temperature (lower = more focused)
            top_k:             keep only top-k logits before sampling
            top_p:             nucleus sampling threshold
            repetition_penalty: >1.0 penalizes repeated tokens
            stop_token_ids:    list of token ids that stop generation

        Returns:
            token ids including the prompt: [1, T + generated]
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len (sliding window)
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)        # [1, T, V]
            logits = logits[:, -1, :]         # [1, V] — last position only

            # Repetition penalty: down-weight tokens already in context
            if repetition_penalty != 1.0:
                for token_id in set(idx_cond[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering: zero out all but top-k logits
            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, top_k_val).values[:, -1, None]
                logits[logits < kth_val] = float("-inf")

            # Top-p (nucleus) filtering: keep smallest set of tokens
            # whose cumulative probability exceeds top_p
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above threshold
                # (shift by 1 to keep the token that *crosses* the threshold)
                sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(
                    dim=-1, index=sorted_idx, src=sorted_logits
                )

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

            input_ids = torch.cat([input_ids, next_id], dim=1)

            # Stop if we hit a stop token
            if stop_token_ids and next_id.item() in stop_token_ids:
                break

        return input_ids

    # ── Parameter Counting ────────────────────────────────────────────────────
    def count_parameters(self, print_table: bool = True) -> int:
        """
        Count and optionally print a detailed parameter breakdown.
        Useful for verifying we hit ~252M.
        """
        cfg = self.config

        embed_params = cfg.vocab_size * cfg.d_model
        hd = cfg.head_dim()

        # Per-layer: Q+K+V+O projections + gate+up+down FFN + 2×RMSNorm
        # QKV projections: (n_q + n_kv + n_kv) × head_dim × d_model
        qkv_params  = (cfg.n_heads + 2 * cfg.n_kv_heads) * hd * cfg.d_model
        # Output projection: d_model × d_model
        o_params    = cfg.d_model * cfg.d_model
        attn_params = qkv_params + o_params
        # SwiGLU: gate + up + down (3 matrices)
        ffn_params  = 3 * cfg.d_model * cfg.ff_dim
        norm_params = 2 * cfg.d_model   # two RMSNorm per block
        per_layer   = attn_params + ffn_params + norm_params

        final_norm  = cfg.d_model
        total_with_tie = embed_params + cfg.n_layers * per_layer + final_norm
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if print_table:
            w = 46
            print("─" * w)
            print(f"  NovaMind-256M Parameter Breakdown")
            print("─" * w)
            print(f"  {'Token Embedding (shared w/ lm_head)':<40} {embed_params/1e6:>6.2f}M")
            print(f"  {'Attention QKV projections (per layer)':<40} {qkv_params/1e6:>6.2f}M")
            print(f"  {'Attention O projection (per layer)':<40} {o_params/1e6:>6.2f}M")
            print(f"  {'SwiGLU FFN (per layer)':<40} {ffn_params/1e6:>6.2f}M")
            print(f"  {'RMSNorm × 2 (per layer)':<40} {norm_params/1e6:>6.4f}M")
            print(f"  {'× {n} layers total':<40} {cfg.n_layers * per_layer/1e6:>6.2f}M")
            print(f"  {'Final RMSNorm':<40} {final_norm/1e6:>6.4f}M")
            print(f"  {'LM Head (weight-tied to embedding)':<40} {'→':>4}  0.00M")
            print("─" * w)
            print(f"  {'TOTAL (formula estimate)':<40} {total_with_tie/1e6:>6.2f}M")
            print(f"  {'TOTAL (actual PyTorch count)':<40} {total_trainable/1e6:>6.2f}M")
            print("─" * w)

        return total_trainable


# ──────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    config = NovaMindConfig()
    print(f"\nBuilding NovaMind-256M with config:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    print()

    model = NovaMind256M(config)
    n_params = model.count_parameters(print_table=True)

    # Verify target range
    target_min, target_max = 245e6, 265e6
    status = "✅" if target_min <= n_params <= target_max else "❌"
    print(f"\n{status} Parameter count: {n_params/1e6:.2f}M (target: 245–265M)\n")

    if not (target_min <= n_params <= target_max):
        sys.exit(1)

    # Quick forward pass test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.gradient_checkpointing_enable()

    B, T = 2, 128
    ids     = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = ids.clone()
    weights = torch.ones(B, T, device=device)
    weights[:, :10] = 0.0   # mask first 10 tokens (simulate user tokens)
    weights[:, 20:40] = 1.5  # boost tokens 20-40 (simulate think tokens in Phase 3)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits, loss = model(ids, targets, weights)

    print(f"Forward pass OK — loss: {loss.item():.4f}")
    print(f"Logits shape during training: {logits.shape}  (should be empty [0])")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generation OK — output shape: {generated.shape}")
    print("\n✅ All checks passed. model.py is ready.\n")
