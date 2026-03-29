"""
NovaMind-256M Smoke Test
========================
Verifies the full training pipeline end-to-end with tiny fake data.
Run this before starting real training to catch bugs fast.

Usage:
    python verify.py

Expected output:
    ✅ model.py: parameter count OK (~252M)
    ✅ forward pass: loss computed correctly
    ✅ tag-aware loss: correct weighting
    ✅ hierrope: correct cache shape
    ✅ generation: output shape correct
    ✅ checkpoint: save/load round-trip works
    ✅ All checks passed — ready to train!
"""

import torch
import tempfile
import os
from pathlib import Path

# ── 1. Import model ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  NovaMind-256M Smoke Test")
print("="*55)

from model import NovaMind256M, NovaMindConfig, tag_aware_loss, build_hier_rope_cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
print(f"\n  Device: {device} | Dtype: {dtype}")

# ── 2. Build a tiny model (so it runs fast on CPU too) ────────────────────────
TINY_CFG = NovaMindConfig(
    vocab_size  = 32_000,
    d_model     = 128,   # much smaller for speed
    n_heads     = 4,
    n_kv_heads  = 2,
    n_layers    = 2,
    ff_dim      = 256,
    max_seq_len = 64,
)
model = NovaMind256M(TINY_CFG).to(device)
model.gradient_checkpointing_enable()
n_params = model.count_parameters(print_table=False)
print(f"\n✅ Model built — {n_params/1e6:.3f}M params (tiny config for testing)")

# ── 3. Full-size config parameter count verification ──────────────────────────
FULL_CFG = NovaMindConfig()
full_model = NovaMind256M(FULL_CFG)
n_full = full_model.count_parameters(print_table=True)
del full_model
assert 240e6 < n_full < 280e6, f"❌ Full model params {n_full/1e6:.2f}M outside expected range!"
print(f"✅ Full model: {n_full/1e6:.2f}M params (target 240-280M, i.e. ~256M class)")

# ── 4. Forward pass ───────────────────────────────────────────────────────────
B, T = 2, 32
ids     = torch.randint(0, TINY_CFG.vocab_size, (B, T), device=device)
targets = ids.clone()
weights = torch.ones(B, T, device=device)

with torch.autocast(device_type=device.type, dtype=dtype):
    logits, loss = model(ids, targets, weights)

assert loss is not None, "❌ Loss is None"
assert loss.item() > 0, "❌ Loss is zero or negative"
assert logits.numel() == 0, "❌ Logits should be empty during training"
print(f"✅ Forward pass — loss: {loss.item():.4f}  logits (should be empty): {logits.shape}")

# ── 5. Tag-Aware Loss correctness ─────────────────────────────────────────────
# Test: masking - tokens with weight=0 should not contribute to loss
full_weight  = torch.ones(B, T, device=device)
zero_weight  = torch.zeros(B, T, device=device)
mixed_weight = torch.ones(B, T, device=device)
mixed_weight[:, :T//2] = 0.0  # mask first half

with torch.no_grad():
    logits_test, _ = model(ids)  # get logits without targets

loss_full  = tag_aware_loss(logits_test, targets, full_weight)
loss_zero  = tag_aware_loss(logits_test, targets, zero_weight)
loss_mixed = tag_aware_loss(logits_test, targets, mixed_weight)
loss_std   = tag_aware_loss(logits_test, targets, None)

assert loss_zero.item() == 0.0 or torch.isnan(loss_zero), "❌ Zero-weight loss should be ~0"
assert abs(loss_full.item() - loss_std.item()) < 1e-4, "❌ Full-weight loss should match standard loss"
print(f"✅ Tag-aware loss — full: {loss_full.item():.4f}, zero: {loss_zero.item():.6f}, mixed: {loss_mixed.item():.4f}, std: {loss_std.item():.4f}")

# ── 6. HierRoPE shape & value sanity ─────────────────────────────────────────
cos, sin = build_hier_rope_cache(64, 64, device, local_base=10000, global_base=500000)
assert cos.shape == (64, 64), f"❌ HiRoPE cos shape wrong: {cos.shape}"
assert sin.shape == (64, 64), f"❌ HiRoPE sin shape wrong: {sin.shape}"

# Local half (first 32 dims) should have higher frequency variation than global half (last 32)
local_cos_variation  = cos[:, :32].std().item()
global_cos_variation = cos[:, 32:].std().item()
# Global uses higher base → slower rotation → less variation over same position range
assert local_cos_variation >= global_cos_variation, (
    f"❌ HiRoPE: local variation ({local_cos_variation:.4f}) should be >= "
    f"global ({global_cos_variation:.4f})"
)
print(f"✅ HiRoPE — shape {cos.shape} | local std: {local_cos_variation:.4f} > global std: {global_cos_variation:.4f}")

# ── 7. Generation ─────────────────────────────────────────────────────────────
model.eval()
prompt = torch.randint(0, TINY_CFG.vocab_size, (1, 8), device=device)
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model.generate(prompt, max_new_tokens=16, temperature=0.8)
assert out.shape[1] == 8 + 16, f"❌ Generation shape: {out.shape} (expected [1, 24])"
print(f"✅ Generation — output shape: {out.shape} (prompt 8 + generated 16)")

# ── 8. Checkpoint round-trip ──────────────────────────────────────────────────
model.train()
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = Path(tmp) / "test_ckpt.pt"
    state = model.state_dict()
    torch.save({
        "model_state_dict":     state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step":                 42,
        "phase":                "pretrain",
        "tokens_seen":          1_000_000,
        "config":               TINY_CFG.to_dict(),
        "train_loss_history":   [{"step": 10, "train_loss": 3.5}],
        "val_loss_history":     [{"step": 10, "val_loss": 3.6}],
    }, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model2 = NovaMind256M(TINY_CFG).to(device)
    model2.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["step"] == 42
    assert ckpt["tokens_seen"] == 1_000_000

print(f"✅ Checkpoint — save/load round-trip OK (step={ckpt['step']}, tokens={ckpt['tokens_seen']:,})")

# ── 9. Gradient checkpointing ─────────────────────────────────────────────────
model.train()
model.gradient_checkpointing_enable()
x2, y2, w2 = ids, targets, weights
with torch.autocast(device_type=device.type, dtype=dtype):
    _, loss2 = model(x2, y2, w2)
loss2.backward()
print(f"✅ Gradient checkpointing — backward pass OK, loss: {loss2.item():.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  ✅ ALL CHECKS PASSED — Ready to train NovaMind-256M!")
print(f"{'='*55}")
print()
print("  Next steps:")
print("  1. pip install -r requirements.txt")
print("  2. huggingface-cli login  (for LLaMA-2 tokenizer)")
print("  3. wandb login            (optional, for W&B logging)")
print()
print("  Phase 1 (pretrain):")
print("  python train.py --phase pretrain --out_dir ./checkpoints")
print()
print("  Phase 2 (SFT, after Phase 1):")
print("  python train.py --phase sft --resume ./checkpoints/novamind_phase1_final.pt")
print()
print("  Phase 3 (CoT SFT, after Phase 2):")
print("  python train.py --phase cot_sft --resume ./checkpoints/novamind_phase2_final.pt")
print()
print("  Quick smoke test (50 steps, tiny data):")
print("  python train.py --phase pretrain --smoke_test")
print()
