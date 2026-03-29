#!/bin/bash
# ============================================================
#  NovaMind-256M — Lightning.ai Setup Script
#  Run this ONCE at the start of every new Lightning.ai session
#  before starting or resuming training.
#
#  Usage:  bash setup_lightning.sh
# ============================================================

set -e  # exit immediately on error
echo ""
echo "============================================================"
echo "  NovaMind-256M — Lightning.ai Environment Setup"
echo "============================================================"
echo ""

# ── 1. System info ────────────────────────────────────────────
echo "🖥️  System info:"
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (no GPU detected — CPU mode)"
echo ""

# ── 2. Install Python dependencies ───────────────────────────
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q \
    "torch>=2.2.0" \
    "transformers>=4.38.0" \
    "datasets>=2.18.0" \
    "huggingface_hub>=0.21.0" \
    "wandb>=0.16.0" \
    "tqdm" \
    "numpy"

echo "   ✅ Dependencies installed"
echo ""

# ── 3. Verify torch + CUDA + BF16 ────────────────────────────
echo "🔥 Verifying PyTorch + CUDA:"
python3 -c "
import torch
print(f'   torch:  {torch.__version__}')
print(f'   CUDA:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'   GPU:    {props.name}')
    print(f'   VRAM:   {props.total_memory/1e9:.1f} GB')
    print(f'   BF16:   {torch.cuda.is_bf16_supported()}')
    print(f'   FA2:    Flash Attention 2 via F.scaled_dot_product_attention = available')
"
echo ""

# ── 4. Quick model sanity check ───────────────────────────────
echo "🧪 Running model verification..."
python3 verify.py
echo ""

# ── 5. HuggingFace login reminder ────────────────────────────
echo "============================================================"
echo "  SETUP COMPLETE ✅"
echo "============================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  [OPTIONAL] Login to HuggingFace for LLaMA-2 tokenizer:"
echo "    huggingface-cli login"
echo "    (if skipped, GPT-2 tokenizer 50K will be used as fallback)"
echo ""
echo "  [OPTIONAL] Login to W&B for cloud loss tracking:"
echo "    wandb login"
echo ""
echo "  Start Phase 1 training:"
echo "    bash run_phase1.sh"
echo ""
echo "  Or manually:"
echo "    python3 train.py --phase pretrain --out_dir ./checkpoints"
echo ""
