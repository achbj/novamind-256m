#!/bin/bash
# ============================================================
#  Phase 3: CoT SFT (Reasoning Boost)
#  ~1.5 hours on H100
#  Run AFTER Phase 2 is complete.
#  Usage: bash run_phase3.sh
# ============================================================

set -e
CKPT_DIR="./checkpoints"
PHASE2_CKPT="$CKPT_DIR/novamind_sft_final.pt"
RESUME_FLAG="--resume $PHASE2_CKPT"

ROLL_CKPT="$CKPT_DIR/novamind_roll_latest.pt"
if [[ "$1" == "--resume" ]] && [ -f "$ROLL_CKPT" ]; then
    PHASE=$(python3 -c "import torch; c=torch.load('$ROLL_CKPT',map_location='cpu',weights_only=False); print(c.get('phase','?'))" 2>/dev/null)
    if [[ "$PHASE" == "cot_sft" ]]; then
        echo "📂 Resuming Phase 3 mid-session from: $ROLL_CKPT"
        RESUME_FLAG="--resume $ROLL_CKPT"
    fi
fi

echo ""
echo "============================================================"
echo "  🚀 Phase 3: CoT SFT — Reasoning Boost"
echo "  Dataset: OpenHermes + GSM8K (Capped)"
echo "  think-token loss weight: 1.5 (BOOSTED)"
echo "  Target: ~40M tokens, ~0.4 hours on H100"
echo "============================================================"
echo ""

python3 train.py \
    --phase cot_sft \
    --out_dir "$CKPT_DIR" \
    --batch_size 8 \
    --grad_accum_steps 8 \
    --max_seq_len 2048 \
    --lr 5e-5 \
    --warmup_steps 200 \
    --epochs 1 \
    --data_cap 50000 \
    --compile \
    --save_every 500 \
    --milestone_every 2000 \
    --val_every 500 \
    --val_batches 50 \
    --log_every 10 \
    --num_workers 4 \
    $RESUME_FLAG

echo ""
echo "✅ Training COMPLETE! Final model:"
echo "   ./checkpoints/novamind_cot_sft_final.pt"
