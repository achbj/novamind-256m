#!/bin/bash
# ============================================================
#  Phase 2: Instruction SFT
#  ~2.5 hours on H100
#  Run AFTER Phase 1 is complete.
#  Usage: bash run_phase2.sh
# ============================================================

set -e
CKPT_DIR="./checkpoints"
PHASE1_CKPT="$CKPT_DIR/novamind_pretrain_final.pt"
RESUME_FLAG="--resume $PHASE1_CKPT"

# If a rolling checkpoint from Phase 2 mid-session exists, prefer that
ROLL_CKPT="$CKPT_DIR/novamind_roll_latest.pt"
if [[ "$1" == "--resume" ]] && [ -f "$ROLL_CKPT" ]; then
    # Check if the rolling ckpt is from phase sft
    PHASE=$(python3 -c "import torch; c=torch.load('$ROLL_CKPT',map_location='cpu',weights_only=False); print(c.get('phase','?'))" 2>/dev/null)
    if [[ "$PHASE" == "sft" ]]; then
        echo "📂 Resuming Phase 2 mid-session from: $ROLL_CKPT"
        RESUME_FLAG="--resume $ROLL_CKPT"
    fi
fi

echo ""
echo "============================================================"
echo "  🚀 Phase 2: Instruction SFT"
echo "  Dataset: Alpaca + DailyDialog (Capped)"
echo "  Target: ~60M tokens, ~0.8 hours on H100"
echo "============================================================"
echo ""

python3 train.py \
    --phase sft \
    --out_dir "$CKPT_DIR" \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --max_seq_len 2048 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --epochs 1 \
    --data_cap 100000 \
    --compile \
    --save_every 500 \
    --milestone_every 2000 \
    --val_every 500 \
    --val_batches 50 \
    --log_every 10 \
    --num_workers 4 \
    $RESUME_FLAG

echo ""
echo "✅ Phase 2 complete! Run Phase 3:"
echo "   bash run_phase3.sh"
