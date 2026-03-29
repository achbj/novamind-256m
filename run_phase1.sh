#!/bin/bash
# ============================================================
#  Phase 1: Language Pretraining
#  ~14 hours total across ~15 Lightning.ai 1-hour sessions
#  Run: bash run_phase1.sh
#  Resume: bash run_phase1.sh --resume
# ============================================================

set -e
CKPT_DIR="./checkpoints"
RESUME_FLAG=""

# Check if --resume flag passed and auto-find latest rolling checkpoint
if [[ "$1" == "--resume" ]]; then
    LATEST="$CKPT_DIR/novamind_roll_latest.pt"
    if [ -f "$LATEST" ]; then
        echo "📂 Resuming from: $LATEST"
        RESUME_FLAG="--resume $LATEST"
    else
        echo "⚠️  No rolling checkpoint found — starting fresh"
    fi
fi

echo ""
echo "============================================================"
echo "  🚀 Phase 1: Language Pretraining"
echo "  Dataset: FineWeb-Edu + TinyStories (Capped)"
echo "  Target: ~1B tokens, ~3.8 hours on H100"
echo "============================================================"
echo ""

python3 train.py \
    --phase pretrain \
    --out_dir "$CKPT_DIR" \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --max_seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --epochs 1 \
    --data_cap 1500000 \
    --compile \
    --save_every 500 \
    --milestone_every 2000 \
    --val_every 500 \
    --val_batches 50 \
    --log_every 10 \
    --num_workers 4 \
    $RESUME_FLAG

echo ""
echo "✅ Phase 1 complete! Run Phase 2:"
echo "   bash run_phase2.sh"
