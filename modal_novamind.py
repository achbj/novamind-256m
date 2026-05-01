"""
NovaMind Modal Training Pipeline
==================================
Step 1 (CPU, cheap): Download & tokenize Phase 1b data
Step 2 (CPU, cheap): Download & tokenize Phase 2 SFT data
Step 3 (H100):       Continue pretraining to ~5B tokens
Step 4 (H100):       SFT fine-tuning

Setup (run once):
    conda activate modal
    modal volume create novamind-vol
    modal volume put novamind-vol ./checkpoints/novamind_pretrain_final.pt /checkpoints/novamind_pretrain_final.pt

Run order:
    modal run modal_novamind.py::prepare_data_phase1b   # CPU, ~60-90 min
    modal run modal_novamind.py::prepare_data_phase2    # CPU, ~20-40 min
    modal run modal_novamind.py::train_phase1b          # H100, ~3-5 hrs
    modal run modal_novamind.py::train_phase2           # H100, ~2-3 hrs

Download result:
    modal volume get novamind-vol /checkpoints/novamind_sft_final.pt ./checkpoints/novamind_sft_final.pt
"""

import subprocess
import sys
from pathlib import Path
import modal

# ── Volume ─────────────────────────────────────────────────────────────────────
volume = modal.Volume.from_name("novamind-vol", create_if_missing=True)
VOL = "/vol"

# ── Image — pip deps + project source files baked in ─────────────────────────
# modal.Mount was removed in Modal 1.x; use image.add_local_dir() instead.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.2.0",
        "torchvision",
        "transformers>=4.38.0",
        "datasets>=2.18.0",
        "huggingface_hub>=0.21.0",
        "tqdm",
        "numpy",
        "wandb",
    ])
    # Only copy the Python files we actually need — avoids .git/FETCH_HEAD error
    .add_local_file("model.py",                  "/workspace/model.py")
    .add_local_file("train.py",                  "/workspace/train.py")
    .add_local_file("prepare_data_phase2.py",    "/workspace/prepare_data_phase2.py")
)

# ── App ────────────────────────────────────────────────────────────────────────
app = modal.App("novamind-training", image=image)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CPU: Phase 1b Data Prep (Wikipedia + TinyStories)
# ══════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=4,
    memory=16384,        # 16 GB RAM — Wikipedia tokenization is memory-heavy
    timeout=60 * 60 * 6, # 6 hours max
)
def prepare_data_phase1b():
    """CPU-only: Download & tokenize Phase 1b data (Wikipedia EN + TinyStories).
    Saves to /vol/data/phase1b_chunks.pt — no GPU cost."""
    import os, time, json, torch
    from pathlib import Path
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from datasets import load_dataset

    SAVE_PATH = f"{VOL}/data/phase1b_chunks.pt"
    META_PATH = f"{VOL}/data/phase1b_chunks_meta.json"
    SEQ_LEN   = 2048

    Path(f"{VOL}/data").mkdir(parents=True, exist_ok=True)

    if Path(SAVE_PATH).exists():
        size_gb = Path(SAVE_PATH).stat().st_size / 1e9
        print(f"✅ Already exists: {SAVE_PATH} ({size_gb:.2f} GB) — skipping download.")
        return

    print("\n" + "="*60)
    print("  Phase 1b Data Prep — Wikipedia EN + TinyStories")
    print("="*60)

    # Tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = None
    for name in ["NousResearch/Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf", "gpt2"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            print(f"   ✅ {name}")
            break
        except Exception as e:
            print(f"   ⚠️  {name}: {e}")
    if tokenizer is None:
        raise RuntimeError("No tokenizer available.")
    eos = tokenizer.eos_token_id

    def pack_texts(texts, desc="packing"):
        chunks, current = [], []
        for text in tqdm(texts, desc=f"  {desc}", leave=False):
            ids = tokenizer.encode(text, add_special_tokens=False) + [eos]
            if len(ids) >= SEQ_LEN + 1:
                if current:
                    chunks.append(current + [eos] * ((SEQ_LEN+1) - len(current)))
                    current = []
                for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
                    chunks.append(ids[i: i+SEQ_LEN+1])
            else:
                if len(current) + len(ids) > SEQ_LEN + 1:
                    chunks.append(current + [eos] * ((SEQ_LEN+1) - len(current)))
                    current = []
                current.extend(ids)
        if len(current) >= 2:
            chunks.append(current + [eos] * ((SEQ_LEN+1) - len(current)))
        return chunks

    all_chunks = []

    # TinyStories
    print("\n📚 Downloading TinyStories...")
    t0 = time.time()
    ts_ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"   ✅ {len(ts_ds):,} stories  [{(time.time()-t0)/60:.1f} min]")
    ts_chunks = pack_texts([r["text"] for r in ts_ds], "Tokenizing TinyStories")
    print(f"   ✅ {len(ts_chunks):,} chunks ({len(ts_chunks)*SEQ_LEN/1e9:.2f}B tokens)")
    all_chunks.extend(ts_chunks)

    # Wikipedia EN
    print("\n📚 Downloading English Wikipedia (first download: ~20-40 min)...")
    t0 = time.time()
    try:
        wiki_ds = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
        print(f"   ✅ {len(wiki_ds):,} articles  [{(time.time()-t0)/60:.1f} min]")
        wiki_chunks = pack_texts([r["text"] for r in wiki_ds], "Tokenizing Wikipedia")
        print(f"   ✅ {len(wiki_chunks):,} chunks ({len(wiki_chunks)*SEQ_LEN/1e9:.2f}B tokens)")
        all_chunks.extend(wiki_chunks)
    except Exception as e:
        print(f"   ⚠️  Wikipedia failed: {e}")
        print("   Fallback: FineWeb-Edu streaming (500K docs)...")
        fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        fw_texts = []
        for i, row in enumerate(tqdm(fw_ds, total=500_000, desc="  FineWeb fallback")):
            if i >= 500_000: break
            fw_texts.append(row["text"])
        fw_chunks = pack_texts(fw_texts, "Tokenizing FineWeb")
        print(f"   ✅ {len(fw_chunks):,} chunks")
        all_chunks.extend(fw_chunks)

    # Save
    print(f"\n💾 Saving {len(all_chunks):,} chunks to {SAVE_PATH}...")
    tensor = torch.tensor(all_chunks, dtype=torch.int16)
    torch.save(tensor, SAVE_PATH)
    tokens_B = len(all_chunks) * SEQ_LEN / 1e9
    size_gb  = Path(SAVE_PATH).stat().st_size / 1e9

    meta = {
        "n_chunks": len(all_chunks), "tokens_B": round(tokens_B, 3),
        "seq_len": SEQ_LEN, "dtype": "int16",
        "sources": ["TinyStories (annealing)", "Wikipedia EN (new data)"],
    }
    with open(META_PATH, "w") as f:
        import json; json.dump(meta, f, indent=2)

    volume.commit()
    print(f"\n✅ Phase 1b data ready: {tokens_B:.2f}B tokens  |  {size_gb:.2f} GB")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CPU: Phase 2 SFT Data Prep
# ══════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 3,
)
def prepare_data_phase2():
    """CPU-only: Download & tokenize Phase 2 SFT data.
    Includes 500x identity seeds, Alpaca, OASST1, Dolly, DailyDialog.
    Saves to /vol/data/phase2_sft_chunks.pt"""
    import os, sys, time, json, torch
    from pathlib import Path

    sys.path.insert(0, "/workspace")

    SAVE_PATH = f"{VOL}/data/phase2_sft_chunks.pt"
    META_PATH = f"{VOL}/data/phase2_sft_chunks_meta.json"

    Path(f"{VOL}/data").mkdir(parents=True, exist_ok=True)

    if Path(SAVE_PATH).exists():
        size_gb = Path(SAVE_PATH).stat().st_size / 1e9
        print(f"✅ Already exists: {SAVE_PATH} ({size_gb:.2f} GB) — skipping.")
        return

    # Run the standalone prepare_data_phase2.py but redirect output to volume path
    # We patch the save paths then import and run
    import importlib.util, types

    # Read and patch save paths before executing
    script = Path("/workspace/prepare_data_phase2.py").read_text()
    script = script.replace(
        'SAVE_PATH = "./data/phase2_sft_chunks.pt"',
        f'SAVE_PATH = "{SAVE_PATH}"',
    ).replace(
        'META_PATH = "./data/phase2_sft_chunks_meta.json"',
        f'META_PATH = "{META_PATH}"',
    ).replace(
        'Path("./data").mkdir(exist_ok=True)',
        f'Path("{VOL}/data").mkdir(parents=True, exist_ok=True)',
    )

    exec(compile(script, "prepare_data_phase2.py", "exec"), {"__name__": "__main__"})
    volume.commit()
    print(f"\n✅ Phase 2 SFT data saved to {SAVE_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — H100: Continue Pretraining (Phase 1b)
# ══════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    volumes={VOL: volume},
    gpu="H100",
    timeout=60 * 60 * 20,   # 20 hours max
    memory=65536,
)
def train_phase1b():
    """H100: Continue pretraining from novamind_pretrain_final.pt using Phase 1b data.
    Annealing LR (1e-4) on Wikipedia EN + TinyStories — targets ~5B total tokens.

    IMPORTANT — step counter fix:
      train.py restores global_step from the checkpoint when the phase matches.
      Phase 1 may have run for e.g. 40,000 steps, but the new Phase 1b dataset
      is smaller (different data), so max_steps could be less → loop exits instantly.
      Fix: we create a 'warm-start' copy of the checkpoint with phase='pretrain_phase1'
      (non-matching). train.py then enters the phase-transition branch which:
        ✅ loads model weights (we get all Phase 1 knowledge)
        ✅ loads optimizer state (warm momentum vectors)
        ✅ resets global_step = 0  (fresh run over new data — no skipping)
        ✅ preserves tokens_seen   (cumulative token count stays accurate)
    """
    import os, subprocess, sys, torch
    from pathlib import Path

    data_file     = f"{VOL}/data/phase1b_chunks.pt"
    original_ckpt = f"{VOL}/checkpoints/novamind_pretrain_final.pt"
    warm_ckpt     = f"{VOL}/checkpoints/novamind_phase1b_warmstart.pt"
    out_dir       = f"{VOL}/checkpoints"

    # Validate inputs
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Phase 1b data not found: {data_file}\n"
                                 "Run prepare_data_phase1b first!")
    if not Path(original_ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {original_ckpt}\n"
                                 "Upload with: modal volume put novamind-vol "
                                 "./checkpoints/novamind_pretrain_final.pt "
                                 "/checkpoints/novamind_pretrain_final.pt")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Create warm-start checkpoint (resets step counter, keeps weights) ──────
    if not Path(warm_ckpt).exists():
        print("🔧 Creating Phase 1b warm-start checkpoint...")
        ckpt = torch.load(original_ckpt, map_location="cpu", weights_only=False)
        original_step   = ckpt.get("step", 0)
        original_tokens = ckpt.get("tokens_seen", 0)
        print(f"   Phase 1 checkpoint: step={original_step:,} | "
              f"tokens={original_tokens/1e9:.2f}B | phase='{ckpt.get('phase')}'")

        # Change phase name so train.py enters phase-transition branch → resets step=0
        ckpt["phase"] = "pretrain_phase1"  # non-matching with "--phase pretrain"
        torch.save(ckpt, warm_ckpt)
        volume.commit()
        print(f"   ✅ Warm-start checkpoint saved: {warm_ckpt}")
        print(f"   ✅ global_step will reset to 0, tokens_seen preserved ({original_tokens/1e9:.2f}B)")
    else:
        print(f"✅ Warm-start checkpoint already exists: {warm_ckpt}")

    cmd = [
        "python", "/workspace/train.py",
        "--phase",           "pretrain",
        "--data_file",       data_file,
        "--resume",          warm_ckpt,    # ← uses warm-start, not original
        "--out_dir",         out_dir,
        "--batch_size",      "8",
        "--grad_accum_steps","4",
        "--lr",              "1e-4",       # annealing LR — lower than original 3e-4
        "--warmup_steps",    "500",
        "--epochs",          "1",          # 1 full pass over new Wikipedia+TinyStories
        "--max_seq_len",     "2048",
        "--save_every",      "500",
        "--milestone_every", "2000",
        "--val_every",       "500",
        "--val_batches",     "50",
        "--log_every",       "10",
        "--num_workers",     "4",
        "--compile",
        "--no_wandb",
    ]

    print("\n" + "="*65)
    print("  🚀 Phase 1b: Continued Pretraining on H100")
    print(f"  Data:     {data_file}")
    print(f"  Resume:   {warm_ckpt} (step reset to 0, weights from Phase 1)")
    print(f"  Out:      {out_dir}")
    print(f"  LR:       1e-4 (annealing — lower than Phase 1's 3e-4)")
    print("="*65)
    print("  CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd="/workspace")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print("\n✅ Phase 1b pretraining complete!")
    print(f"   Checkpoint: {out_dir}/novamind_pretrain_final.pt")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — H100: SFT Fine-tuning (Phase 2)
# ══════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    volumes={VOL: volume},
    gpu="H100",
    timeout=60 * 60 * 10,
    memory=65536,
)
def train_phase2():
    """H100: SFT fine-tuning from Phase 1b pretrained checkpoint.
    Uses pre-tokenized SFT data with 500x identity seeds and stop-token fixes."""
    import os, subprocess
    from pathlib import Path

    data_file  = f"{VOL}/data/phase2_sft_chunks.pt"
    resume     = f"{VOL}/checkpoints/novamind_pretrain_final.pt"
    out_dir    = f"{VOL}/checkpoints"

    if not Path(data_file).exists():
        raise FileNotFoundError(f"Phase 2 SFT data not found: {data_file}\n"
                                 "Run prepare_data_phase2 first!")
    if not Path(resume).exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume}\n"
                                 "Run train_phase1b first!")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "/workspace/train.py",
        "--phase",           "sft",
        "--data_file",       data_file,
        "--resume",          resume,
        "--out_dir",         out_dir,
        "--batch_size",      "16",
        "--grad_accum_steps","4",
        "--lr",              "1e-4",
        "--warmup_steps",    "500",
        "--epochs",          "2",
        "--max_seq_len",     "2048",
        "--save_every",      "500",
        "--milestone_every", "2000",
        "--val_every",       "500",
        "--val_batches",     "50",
        "--log_every",       "10",
        "--num_workers",     "4",
        "--compile",
        "--no_wandb",
    ]

    print("="*65)
    print("  🚀 Phase 2: SFT Fine-tuning on H100")
    print(f"  Data:     {data_file}")
    print(f"  Resume:   {resume}")
    print(f"  Out:      {out_dir}")
    print("  Fixes:    500x identity seeds | 2x EOS weight | pre-tokenized")
    print("="*65)
    print("  CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd="/workspace")
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"SFT training failed with exit code {result.returncode}")

    print("\n✅ Phase 2 SFT complete!")
    print(f"   Final checkpoint: {out_dir}/novamind_sft_final.pt")
    print()
    print("  Download with:")
    print("    modal volume get novamind-vol /checkpoints/novamind_sft_final.pt ./checkpoints/novamind_sft_final.pt")
