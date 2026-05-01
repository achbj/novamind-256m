"""
Phase 1b Data Preparation Script
=================================
Run this on the CPU-only free tier BEFORE starting your GPU session.
Downloads, tokenizes, and saves all training data to disk so that
the GPU session starts training immediately — zero data loading time.

Data sources:
  - TinyStories  (~2.1M stories, already seen in Phase 1 — good for annealing)
  - Wikipedia EN (~6.5M articles, NEVER seen by model — genuinely new data!)
    → Phase 1 only used FineWeb-Edu + TinyStories, so Wikipedia is 100% fresh.
    → Downloads as cached files (fast), NOT streaming.

Usage:
    python3 prepare_data_phase1b.py

Output:
    ./data/phase1b_chunks.pt   (~3-5 GB on disk)

Time on CPU free tier:
    TinyStories download:  ~3-5 min
    Wikipedia download:    ~20-40 min (~21 GB compressed)
    Tokenization:          ~20-40 min
    TOTAL:                 ~45-90 min  (no GPU cost!)

Then on GPU:
    bash run_phase1b.sh
    → Loads data from disk in ~30 sec, training starts immediately.
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
SEQ_LEN        = 2048
SAVE_PATH      = "./data/phase1b_chunks.pt"
META_PATH      = "./data/phase1b_chunks_meta.json"

# Data sources — both genuinely complement Phase 1 (FineWeb-Edu + TinyStories):
#   Wikipedia: 6.5M articles, NEVER seen in Phase 1 → fresh new knowledge
#   TinyStories: already seen, but repeating at lower LR = data annealing (valid!)

# ── Setup ─────────────────────────────────────────────────────────────────────
Path("./data").mkdir(exist_ok=True)

print("\n" + "="*60)
print("  Phase 1b Data Preparation")
print("="*60)
print(f"  Data sources:    Wikipedia EN (new!) + TinyStories (annealing)")
print(f"  Seq len:         {SEQ_LEN}")
print(f"  Save path:       {SAVE_PATH}")
print()

# ── Check if already done ──────────────────────────────────────────────────────
if Path(SAVE_PATH).exists():
    size_gb = Path(SAVE_PATH).stat().st_size / 1e9
    print(f"✅ Data file already exists: {SAVE_PATH} ({size_gb:.2f} GB)")
    if Path(META_PATH).exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        print(f"   Chunks: {meta['n_chunks']:,} | Tokens: {meta['tokens_B']:.2f}B")
    print("\n👉 Ready to train! Run:  bash run_phase1b.sh")
    sys.exit(0)

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print("🔤 Loading tokenizer...")
from transformers import AutoTokenizer

tokenizer = None
for tok_name in [
    "NousResearch/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-hf",
    "gpt2",
]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        print(f"   ✅ {tok_name} — vocab: {tokenizer.vocab_size:,}")
        break
    except Exception as e:
        print(f"   ⚠️  {tok_name} unavailable: {e}")

if tokenizer is None:
    raise RuntimeError("No tokenizer available. Run: huggingface-cli login")

eos = tokenizer.eos_token_id

# ── Token packing helper ───────────────────────────────────────────────────────
def pack_texts(texts, desc="packing") -> list:
    """Pack text list into [SEQ_LEN+1] overlapping token chunks."""
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

# ── 1. TinyStories (downloadable files, fast) ──────────────────────────────────
print("\n📚 Downloading TinyStories (~1 GB, fast download)...")
from datasets import load_dataset

t0 = time.time()
ts_ds = load_dataset("roneneldan/TinyStories", split="train")
print(f"   ✅ {len(ts_ds):,} stories downloaded in {(time.time()-t0)/60:.1f} min")

ts_chunks = pack_texts([r["text"] for r in ts_ds], "Tokenizing TinyStories")
print(f"   ✅ {len(ts_chunks):,} chunks ({len(ts_chunks)*SEQ_LEN/1e9:.2f}B tokens)")
all_chunks.extend(ts_chunks)

# ── 2. Wikipedia EN (downloadable, ~21GB, NEVER seen in Phase 1) ──────────────
print(f"\n📚 Downloading English Wikipedia (genuinely new data for model)...")
print("   Phase 1 only used FineWeb-Edu + TinyStories — Wikipedia is 100% fresh.")
print("   Downloading as cached files (not streaming) — much faster than FineWeb!")
print("   First download: ~20-40 min. Subsequent runs: instant from cache.\n")

t0 = time.time()
try:
    wiki_ds = load_dataset("wikipedia", "20220301.en", split="train",
                           trust_remote_code=True)
    elapsed_min = (time.time() - t0) / 60
    print(f"   ✅ {len(wiki_ds):,} articles downloaded in {elapsed_min:.1f} min")

    wiki_chunks = pack_texts([r["text"] for r in wiki_ds], "Tokenizing Wikipedia")
    print(f"   ✅ {len(wiki_chunks):,} chunks ({len(wiki_chunks)*SEQ_LEN/1e9:.2f}B tokens)")
    all_chunks.extend(wiki_chunks)
except Exception as e:
    print(f"   ⚠️  Wikipedia download failed: {e}")
    print("   Falling back to FineWeb-Edu streaming (500K docs)...")
    fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                         split="train", streaming=True)
    fw_texts = []
    for i, row in enumerate(tqdm(fw_ds, total=500_000, desc="  FineWeb fallback")):
        if i >= 500_000: break
        fw_texts.append(row["text"])
    fw_chunks = pack_texts(fw_texts, "Tokenizing FineWeb")
    print(f"   ✅ {len(fw_chunks):,} chunks")
    all_chunks.extend(fw_chunks)

# ── Save to disk ───────────────────────────────────────────────────────────────
print(f"\n💾 Saving {len(all_chunks):,} total chunks to disk...")
print("   (Saving as int16 to halve file size — auto-cast to int64 when loading)")

# int16 works because vocab_size=32000 < 32767
tensor = torch.tensor(all_chunks, dtype=torch.int16)
torch.save(tensor, SAVE_PATH)

size_gb = Path(SAVE_PATH).stat().st_size / 1e9
tokens_B = len(all_chunks) * SEQ_LEN / 1e9

# Save metadata
meta = {
    "n_chunks": len(all_chunks),
    "tokens_B": round(tokens_B, 3),
    "seq_len": SEQ_LEN,
    "sources": ["TinyStories (annealing)", "Wikipedia EN (new data)"],
    "dtype": "int16",
    "instructions": "Load with: torch.load(path).long()",
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print(f"  ✅ DATA PREP COMPLETE")
print(f"{'='*60}")
print(f"  File:   {SAVE_PATH}")
print(f"  Size:   {size_gb:.2f} GB")
print(f"  Chunks: {len(all_chunks):,}")
print(f"  Tokens: {tokens_B:.2f}B")
print()
print("  Now start your GPU session and run:")
print("    bash run_phase1b.sh")
print()
