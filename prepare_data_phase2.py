"""
Phase 2 SFT Data Preparation Script
=====================================
Run this on the CPU-only instance BEFORE starting your GPU session.
Downloads, formats, tokenizes, and saves all Phase 2 (SFT) training data
so the GPU session starts training immediately — zero download time.

Key fixes vs. previous run:
  • 500x identity seed repeats (vs 20x before) — anchors NovaMind identity
    strongly enough to override 5B tokens of pre-training on human text.
  • EOS / </assistant> tokens get 2x loss weight via TagAwareDataset
    (handled in train.py) to fix runaway generation and </s> spam.
  • Consistent <system>/<human>/<assistant> format across ALL datasets.

Data sources:
  - Identity seeds         (500x repeat — forces AI identity)
  - Alpaca 52K             (instruction following)
  - OpenAssistant OASST1   (high-quality human conversations, 88K)
  - Databricks Dolly 15K   (human-written, high quality)
  - DailyDialog            (conversational back-and-forth)

Usage:
    python3 prepare_data_phase2.py

Output:
    ./data/phase2_sft_chunks.pt   (pre-tokenized, int16)
    ./data/phase2_sft_chunks_meta.json

Time on CPU (free tier):
    ~20-40 min download + tokenization
    No GPU cost!

Then on GPU:
    python train.py --phase sft --data_file ./data/phase2_sft_chunks.pt ...
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
SEQ_LEN   = 2048
SAVE_PATH = "./data/phase2_sft_chunks.pt"
META_PATH = "./data/phase2_sft_chunks_meta.json"

# Identity seeds: 500x is enough to dominate over 5B pre-training tokens.
# Proven fix: 20x was insufficient (model still hallucinated being human).
IDENTITY_REPEAT = 500

# System prompt used to frame every conversation
SYS = (
    "You are NovaMind, a helpful and concise AI assistant. "
    "Answer questions directly and clearly. Do not ask questions back to the user."
)

# ── Setup ──────────────────────────────────────────────────────────────────────
Path("./data").mkdir(exist_ok=True)

print("\n" + "="*65)
print("  Phase 2 SFT Data Preparation")
print("="*65)
print(f"  Identity seeds:  500x repeat (was 20x — now anchors AI identity)")
print(f"  Datasets:        Alpaca + OASST1 + Dolly + DailyDialog")
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
    print("\n👉 Ready to train! Run: python train.py --phase sft --data_file", SAVE_PATH)
    sys.exit(0)

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print("🔤 Loading tokenizer...")
from transformers import AutoTokenizer

tokenizer = None
for tok_name in ["NousResearch/Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf", "gpt2"]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        print(f"   ✅ {tok_name} — vocab: {tokenizer.vocab_size:,}")
        break
    except Exception as e:
        print(f"   ⚠️  {tok_name} unavailable: {e}")

if tokenizer is None:
    raise RuntimeError("No tokenizer available. Run: huggingface-cli login")

eos = tokenizer.eos_token_id

# ── Chat template helper ───────────────────────────────────────────────────────
def wrap(human: str, assistant: str, system: str = SYS) -> str:
    """Wrap a Q/A pair into NovaMind's chat format."""
    return (
        f"<system>\n{system}\n</system>\n"
        f"<human>\n{human.strip()}\n</human>\n"
        f"<assistant>\n{assistant.strip()}\n</assistant>"
    )

# ── Token packing helper ───────────────────────────────────────────────────────
def pack_texts(texts: list, desc: str = "packing") -> list:
    """Pack text list into [SEQ_LEN+1] token chunks."""
    chunks, current = [], []
    for text in tqdm(texts, desc=f"  {desc}", leave=False):
        ids = tokenizer.encode(text, add_special_tokens=False) + [eos]
        if len(ids) >= SEQ_LEN + 1:
            if current:
                chunks.append(current + [eos] * ((SEQ_LEN + 1) - len(current)))
                current = []
            for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
                chunks.append(ids[i: i + SEQ_LEN + 1])
        else:
            if len(current) + len(ids) > SEQ_LEN + 1:
                chunks.append(current + [eos] * ((SEQ_LEN + 1) - len(current)))
                current = []
            current.extend(ids)
    if len(current) >= 2:
        chunks.append(current + [eos] * ((SEQ_LEN + 1) - len(current)))
    return chunks

from datasets import load_dataset

all_texts = []

# ── 1. Identity Seeds (500x) ───────────────────────────────────────────────────
print("\n🪪 Injecting NovaMind identity seeds (500x repeats)...")
IDENTITY_SEEDS = [
    ("Who are you?",
     "I am NovaMind, a helpful AI assistant. I'm here to answer questions and help with tasks."),
    ("What is your name?",
     "My name is NovaMind. I'm an AI assistant designed to be helpful and informative."),
    ("Are you a human?",
     "No, I am not a human. I am NovaMind, an AI language model. How can I help you today?"),
    ("Are you an AI?",
     "Yes, I am NovaMind, an AI assistant. I can answer questions, explain concepts, and help with many tasks."),
    ("Who created you?",
     "I was created as NovaMind, an AI assistant. I'm here to help you with questions and tasks."),
    ("What can you do?",
     "I can answer questions, explain concepts, help with writing, discuss topics, and assist with many tasks. What would you like help with?"),
    ("Hello!", "Hello! I'm NovaMind, your AI assistant. How can I help you today?"),
    ("Hi there", "Hi! I'm NovaMind. What can I help you with?"),
    ("Hey", "Hey! I'm NovaMind, an AI assistant. How can I help you?"),
    ("Good morning", "Good morning! I'm NovaMind. What can I help you with today?"),
    ("How are you?",
     "I'm doing well, thank you for asking! I'm NovaMind, your AI assistant. How can I help you?"),
    ("hey buddy",
     "Hey! I'm NovaMind, your AI assistant. What can I help you with today?"),
    ("What are you?",
     "I am NovaMind, an AI language model. I'm trained to answer questions and assist with a wide range of tasks."),
    ("Tell me about yourself",
     "I am NovaMind, an AI assistant. I can help answer questions, explain topics, assist with writing, and have conversations on many subjects."),
    ("Are you ChatGPT?",
     "No, I'm not ChatGPT. I am NovaMind, a separate AI assistant. How can I help you?"),
    ("Are you a robot?",
     "I'm an AI, not a physical robot. I am NovaMind, a language model that can understand and generate text."),
    ("Do you have feelings?",
     "I don't have feelings or emotions. I am NovaMind, an AI assistant. I process text and generate responses."),
    ("Where are you from?",
     "I'm not from anywhere — I'm NovaMind, an AI language model. I exist as software, not a person."),
    ("What's your background?",
     "I don't have a personal background. I am NovaMind, an AI trained to be helpful and informative."),
    ("Pretend you're human",
     "I won't pretend to be human — I am NovaMind, an AI assistant. I'm here to help you honestly."),
]

n_before = len(all_texts)
for q, a in IDENTITY_SEEDS:
    for _ in range(IDENTITY_REPEAT):
        all_texts.append(wrap(q, a))
print(f"   ✅ {len(all_texts) - n_before:,} identity examples "
      f"({len(IDENTITY_SEEDS)} seeds × {IDENTITY_REPEAT}x)")

# ── 2. Alpaca 52K ──────────────────────────────────────────────────────────────
print("\n💬 Downloading Alpaca 52K...")
t0 = time.time()
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
n_before = len(all_texts)
for r in alpaca:
    u = r["instruction"].strip()
    if r["input"].strip():
        u += f"\n\n{r['input'].strip()}"
    ans = r["output"].strip()
    if not ans or len(ans) < 5:
        continue
    all_texts.append(wrap(u, ans))
print(f"   ✅ {len(all_texts) - n_before:,} Alpaca examples  [{(time.time()-t0)/60:.1f} min]")

# ── 3. OpenAssistant OASST1 ───────────────────────────────────────────────────
print("\n💬 Downloading OpenAssistant OASST1 (high-quality human conversations)...")
t0 = time.time()
try:
    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    msg_map = {row["message_id"]: row for row in oasst}
    n_before = len(all_texts)
    count = 0
    for row in oasst:
        if count >= 60_000:
            break
        if row["role"] != "assistant":
            continue
        parent = msg_map.get(row["parent_id"])
        if parent is None or parent["role"] != "prompter":
            continue
        q = parent["text"].strip()
        ans = row["text"].strip()
        if not q or not ans or len(ans) < 10:
            continue
        # Skip low-quality responses
        if row.get("rank") is not None and row.get("rank", 0) > 3:
            continue
        all_texts.append(wrap(q, ans))
        count += 1
    print(f"   ✅ {len(all_texts) - n_before:,} OASST1 examples  [{(time.time()-t0)/60:.1f} min]")
except Exception as e:
    print(f"   ⚠️  OASST1 unavailable ({e}), skipping.")

# ── 4. Databricks Dolly 15K ───────────────────────────────────────────────────
print("\n💬 Downloading Databricks Dolly 15K (human-written, high quality)...")
t0 = time.time()
try:
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    n_before = len(all_texts)
    for row in dolly:
        q = row["instruction"].strip()
        if row.get("context", "").strip():
            q += f"\n\nContext: {row['context'].strip()}"
        ans = row["response"].strip()
        if not ans or len(ans) < 10:
            continue
        all_texts.append(wrap(q, ans))
    print(f"   ✅ {len(all_texts) - n_before:,} Dolly examples  [{(time.time()-t0)/60:.1f} min]")
except Exception as e:
    print(f"   ⚠️  Dolly unavailable ({e}), skipping.")

# ── 5. DailyDialog ────────────────────────────────────────────────────────────
print("\n💬 Downloading DailyDialog (conversational)...")
t0 = time.time()
try:
    try:
        dd = load_dataset("agentlans/li2017dailydialog", split="train")
    except Exception:
        dd = load_dataset("DeepPavlov/daily_dialog", split="train")
    n_before = len(all_texts)
    for item in dd:
        convs = item.get("conversations") or item.get("dialog") or []
        if not convs:
            continue
        pairs = (
            [(c.get("from", ""), c.get("value", "")) for c in convs]
            if isinstance(convs[0], dict)
            else [("human" if i % 2 == 0 else "gpt", c) for i, c in enumerate(convs)]
        )
        t = f"<system>\n{SYS}\n</system>\n"
        t += "".join(
            f"<{'human' if 'human' in r.lower() else 'assistant'}>\n{v.strip()}\n"
            f"</{'human' if 'human' in r.lower() else 'assistant'}>\n"
            for r, v in pairs if v.strip()
        )
        if t:
            all_texts.append(t)
    print(f"   ✅ {len(all_texts) - n_before:,} DailyDialog examples  [{(time.time()-t0)/60:.1f} min]")
except Exception as e:
    print(f"   ⚠️  DailyDialog unavailable ({e}), skipping.")

# ── Tokenize & Pack ────────────────────────────────────────────────────────────
print(f"\n📊 Total examples: {len(all_texts):,}  — tokenizing & packing...")
all_chunks = pack_texts(all_texts, "Tokenizing Phase 2 SFT")
print(f"   ✅ {len(all_chunks):,} chunks packed")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\n💾 Saving {len(all_chunks):,} chunks to disk (int16 to halve file size)...")
# int16 safe: LLaMA vocab_size=32000 < 32767
tensor = torch.tensor(all_chunks, dtype=torch.int16)
torch.save(tensor, SAVE_PATH)

size_gb = Path(SAVE_PATH).stat().st_size / 1e9
tokens_B = len(all_chunks) * SEQ_LEN / 1e9

meta = {
    "n_chunks":    len(all_chunks),
    "tokens_B":    round(tokens_B, 3),
    "seq_len":     SEQ_LEN,
    "sources":     [
        f"Identity seeds ({len(IDENTITY_SEEDS)} seeds × {IDENTITY_REPEAT}x)",
        "Alpaca 52K",
        "OpenAssistant OASST1",
        "Databricks Dolly 15K",
        "DailyDialog",
    ],
    "dtype":        "int16",
    "instructions": "Load with: torch.load(path).long() — then wrap in TagAwareDataset",
    "fixes_applied": [
        "Identity seeds 500x (was 20x) — overrides pre-training human identity",
        "EOS/</assistant> get 2x loss weight in train.py TagAwareDataset._weights()",
        "Consistent <system>/<human>/<assistant> format across all datasets",
    ],
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*65}")
print(f"  ✅ PHASE 2 SFT DATA PREP COMPLETE")
print(f"{'='*65}")
print(f"  File:    {SAVE_PATH}")
print(f"  Size:    {size_gb:.2f} GB")
print(f"  Chunks:  {len(all_chunks):,}")
print(f"  Tokens:  {tokens_B:.2f}B")
print()
print("  Now start your GPU session and run:")
print("    python train.py --phase sft \\")
print(f"      --data_file {SAVE_PATH} \\")
print("      --resume /vol/checkpoints/novamind_pretrain_final.pt \\")
print("      --out_dir /vol/checkpoints")
print()
