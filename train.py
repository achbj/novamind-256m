"""
NovaMind-256M Training Script
==============================
3-phase training pipeline with FULL logging and tracking:

  LOGGING (every optimizer step):
    • Rich console table: step, loss (total/think/response), LR, grad_norm,
      tokens/sec, GPU mem, ETA, elapsed time
    • Local JSON-lines log (append-only, survives crashes)
    • CSV log (easy to open in Excel / pandas)
    • W&B integration (graceful fallback to local if unavailable)
    • Red-flag warnings (high grad_norm, loss stuck, OOM risk)

  CHECKPOINTING (aggressive — for Lightning.ai 1-hr sessions):
    • Rolling checkpoint every `save_every` steps (default 500, ~10 min)
    • Milestone checkpoint every `milestone_every` steps (kept forever)
    • Phase-final checkpoint at end of each phase
    • Every checkpoint is 100% resumable: model + optimizer + scaler +
      scheduler_state + step + tokens_seen + full loss histories

  TRAINING:
    • Tag-Aware Loss Curriculum (Novel Contribution #2)
    • BF16 + Flash Attention 2 on H100
    • Gradient checkpointing (saves ~40% VRAM)
    • Separate weight-decay / no-decay param groups
    • Mid-epoch resume after Lightning.ai session expiry

Usage:
  python train.py --phase pretrain --out_dir ./checkpoints
  python train.py --phase sft     --resume ./checkpoints/novamind_phase1_final.pt
  python train.py --phase cot_sft --resume ./checkpoints/novamind_phase2_final.pt
  python train.py --phase pretrain --smoke_test   # quick 50-step pipeline test
"""

import os, sys, json, csv, time, math, argparse, datetime, shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Optional W&B ──────────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import NovaMind256M, NovaMindConfig, tag_aware_loss


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="NovaMind-256M Training")

    # Phase
    p.add_argument("--phase", required=True, choices=["pretrain", "sft", "cot_sft"])

    # Checkpointing
    p.add_argument("--resume",    default=None,          help="Path to checkpoint to resume from")
    p.add_argument("--out_dir",   default="./checkpoints", help="Output dir for checkpoints + logs")

    # Model (defaults match NovaMindConfig)
    p.add_argument("--d_model",    type=int, default=1024)
    p.add_argument("--n_heads",    type=int, default=16)
    p.add_argument("--n_kv_heads", type=int, default=4)
    p.add_argument("--n_layers",   type=int, default=24)
    p.add_argument("--ff_dim",     type=int, default=2304)   # ← 2304, not 2816
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--max_seq_len",type=int, default=2048)

    # Training hypers (None = use phase defaults)
    p.add_argument("--batch_size",        type=int,   default=None)
    p.add_argument("--grad_accum_steps",  type=int,   default=None)
    p.add_argument("--lr",                type=float, default=None)
    p.add_argument("--min_lr_ratio",      type=float, default=0.1)
    p.add_argument("--warmup_steps",      type=int,   default=None)
    p.add_argument("--epochs",            type=int,   default=None)
    p.add_argument("--max_steps",         type=int,   default=None)
    p.add_argument("--max_grad_norm",     type=float, default=1.0)
    p.add_argument("--weight_decay",      type=float, default=0.1)

    # Data
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_cap",    type=int, default=None,
                   help="Cap dataset samples (useful for smoke tests)")

    # Checkpoint schedule
    p.add_argument("--save_every",     type=int, default=500,
                   help="Rolling checkpoint every N optimizer steps (~10 min on H100)")
    p.add_argument("--milestone_every",type=int, default=2000,
                   help="Permanent milestone checkpoint every N steps")
    p.add_argument("--val_every",      type=int, default=500,
                   help="Validation + sample generation every N steps")
    p.add_argument("--val_batches",    type=int, default=50)

    # Logging
    p.add_argument("--log_every",      type=int, default=10,
                   help="Console + JSON/CSV log every N optimizer steps")
    p.add_argument("--wandb_project",  default="novamind-256m")
    p.add_argument("--wandb_run",      default=None)
    p.add_argument("--no_wandb",       action="store_true")

    # Misc
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--compile",    action="store_true", help="torch.compile (PyTorch 2.0+)")
    p.add_argument("--smoke_test", action="store_true", help="50-step end-to-end test")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
PHASE_DEFAULTS = {
    "pretrain": {
        "lr": 3e-4, "warmup_steps": 2000, "epochs": 2,
        "batch_size": 4, "grad_accum": 8,
        "think_weight": 0.0,
        "desc": "Phase 1 — Language Pretraining (~5.5B tokens: FineWeb-Edu + TinyStories)",
    },
    "sft": {
        "lr": 1e-4, "warmup_steps": 500, "epochs": 2,
        "batch_size": 4, "grad_accum": 4,
        "think_weight": 0.5,
        "desc": "Phase 2 — Instruction SFT (~300M tokens: Alpaca + DailyDialog)",
    },
    "cot_sft": {
        "lr": 5e-5, "warmup_steps": 200, "epochs": 2,
        "batch_size": 2, "grad_accum": 8,
        "think_weight": 1.5,
        "desc": "Phase 3 — CoT SFT (~150M tokens: OpenHermes + GSM8K)",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════
class PretrainDataset(Dataset):
    """Phase 1: raw text chunks, uniform loss weights."""
    def __init__(self, chunks: torch.Tensor):
        self.chunks = chunks  # [N, seq_len+1]

    def __len__(self): return self.chunks.shape[0]

    def __getitem__(self, idx):
        c = self.chunks[idx].clone()
        return c[:-1], c[1:].clone(), torch.ones(c.shape[0]-1, dtype=torch.float32)


class TagAwareDataset(Dataset):
    """
    Phase 2/3: returns (input_ids, targets, token_weights).
    Weights encode Tag-Aware Loss Curriculum — see model.py for full explanation.
    """
    def __init__(self, chunks: torch.Tensor, think_weight: float,
                 think_start_id=-1, think_end_id=-1,
                 asst_start_id=-1, user_start_id=-1, eos_id=2):
        self.chunks = chunks
        self.tw = think_weight
        self.think_start = think_start_id
        self.think_end   = think_end_id
        self.asst_start  = asst_start_id
        self.user_start  = user_start_id
        self.eos         = eos_id

    def __len__(self): return self.chunks.shape[0]

    def __getitem__(self, idx):
        c = self.chunks[idx].clone()
        x, y = c[:-1], c[1:].clone()
        return x, y, self._weights(y)

    def _weights(self, tokens: torch.Tensor) -> torch.Tensor:
        T = len(tokens)
        w = torch.ones(T, dtype=torch.float32)
        if self.think_start < 0 and self.asst_start < 0:
            return w
        state = "default"
        for i, tok in enumerate(tokens.tolist()):
            if   tok == self.user_start:  state = "user"
            elif tok == self.think_start: state = "think"
            elif tok == self.think_end:   state = "default"
            elif tok == self.asst_start:  state = "assistant"
            elif tok == self.eos:         state = "default"
            w[i] = 0.0 if state == "user" else (self.tw if state == "think" else 1.0)
        return w


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def pack_texts(texts, tokenizer, seq_len: int, desc="packing") -> torch.Tensor:
    """Document-boundary-aware token packing → [N, seq_len+1] stacked tensor."""
    eos = tokenizer.eos_token_id
    chunks, current = [], []
    for text in tqdm(texts, desc=desc, leave=False):
        ids = tokenizer.encode(text, add_special_tokens=False) + [eos]
        if len(ids) >= seq_len + 1:
            if current:
                chunks.append(current + [eos] * ((seq_len+1) - len(current)))
                current = []
            for i in range(0, len(ids) - seq_len, seq_len):
                chunks.append(ids[i: i+seq_len+1])
        else:
            if len(current) + len(ids) > seq_len + 1:
                chunks.append(current + [eos] * ((seq_len+1) - len(current)))
                current = []
            current.extend(ids)
    if len(current) >= 2:
        chunks.append(current + [eos] * ((seq_len+1) - len(current)))
    if not chunks:
        return torch.empty(0, seq_len+1, dtype=torch.long)
    return torch.tensor(chunks, dtype=torch.long)


def _tag_id(tokenizer, tag: str) -> int:
    ids = tokenizer.encode(tag, add_special_tokens=False)
    return ids[0] if ids else -1


def load_pretrain_data(tokenizer, seq_len, cap=None) -> torch.Tensor:
    from datasets import load_dataset
    print("📚 Loading FineWeb-Edu...")
    fw_cap = cap if cap else 2_500_000
    fw_ds  = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    fw_texts = [row["text"] for i, row in enumerate(fw_ds) if i < fw_cap]
    fw = pack_texts(fw_texts, tokenizer, seq_len, "FineWeb-Edu")
    print(f"   ✅ {len(fw):,} chunks")

    print("📚 Loading TinyStories...")
    ts_ds = load_dataset("roneneldan/TinyStories", split="train")
    if cap: ts_ds = ts_ds.select(range(min(cap//2, len(ts_ds))))
    ts = pack_texts([r["text"] for r in ts_ds], tokenizer, seq_len, "TinyStories")
    print(f"   ✅ {len(ts):,} chunks")

    all_c = torch.cat([fw, ts], dim=0)
    print(f"   ✅ Phase 1 total: {len(all_c):,} chunks ({len(all_c)*seq_len/1e9:.2f}B tokens)")
    return all_c


def load_sft_data(tokenizer, seq_len, think_weight, cap=None) -> TagAwareDataset:
    from datasets import load_dataset
    texts = []

    print("💬 Loading Alpaca...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    if cap: alpaca = alpaca.select(range(min(cap, len(alpaca))))
    for r in alpaca:
        u = r["instruction"].strip() + (f"\n\n{r['input'].strip()}" if r["input"].strip() else "")
        texts.append(f"<human>\n{u}\n</human>\n<assistant>\n{r['output'].strip()}\n</assistant>")
    print(f"   ✅ {len(texts):,} Alpaca examples")

    print("💬 Loading DailyDialog...")
    try:
        dd = load_dataset("agentlans/li2017dailydialog", split="train")
    except Exception:
        dd = load_dataset("DeepPavlov/daily_dialog", split="train")
    if cap: dd = dd.select(range(min(cap//2, len(dd))))
    for item in dd:
        convs = item.get("conversations") or item.get("dialog") or []
        if not convs: continue
        pairs = [(c.get("from",""), c.get("value","")) for c in convs] \
                if isinstance(convs[0], dict) \
                else [("human" if i%2==0 else "gpt", c) for i,c in enumerate(convs)]
        t = "".join(
            f"<{'human' if 'human' in r.lower() else 'assistant'}>\n{v.strip()}\n"
            f"</{'human' if 'human' in r.lower() else 'assistant'}>\n"
            for r,v in pairs if v.strip()
        )
        if t: texts.append(t)

    print("💬 Loading ShareGPT (for deep conversations)...")
    try:
        sg = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        sg_cap = min(cap, 50_000) if cap else 50_000
        for i, row in enumerate(sg):
            if i >= sg_cap: break
            convs = row.get("conversations", [])
            if not convs: continue
            
            t = ""
            for msg in convs:
                role = msg.get("from", "")
                val = msg.get("value", "").strip()
                if role == "human":
                    t += f"<human>\n{val}\n</human>\n"
                elif role == "gpt":
                    t += f"<assistant>\n{val}\n</assistant>\n"
            if t: texts.append(t)
    except Exception as e:
        print(f"   ⚠️  ShareGPT unavailable ({e}), skipping.")

    print(f"   ✅ {len(texts):,} total SFT examples")

    chunks = pack_texts(texts, tokenizer, seq_len, "SFT packing")
    print(f"   ✅ {len(chunks):,} chunks packed")
    return TagAwareDataset(chunks, think_weight,
                           _tag_id(tokenizer,"<think>"), _tag_id(tokenizer,"</think>"),
                           _tag_id(tokenizer,"<assistant>"), _tag_id(tokenizer,"<human>"),
                           tokenizer.eos_token_id)


def load_cot_data(tokenizer, seq_len, think_weight, cap=None) -> TagAwareDataset:
    from datasets import load_dataset
    texts = []

    print("🧠 Loading OpenHermes-2.5...")
    try:
        oh = load_dataset("teknium/OpenHermes-2.5", split="train")
        oh_cap = min(cap, 100_000) if cap else 100_000
        for i, row in enumerate(oh):
            if i >= oh_cap: break
            t = ""
            for c in row.get("conversations", []):
                r, v = c.get("from",""), c.get("value","").strip()
                if r == "system":  t += f"<system>\n{v}\n</system>\n"
                elif r == "human": t += f"<human>\n{v}\n</human>\n"
                elif r == "gpt":
                    parts = v.split("\n\n")
                    if len(parts) > 1 and any(k in v for k in ["Step ","##","Let me"]):
                        t += f"<think>\n{chr(10).join(parts[:-1]).strip()}\n</think>\n"
                        t += f"<assistant>\n{parts[-1].strip()}\n</assistant>\n"
                    else:
                        t += f"<assistant>\n{v}\n</assistant>\n"
            if t: texts.append(t)
        print(f"   ✅ {len(texts):,} OpenHermes examples")
    except Exception as e:
        print(f"   ⚠️  OpenHermes unavailable ({e}), skipping.")

    print("🧠 Loading GSM8K math reasoning...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    if cap: gsm = gsm.select(range(min(cap//5, len(gsm))))
    for row in gsm:
        q, a = row["question"].strip(), row["answer"].strip()
        if "####" in a:
            reasoning, final = a.split("####", 1)
            texts.append(
                f"<human>\n{q}\n</human>\n"
                f"<think>\n{reasoning.strip()}\n</think>\n"
                f"<assistant>\n{final.strip()}\n</assistant>"
            )
        else:
            texts.append(f"<human>\n{q}\n</human>\n<assistant>\n{a}\n</assistant>")
    print(f"   ✅ {len(texts):,} total CoT examples")

    chunks = pack_texts(texts, tokenizer, seq_len, "CoT packing")
    print(f"   ✅ {len(chunks):,} chunks packed")
    return TagAwareDataset(chunks, think_weight,
                           _tag_id(tokenizer,"<think>"), _tag_id(tokenizer,"</think>"),
                           _tag_id(tokenizer,"<assistant>"), _tag_id(tokenizer,"<human>"),
                           tokenizer.eos_token_id)


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
def get_lr(step, warmup, max_steps, peak, min_lr):
    """Linear warmup → cosine decay. Standard for all modern LLMs."""
    if step < warmup:
        return peak * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class TrainingLogger:
    """
    Full logging to three sinks simultaneously:
      1. Console — rich formatted table printed every log_every steps
      2. JSON-lines — one JSON object per log event, append-only (survives crashes)
      3. CSV — spreadsheet-friendly, one row per log event
      4. W&B  — cloud tracking (optional, graceful fallback)

    Also maintains running averages so the console always shows the smoothed
    loss over the last `log_every` steps, not just the current noisy batch.
    """

    # CSV columns — every metric we track
    CSV_FIELDS = [
        "step", "epoch_frac", "phase",
        "loss_total", "loss_think", "loss_response",
        "lr", "grad_norm",
        "tokens_seen_B", "tokens_per_sec",
        "gpu_mem_gb", "gpu_mem_reserved_gb",
        "loss_weight_think",
        "val_loss",
        "time_elapsed_min", "eta_min",
        "timestamp",
    ]

    def __init__(self, out_dir: str, phase: str, config: dict,
                 wandb_project="novamind-256m", wandb_run=None,
                 use_wandb=True, log_every=10):
        self.out_dir    = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.phase      = phase
        self.log_every  = log_every
        self.use_wandb  = use_wandb and WANDB_AVAILABLE

        # ── File paths ────────────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.out_dir / f"log_{phase}_{ts}.jsonl"
        self.csv_path   = self.out_dir / f"log_{phase}_{ts}.csv"
        # Keep a "latest" symlink so scripts can always find the current log
        self._symlink(self.jsonl_path, self.out_dir / f"log_{phase}_latest.jsonl")
        self._symlink(self.csv_path,   self.out_dir / f"log_{phase}_latest.csv")

        # ── CSV: write header immediately ────────────────────────────────
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.CSV_FIELDS, extrasaction="ignore").writeheader()

        # ── W&B ──────────────────────────────────────────────────────────
        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run or f"novamind-{phase}-{ts}",
                    config=config,
                    resume="allow",
                )
                print("📡 W&B initialized:", wandb.run.url)
            except Exception as e:
                print(f"⚠️  W&B failed ({e}) — falling back to local logs only.")
                self.use_wandb = False

        # ── Running accumulators (reset every log_every steps) ────────────
        self._acc_loss_total    = 0.0
        self._acc_loss_think    = 0.0
        self._acc_loss_response = 0.0
        self._acc_grad_norm     = 0.0
        self._acc_tok_per_sec   = 0.0
        self._acc_count         = 0

        print(f"\n📊 Logger ready:")
        print(f"   JSON-lines → {self.jsonl_path}")
        print(f"   CSV        → {self.csv_path}")
        if not self.use_wandb:
            print(f"   W&B        → disabled (install wandb for cloud tracking)")

    def _symlink(self, target: Path, link: Path):
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target.name)
        except Exception:
            pass  # symlinks may fail on some systems, not critical

    def accumulate(self, loss_total: float, loss_think: float, loss_response: float,
                   grad_norm: float, tok_per_sec: float):
        """Accumulate per-step values for smoothed logging."""
        self._acc_loss_total    += loss_total
        self._acc_loss_think    += loss_think
        self._acc_loss_response += loss_response
        self._acc_grad_norm     += grad_norm
        self._acc_tok_per_sec   += tok_per_sec
        self._acc_count         += 1

    def log_step(self, step: int, epoch_frac: float, lr: float,
                 tokens_seen: int, think_weight: float,
                 t_start: float, max_steps: int,
                 val_loss: Optional[float] = None):
        """
        Called every log_every steps. Drains the accumulators, prints to
        console, and writes to JSON + CSV + W&B.
        """
        n = max(self._acc_count, 1)
        avg_loss     = self._acc_loss_total    / n
        avg_think    = self._acc_loss_think    / n
        avg_response = self._acc_loss_response / n
        avg_gnorm    = self._acc_grad_norm     / n
        avg_tps      = self._acc_tok_per_sec   / n

        elapsed_min = (time.time() - t_start) / 60.0
        steps_done  = step
        steps_left  = max(max_steps - steps_done, 0)
        secs_per_step = (time.time() - t_start) / max(steps_done, 1)
        eta_min     = steps_left * secs_per_step / 60.0

        gpu_mem    = torch.cuda.memory_allocated()  / 1e9 if torch.cuda.is_available() else 0.0
        gpu_res    = torch.cuda.memory_reserved()   / 1e9 if torch.cuda.is_available() else 0.0
        ts_str     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Build metrics dict ────────────────────────────────────────────
        metrics = {
            "step":              step,
            "epoch_frac":        round(epoch_frac, 4),
            "phase":             self.phase,
            "loss_total":        round(avg_loss,     6),
            "loss_think":        round(avg_think,    6),
            "loss_response":     round(avg_response, 6),
            "lr":                lr,
            "grad_norm":         round(avg_gnorm,    4),
            "tokens_seen_B":     round(tokens_seen / 1e9, 6),
            "tokens_per_sec":    round(avg_tps,     1),
            "gpu_mem_gb":        round(gpu_mem, 3),
            "gpu_mem_reserved_gb":round(gpu_res, 3),
            "loss_weight_think": think_weight,
            "val_loss":          round(val_loss, 6) if val_loss is not None else None,
            "time_elapsed_min":  round(elapsed_min, 2),
            "eta_min":           round(eta_min, 1),
            "timestamp":         ts_str,
        }

        # ── Console output ──────────────────────────────────────────────
        self._print_step(metrics)

        # ── Red-flag warnings ───────────────────────────────────────────
        if avg_gnorm > 10.0:
            print(f"  🚨 HIGH GRAD NORM {avg_gnorm:.2f} — training may be unstable! Consider reducing LR.")
        if step > 200 and avg_loss > 4.0:
            print(f"  🚨 LOSS STILL HIGH ({avg_loss:.4f}) after {step} steps — check dataset & LR!")
        if gpu_res > 38.0:
            print(f"  ⚠️  GPU MEM {gpu_res:.1f}GB / 40GB — close to OOM! Reduce batch_size or seq_len.")
        if avg_think > 0 and avg_response > 0 and avg_think > avg_response * 3:
            print(f"  ℹ️  Think loss ({avg_think:.3f}) >> Response loss ({avg_response:.3f}) — normal in Phase 3.")

        # ── JSON-lines ──────────────────────────────────────────────────
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # ── CSV ─────────────────────────────────────────────────────────
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.CSV_FIELDS, extrasaction="ignore").writerow(metrics)

        # ── W&B ─────────────────────────────────────────────────────────
        if self.use_wandb:
            try:
                wandb.log({
                    "train/loss":          avg_loss,
                    "train/loss_think":    avg_think,
                    "train/loss_response": avg_response,
                    "train/grad_norm":     avg_gnorm,
                    "train/lr":            lr,
                    "train/tokens_seen_B": tokens_seen / 1e9,
                    "train/tokens_per_sec":avg_tps,
                    "train/gpu_mem_gb":    gpu_mem,
                    "train/eta_min":       eta_min,
                    **({"val/loss": val_loss} if val_loss is not None else {}),
                }, step=step)
            except Exception:
                pass

        # ── Reset accumulators ──────────────────────────────────────────
        self._acc_loss_total = self._acc_loss_think = self._acc_loss_response = 0.0
        self._acc_grad_norm  = self._acc_tok_per_sec = 0.0
        self._acc_count      = 0

        return metrics

    def log_val(self, step: int, val_loss: float, samples: list[tuple[str,str]]):
        """Log validation loss and sample generations."""
        metrics = {"step": step, "val_loss": round(val_loss, 6),
                   "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        # Write to JSON log
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"type": "validation", **metrics}) + "\n")

        # Console
        print(f"\n{'─'*65}")
        print(f"  📊 VALIDATION  step={step:,}  val_loss={val_loss:.6f}")
        print(f"{'─'*65}")
        for q, a in samples:
            print(f"  Q: {q}")
            resp = a[:200] + ("..." if len(a) > 200 else "")
            print(f"  A: {resp}")
            print()
        print(f"{'─'*65}\n")

        if self.use_wandb:
            try:
                wandb.log({"val/loss": val_loss}, step=step)
                wandb.log({"samples": wandb.Table(
                    columns=["prompt","response"],
                    data=[[q, a] for q,a in samples],
                )}, step=step)
            except Exception:
                pass

    def _print_step(self, m: dict):
        """Rich console table row."""
        bar_width  = 20
        filled     = int(bar_width * m["step"] / max(m.get("_max_steps", m["step"]), 1))
        # Progress indicator: ████████░░░░░░░░░░░░
        progress_bar = "█" * filled + "░" * (bar_width - filled)

        print(
            f"  step {m['step']:>7,} [{progress_bar}] "
            f"loss={m['loss_total']:.4f} "
            f"(think={m['loss_think']:.4f} resp={m['loss_response']:.4f}) | "
            f"lr={m['lr']:.2e} gnorm={m['grad_norm']:.3f} | "
            f"{m['tokens_per_sec']:>9,.0f} tok/s | "
            f"mem={m['gpu_mem_reserved_gb']:.1f}GB | "
            f"elapsed={m['time_elapsed_min']:.1f}m ETA={m['eta_min']:.0f}m"
        )

    def finish(self):
        print(f"\n📊 Training complete. Logs saved:")
        print(f"   {self.jsonl_path}")
        print(f"   {self.csv_path}")
        if self.use_wandb:
            wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# TAG-SPECIFIC LOSS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_tag_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_weights: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute the weighted total loss PLUS separate think and response sub-losses
    for monitoring. The total loss is what we .backward() on; the sub-losses
    are purely for logging to understand training progress by token type.

    Returns:
        (total_loss, think_loss_item, response_loss_item)
    """
    total_loss = tag_aware_loss(logits, targets, token_weights)

    with torch.no_grad():
        per_tok = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1, reduction="none",
        )  # [B*T]
        flat_w = token_weights.view(-1)
        valid  = (targets.view(-1) != -1).float()

        # Think tokens: 0 < weight < 1 (phase 2) or weight > 1 (phase 3)
        # Response tokens: weight == 1.0
        # User/system tokens: weight == 0.0
        think_mask    = ((flat_w > 0.0) & (flat_w != 1.0) & valid.bool()).float()
        response_mask = ((flat_w == 1.0) & valid.bool()).float()

        think_loss    = (per_tok * think_mask).sum()    / think_mask.sum().clamp(min=1e-8)
        response_loss = (per_tok * response_mask).sum() / response_mask.sum().clamp(min=1e-8)

    return total_loss, think_loss.item(), response_loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(
    out_dir: str, model: nn.Module, optimizer, scaler,
    step: int, phase: str, tokens_seen: int, epoch_frac: float,
    train_loss: float, think_loss: float, response_loss: float,
    lr: float, grad_norm: float,
    config: NovaMindConfig,
    train_loss_history: list, val_loss_history: list,
    milestone=False, phase_final=False,
):
    """
    Save a 100% resumable checkpoint containing:
      • model weights (DataParallel-unwrapped)
      • optimizer state
      • scaler state (for BF16/FP16)
      • scheduler state (current LR, step)
      • step counter + tokens seen (for resuming data iteration)
      • full loss histories (for plotting)
      • all hyperparameters

    Checkpoint types:
      • Rolling  : novamind_roll_latest.pt       — always overwritten
      • Milestone: novamind_milestone_NNNNNNN.pt — kept forever
      • Final    : novamind_{phase}_final.pt     — kept forever
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_state = (model.module.state_dict() if hasattr(model, "module")
                   else model.state_dict())

    ckpt = {
        # Model
        "model_state_dict":     model_state,
        "config":               config.to_dict(),
        # Optimizer / precision
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict() if scaler is not None else None,
        # Training state (all needed to resume mid-epoch)
        "step":                 step,
        "phase":                phase,
        "epoch_frac":           epoch_frac,
        "tokens_seen":          tokens_seen,
        "lr":                   lr,
        # Last-step metrics (for human inspection)
        "train_loss":           train_loss,
        "think_loss":           think_loss,
        "response_loss":        response_loss,
        "grad_norm":            grad_norm,
        # Full histories (for loss curves)
        "train_loss_history":   train_loss_history,
        "val_loss_history":     val_loss_history,
        # Meta
        "timestamp":            datetime.datetime.now().isoformat(),
        "torch_version":        torch.__version__,
    }

    # 1. Always save rolling (overwrites previous)
    roll = out / "novamind_roll_latest.pt"
    torch.save(ckpt, roll)
    mb = roll.stat().st_size / 1e6

    print(f"\n{'─'*65}")
    print(f"  💾 CHECKPOINT  step={step:,} | loss={train_loss:.4f} | "
          f"think={think_loss:.4f} | resp={response_loss:.4f}")
    print(f"     tokens={tokens_seen/1e9:.3f}B | lr={lr:.2e} | size={mb:.0f} MB")
    print(f"     → {roll}")

    # 2. Milestone — always copy (never overwrite) for permanent history
    if milestone:
        ms = out / f"novamind_milestone_{step:07d}.pt"
        shutil.copy2(roll, ms)
        print(f"     → MILESTONE {ms.name}")

    # 3. Phase final
    if phase_final:
        fin = out / f"novamind_{phase}_final.pt"
        shutil.copy2(roll, fin)
        print(f"     → PHASE FINAL {fin.name}")

    print(f"{'─'*65}\n")
    return roll


def write_checkpoint_index(out_dir: str, step: int, loss: float, tokens_seen: int, ckpt_path: str):
    """
    Maintain a human-readable checkpoint index file so you can quickly see
    what checkpoints exist and at what training state.
    """
    index_path = Path(out_dir) / "checkpoint_index.jsonl"
    with open(index_path, "a") as f:
        f.write(json.dumps({
            "step": step, "loss": round(loss, 6),
            "tokens_seen_B": round(tokens_seen/1e9, 4),
            "path": str(ckpt_path),
            "saved_at": datetime.datetime.now().isoformat(),
        }) + "\n")


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler, device):
    """Load checkpoint and return the full ckpt dict for state restoration."""
    print(f"\n📂 Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle DataParallel prefix mismatch
    state = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    (model.module if hasattr(model, "module") else model).load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    print(f"   ✅ Resumed: phase={ckpt['phase']} | step={ckpt['step']:,} | "
          f"tokens={ckpt['tokens_seen']/1e9:.3f}B | "
          f"last_loss={ckpt.get('train_loss', '?')}")
    return ckpt


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def estimate_val_loss(model, loader, device, num_batches=50, dtype=torch.bfloat16) -> float:
    model.eval()
    losses = []
    for i, (x, y, w) in enumerate(loader):
        if i >= num_batches: break
        x, y, w = x.to(device), y.to(device), w.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=torch.cuda.is_available()):
            _, loss = model(x, y, w)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts, max_new_tokens=100):
    model.eval()
    inner   = model.module if hasattr(model, "module") else model
    results = []
    for prompt in prompts:
        fmt = f"<human>\n{prompt}\n</human>\n<assistant>\n"
        ids = tokenizer.encode(fmt, return_tensors="pt").to(device)
        try:
            stop_ids = [tokenizer.encode("</assistant>", add_special_tokens=False)[0]]
        except Exception:
            stop_ids = None
        out = inner.generate(ids, max_new_tokens=max_new_tokens,
                             temperature=0.7, top_k=40, top_p=0.9,
                             stop_token_ids=stop_ids)
        full   = tokenizer.decode(out[0], skip_special_tokens=False)
        resp   = full.split("<assistant>\n")[-1].split("</assistant>")[0].strip()
        results.append((prompt, resp))
    model.train()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
TEST_PROMPTS = [
    "What is the capital of France?",
    "What is 15 multiplied by 17?",
    "Tell me a short joke.",
    "Explain what a neural network is in simple terms.",
]

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Phase defaults (CLI args override if provided) ────────────────────────
    pd = PHASE_DEFAULTS[args.phase]
    peak_lr       = args.lr             or pd["lr"]
    warmup_steps  = args.warmup_steps   or pd["warmup_steps"]
    batch_size    = args.batch_size     or pd["batch_size"]
    grad_accum    = args.grad_accum_steps or pd["grad_accum"]
    n_epochs      = args.epochs         or pd["epochs"]
    think_weight  = pd["think_weight"]
    min_lr        = peak_lr * args.min_lr_ratio

    # ── Device & Precision ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    # H100 supports BF16 natively — more numerically stable than FP16
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  🚀 NovaMind-256M Training")
    print(f"  Phase : {args.phase.upper()} — {pd['desc']}")
    print(f"  Device: {device}  Precision: {dtype}")
    if use_cuda:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i} : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"  Peak LR: {peak_lr:.2e} → min {min_lr:.2e}  | Warmup: {warmup_steps} steps")
    print(f"  Batch: {batch_size} seqs × {args.max_seq_len} tokens × {grad_accum} accum = "
          f"{batch_size*args.max_seq_len*grad_accum:,} tok/update")
    print(f"  Think-token loss weight: {think_weight}")
    print(f"  Checkpoint every: {args.save_every} steps  |  Milestone every: {args.milestone_every} steps")
    print(f"{'═'*65}\n")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print("🔤 Loading tokenizer...")
    for tok_name in ["NousResearch/Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf", "gpt2"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_name)
            print(f"   ✅ {tok_name} — vocab: {tokenizer.vocab_size:,}")
            break
        except Exception as e:
            print(f"   ⚠️  {tok_name} failed: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────────
    cap = 500 if args.smoke_test else args.data_cap
    print(f"\n📂 Loading {args.phase} dataset{'  [SMOKE TEST: cap=500]' if args.smoke_test else ''}...")

    if args.phase == "pretrain":
        chunks  = load_pretrain_data(tokenizer, args.max_seq_len, cap)
        dataset = PretrainDataset(chunks)
    elif args.phase == "sft":
        dataset = load_sft_data(tokenizer, args.max_seq_len, think_weight, cap)
    else:
        dataset = load_cot_data(tokenizer, args.max_seq_len, think_weight, cap)

    n_val   = max(1, int(0.05 * len(dataset)))  # 5% held out for validation
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"   Train: {n_train:,} chunks | Val: {n_val:,} chunks")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_cuda,
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=min(2, args.num_workers), pin_memory=use_cuda)
    print(f"   DataLoader: {len(train_loader):,} batches × {batch_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = NovaMindConfig(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_heads=args.n_heads,       n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,     ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
    )
    print(f"\n🏗️  Building model...")
    model = NovaMind256M(model_config).to(device)
    model.gradient_checkpointing_enable()  # saves ~40% VRAM at cost of ~20% speed
    if args.compile and hasattr(torch, "compile"):
        print("⚡ torch.compile enabled...")
        model = torch.compile(model)
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"🖥️  DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    inner = model.module if hasattr(model, "module") else model
    inner.count_parameters(print_table=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate param groups: weight decay only on >= 2D params (matrices),
    # NOT on biases or norms (1D params). Standard best practice.
    decay_p    = [p for n,p in model.named_parameters() if p.requires_grad and p.dim()>=2]
    no_decay_p = [p for n,p in model.named_parameters() if p.requires_grad and p.dim()<2]
    optimizer  = optim.AdamW(
        [{"params": decay_p, "weight_decay": args.weight_decay},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
    # GradScaler: needed for FP16, no-op for BF16 (H100)
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))

    # ── Max steps ─────────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    max_steps = args.max_steps or (n_epochs * steps_per_epoch)
    if args.smoke_test: max_steps = 50
    print(f"\n   Steps/epoch: {steps_per_epoch:,}  |  Total max steps: {max_steps:,}")
    print(f"   Estimated training time on H100: "
          f"~{max_steps * batch_size * args.max_seq_len * grad_accum / 220_000 / 3600:.1f} hrs\n")

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step        = 0
    tokens_seen        = 0
    train_loss_history = []
    val_loss_history   = []

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler, device)
        if ckpt.get("phase") == args.phase:
            global_step        = ckpt["step"]
            tokens_seen        = ckpt["tokens_seen"]
            train_loss_history = ckpt.get("train_loss_history", [])
            val_loss_history   = ckpt.get("val_loss_history", [])
            # Restore LR to where we left off
            resumed_lr = get_lr(global_step, warmup_steps, max_steps, peak_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = resumed_lr
        else:
            print(f"\n   ⏩ Transitioning from phase '{ckpt.get('phase')}' to '{args.phase}'. Resetting step counter.")
            tokens_seen = ckpt.get("tokens_seen", 0)

    # ── Logger ────────────────────────────────────────────────────────────────
    cfg_for_log = {**vars(args), **model_config.to_dict(),
                   "peak_lr": peak_lr, "min_lr": min_lr,
                   "think_weight": think_weight, "max_steps": max_steps}
    logger = TrainingLogger(
        out_dir=args.out_dir, phase=args.phase, config=cfg_for_log,
        wandb_project=args.wandb_project, wandb_run=args.wandb_run,
        use_wandb=(not args.no_wandb),
        log_every=args.log_every,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print(f"  🚦 TRAINING STARTED — Phase: {args.phase.upper()}")
    print(f"  Resume from step: {global_step:,}  |  Target: {max_steps:,} steps")
    print(f"{'═'*65}\n")

    model.train()
    optimizer.zero_grad()

    t_start   = time.time()
    t_batch   = time.time()
    last_loss = float("nan")      # track for checkpoint metadata
    last_think_loss    = 0.0
    last_response_loss = 0.0
    last_grad_norm     = 0.0

    start_epoch = 1
    skip_micro_steps = 0
    if args.resume and ckpt.get("phase") == args.phase:
        total_micro_steps_done = global_step * grad_accum
        start_epoch = (total_micro_steps_done // len(train_loader)) + 1
        skip_micro_steps = total_micro_steps_done % len(train_loader)

    for epoch in range(start_epoch, n_epochs + 1):
        if global_step >= max_steps:
            break

        print(f"  📅 Epoch {epoch}/{n_epochs}")

        for micro_step, (x, y, w) in enumerate(train_loader):
            if global_step >= max_steps:
                break
                
            # Skip already processed batches if resuming mid-epoch
            if epoch == start_epoch and micro_step < skip_micro_steps:
                continue

            x, y, w = x.to(device), y.to(device), w.to(device)
            batch_tokens = x.numel()
            t_b0 = time.time()

            # ── Forward + loss ─────────────────────────────────────────────
            with torch.autocast(device_type=device.type, dtype=dtype,
                                 enabled=use_cuda):
                logits, total_loss = model(x, y, w)
                # During training, model returns empty logits — recompute
                # logits only if needed for sub-loss breakdown (no_grad below)
                scaled_loss = total_loss.mean() / grad_accum

            # ── Backward ──────────────────────────────────────────────────
            if dtype == torch.float16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # ── Compute per-tag sub-losses for logging (cheap, no_grad) ──
            # We need logits for this — call model again in eval mode for
            # a micro-subset to get the breakdown without full batch cost.
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_cuda):
                    logits_for_log, _ = (model.module if hasattr(model,"module") else model)(
                        x[:1], None  # just first sample, no loss
                    )
            # Compute sub-losses on that single sample
            _, think_l, response_l = compute_tag_losses(
                logits_for_log, y[:1], w[:1]
            )

            # ── Optimizer step (every grad_accum micro-steps) ──────────────
            if (micro_step + 1) % grad_accum == 0:
                # Clip gradient norm — critical: must unscale first for FP16
                if dtype == torch.float16:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                ).item()

                if dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                global_step  += 1
                tokens_seen  += batch_tokens * grad_accum

                # Update LR
                lr = get_lr(global_step, warmup_steps, max_steps, peak_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Track last values for checkpointing
                last_loss          = total_loss.mean().item()
                last_think_loss    = think_l
                last_response_loss = response_l
                last_grad_norm     = grad_norm

                tok_per_sec = batch_tokens * grad_accum / max(time.time() - t_b0, 1e-6)
                epoch_frac  = (epoch - 1) + (micro_step / len(train_loader))

                # ── Accumulate for smoothed logging ───────────────────────
                logger.accumulate(last_loss, think_l, response_l, grad_norm, tok_per_sec)

                # ── Log to console + files every log_every steps ──────────
                if global_step % args.log_every == 0:
                    m = logger.log_step(
                        step=global_step,
                        epoch_frac=epoch_frac,
                        lr=lr,
                        tokens_seen=tokens_seen,
                        think_weight=think_weight,
                        t_start=t_start,
                        max_steps=max_steps,
                    )
                    m["_max_steps"] = max_steps   # used by progress bar
                    train_loss_history.append({
                        "step": global_step,
                        "loss": last_loss,
                        "think_loss": think_l,
                        "response_loss": response_l,
                    })

                # ── Validation every val_every steps ──────────────────────
                if global_step % args.val_every == 0:
                    val_loss = estimate_val_loss(model, val_loader, device,
                                                 args.val_batches, dtype)
                    samples  = generate_samples(inner, tokenizer, device, TEST_PROMPTS)
                    logger.log_val(global_step, val_loss, samples)
                    val_loss_history.append({"step": global_step, "val_loss": val_loss})

                # ── Checkpoint every save_every steps ─────────────────────
                if global_step % args.save_every == 0:
                    milestone = (global_step % args.milestone_every == 0)
                    ckpt_path = save_checkpoint(
                        args.out_dir, model, optimizer, scaler,
                        step=global_step, phase=args.phase,
                        tokens_seen=tokens_seen, epoch_frac=epoch_frac,
                        train_loss=last_loss,
                        think_loss=last_think_loss,
                        response_loss=last_response_loss,
                        lr=lr, grad_norm=last_grad_norm,
                        config=model_config,
                        train_loss_history=train_loss_history,
                        val_loss_history=val_loss_history,
                        milestone=milestone,
                    )
                    write_checkpoint_index(args.out_dir, global_step, last_loss,
                                           tokens_seen, str(ckpt_path))

            t_b0 = time.time()

        print(f"\n  ✅ Epoch {epoch}/{n_epochs} done | "
              f"steps={global_step:,} | tokens={tokens_seen/1e9:.3f}B\n")

    # ── Phase Final Checkpoint ────────────────────────────────────────────────
    save_checkpoint(
        args.out_dir, model, optimizer, scaler,
        step=global_step, phase=args.phase,
        tokens_seen=tokens_seen, epoch_frac=float(n_epochs),
        train_loss=last_loss, think_loss=last_think_loss,
        response_loss=last_response_loss, lr=min_lr, grad_norm=0.0,
        config=model_config,
        train_loss_history=train_loss_history, val_loss_history=val_loss_history,
        milestone=True, phase_final=True,
    )
    write_checkpoint_index(args.out_dir, global_step, last_loss,
                           tokens_seen, f"novamind_{args.phase}_final.pt")

    total_min = (time.time() - t_start) / 60
    print(f"\n{'═'*65}")
    print(f"  ✅ Phase {args.phase.upper()} COMPLETE")
    print(f"     Steps        : {global_step:,} / {max_steps:,}")
    print(f"     Tokens seen  : {tokens_seen/1e9:.3f}B")
    print(f"     Total time   : {total_min:.1f} min")
    print(f"     Final loss   : {last_loss:.6f}")
    print(f"     Checkpoints  : {args.out_dir}/")
    print(f"{'═'*65}\n")

    logger.finish()

    # Print what to run next
    next_phase = {"pretrain":"sft","sft":"cot_sft","cot_sft":None}[args.phase]
    if next_phase:
        print(f"  ▶ Next: python train.py --phase {next_phase} "
              f"--resume {args.out_dir}/novamind_{args.phase}_final.pt\n")


if __name__ == "__main__":
    main()
