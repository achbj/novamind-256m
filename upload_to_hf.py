"""
Upload NovaMind-256M to Hugging Face Hub
=========================================
Prepares a clean HF-compatible repository and pushes it.

What this does:
  1. Loads the trained checkpoint (cot_sft_final.pt by default)
  2. Saves the model weights as safetensors (the modern HF standard)
  3. Writes config.json, tokenizer_config.json, generation_config.json
  4. Copies model.py + chat.py + README.md into the upload folder
  5. Pushes everything to your HF repo

Usage:
  pip install huggingface_hub safetensors
  huggingface-cli login
  python upload_to_hf.py --repo YOUR_USERNAME/novamind-256m
"""

import os
import json
import shutil
import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="./checkpoints/novamind_cot_sft_final.pt",
        help="Path to trained checkpoint (.pt file)",
    )
    p.add_argument(
        "--repo",
        required=True,
        help="HF repo id, e.g. your-username/novamind-256m",
    )
    p.add_argument(
        "--out_dir",
        default="./hf_upload",
        help="Local staging folder before push",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Make the HF repo private",
    )
    return p.parse_args()


def save_safetensors(state_dict: dict, path: str):
    """Save weights as safetensors (preferred by HF — faster, safer than pickle)."""
    try:
        from safetensors.torch import save_file
        save_file(state_dict, path)
        print(f"  ✅ Saved safetensors: {path}")
    except ImportError:
        # Fallback: save as pytorch_model.bin
        pt_path = path.replace(".safetensors", ".bin")
        torch.save(state_dict, pt_path)
        print(f"  ⚠️  safetensors not installed — saved as {pt_path}")
        print(f"       (run `pip install safetensors` for the better format)")


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    raw_config = ckpt["config"]       # dict with all NovaMindConfig fields
    state_dict = ckpt["model_state_dict"]

    # Strip DataParallel / torch.compile prefixes
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):   k = k[7:]
        if k.startswith("_orig_mod."): k = k[10:]
        clean_state[k] = v

    # Fix tied weights sharing memory (safetensors strict constraint)
    if "lm_head.weight" in clean_state and "token_emb.weight" in clean_state:
        # If they are exactly the same tensor object in memory
        if clean_state["lm_head.weight"].data_ptr() == clean_state["token_emb.weight"].data_ptr():
            # Clone to separate memory locations
            clean_state["lm_head.weight"] = clean_state["lm_head.weight"].clone()

    print(f"  Checkpoint phase : {ckpt.get('phase', 'unknown')}")
    print(f"  Tokens seen      : {ckpt.get('tokens_seen', 0)/1e9:.2f}B")
    print(f"  Last train loss  : {ckpt.get('train_loss', '?')}")
    print(f"  Parameters (keys): {len(clean_state)}")

    # ── 2. Save model weights as safetensors ──────────────────────────────────
    print("\n💾 Saving model weights...")
    save_safetensors(clean_state, str(out / "model.safetensors"))

    # ── 3. Write config.json ──────────────────────────────────────────────────
    print("\n📝 Writing config.json...")
    hf_config = {
        # HF standard fields
        "model_type": "novamind",
        "architectures": ["NovaMind256M"],
        # NovaMind architecture fields (mirrors NovaMindConfig)
        "vocab_size":       raw_config.get("vocab_size", 32000),
        "d_model":          raw_config.get("d_model", 1024),
        "n_heads":          raw_config.get("n_heads", 16),
        "n_kv_heads":       raw_config.get("n_kv_heads", 4),
        "n_layers":         raw_config.get("n_layers", 24),
        "ff_dim":           raw_config.get("ff_dim", 2304),
        "max_seq_len":      raw_config.get("max_seq_len", 2048),
        "rope_local_base":  raw_config.get("rope_local_base", 10000.0),
        "rope_global_base": raw_config.get("rope_global_base", 500000.0),
        "dropout":          raw_config.get("dropout", 0.0),
        "attn_dropout":     raw_config.get("attn_dropout", 0.0),
        "init_std":         raw_config.get("init_std", 0.02),
        # Training metadata
        "torch_dtype": "bfloat16",
        "transformers_version": "4.x",
        # Chat template info
        "chat_template": "<human>\n{user}\n</human>\n<assistant>\n",
        "eos_token_id": 2,
        "bos_token_id": 1,
        "pad_token_id": 2,
    }
    with open(out / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"  ✅ config.json written")

    # ── 4. Write generation_config.json ───────────────────────────────────────
    gen_config = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "eos_token_id": 2,
        "bos_token_id": 1,
    }
    with open(out / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)
    print(f"  ✅ generation_config.json written")

    # ── 5. Write tokenizer config (points to LLaMA-2 tokenizer) ──────────────
    print("\n🔤 Writing tokenizer config...")
    tok_config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": 2048,
        "padding_side": "right",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        # Important note for users
        "_note": "This model uses the LLaMA-2 tokenizer (32K vocab). Load with: AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')",
    }
    with open(out / "tokenizer_config.json", "w") as f:
        json.dump(tok_config, f, indent=2)
    print(f"  ✅ tokenizer_config.json written")

    # ── 6. Copy source files ──────────────────────────────────────────────────
    print("\n📁 Copying source files...")
    files_to_copy = [
        ("model.py",        "model.py"),
        ("chat.py",         "chat.py"),
        ("requirements.txt","requirements.txt"),
        ("README.md",       "README.md"),
    ]
    for src, dst in files_to_copy:
        if Path(src).exists():
            shutil.copy2(src, out / dst)
            print(f"  ✅ {src}")
        else:
            print(f"  ⚠️  {src} not found — skipping")

    # Copy images folder
    if Path("images").exists():
        shutil.copytree("images", out / "images", dirs_exist_ok=True)
        print(f"  ✅ images/")

    # ── 7. Write a minimal .gitattributes for LFS tracking ───────────────────
    # HF uses Git LFS for large files. The hub library handles this automatically,
    # but it's good practice to declare it.
    gitattributes = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n*.bin filter=lfs diff=lfs merge=lfs -text\n"
    with open(out / ".gitattributes", "w") as f:
        f.write(gitattributes)
    print(f"  ✅ .gitattributes (LFS tracking)")

    # ── 8. Push to Hub ────────────────────────────────────────────────────────
    print(f"\n🚀 Pushing to Hugging Face Hub: {args.repo}")
    print(f"   Private: {args.private}")
    print(f"   Staging folder: {out}")

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"  ✅ Repo ready: https://huggingface.co/{args.repo}")

        # Upload all files from the staging folder
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo,
            repo_type="model",
            commit_message="Upload NovaMind-256M trained model",
        )
        print(f"\n✅ Upload complete!")
        print(f"   👉 https://huggingface.co/{args.repo}")

    except ImportError:
        print("\n❌ huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
        print(f"\n   Your files are staged in: {out}")
        print(f"   You can also drag-and-drop them at: https://huggingface.co/new")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"   Files are staged in: {out}")
        print(f"   Try: huggingface-cli login  (then re-run this script)")


if __name__ == "__main__":
    main()
