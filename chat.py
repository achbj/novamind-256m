"""
NovaMind Chat — Interactive terminal interface for novamind_sft_final.pt
========================================================================
Usage:
    python chat.py                                    # auto-finds checkpoint
    python chat.py --checkpoint checkpoints/novamind_sft_final.pt
    python chat.py --temp 0.8 --top_p 0.85 --max_tokens 512

Commands (type during chat):
    /exit  or  /quit    — quit
    /reset              — clear conversation history
    /info               — show model info & generation settings
    /help               — show this list
"""

import sys
import os
import argparse
import textwrap
import time

# ── Silence noisy HF / tokenizer logs ──────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch

# ── ANSI colour helpers ─────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"

def _c(text, *codes):
    return "".join(codes) + str(text) + C.RESET

def banner():
    lines = [
        "",
        _c("╔══════════════════════════════════════════════════╗", C.CYAN, C.BOLD),
        _c("║", C.CYAN, C.BOLD) + _c("         NovaMind-256M  ·  SFT Chat             ", C.WHITE, C.BOLD) + _c("║", C.CYAN, C.BOLD),
        _c("║", C.CYAN, C.BOLD) + _c("  Helpful · Concise · Identity-anchored AI      ", C.GREY)          + _c("║", C.CYAN, C.BOLD),
        _c("╚══════════════════════════════════════════════════╝", C.CYAN, C.BOLD),
        "",
        _c("  Type your message and press Enter.", C.DIM),
        _c("  Commands: /reset  /info  /help  /exit", C.DIM),
        "",
    ]
    print("\n".join(lines))

def info_block(checkpoint_path, device, config, tok_name, args):
    print()
    print(_c("  ── Model Info ─────────────────────────────────────", C.CYAN))
    print(f"  Checkpoint : {_c(checkpoint_path, C.YELLOW)}")
    print(f"  Tokenizer  : {_c(tok_name, C.YELLOW)}")
    print(f"  Device     : {_c(device, C.GREEN)}")
    print(f"  Params     : {_c(f'd_model={config.d_model}  n_layers={config.n_layers}  n_heads={config.n_heads}', C.WHITE)}")
    print(f"  Vocab size : {_c(config.vocab_size, C.WHITE)}")
    print()
    print(_c("  ── Generation Settings ────────────────────────────", C.CYAN))
    print(f"  temperature       : {_c(args.temp, C.YELLOW)}")
    print(f"  top_k             : {_c(args.top_k, C.YELLOW)}")
    print(f"  top_p             : {_c(args.top_p, C.YELLOW)}")
    print(f"  repetition_penalty: {_c(args.rep_penalty, C.YELLOW)}")
    print(f"  max_new_tokens    : {_c(args.max_tokens, C.YELLOW)}")
    print(f"  multi-turn history: {_c('on', C.GREEN)}")
    print()

def wrap_text(text, width=72, indent="    "):
    """Wrap assistant text for cleaner terminal display."""
    paragraphs = text.split("\n")
    wrapped = []
    for para in paragraphs:
        if para.strip() == "":
            wrapped.append("")
        else:
            wrapped.append(textwrap.fill(para, width=width, initial_indent=indent,
                                         subsequent_indent=indent,
                                         break_long_words=False, break_on_hyphens=False))
    return "\n".join(wrapped)


# ── Chat format (must match prepare_data_phase2.py exactly) ────────────────────
SYS_PROMPT = (
    "You are NovaMind, a helpful and concise AI assistant. "
    "Answer questions directly and clearly. Do not ask questions back to the user."
)

def build_prompt(history: list[dict], new_user_msg: str, system: str = SYS_PROMPT) -> str:
    """
    Build the full prompt string from conversation history + new user message.
    Format (trained on):
        <system>\\n{sys}\\n</system>\\n
        <human>\\n{user}\\n</human>\\n
        <assistant>\\n{reply}\\n</assistant>\\n
        ...
        <human>\\n{new_user_msg}\\n</human>\\n
        <assistant>\\n          ← model generates from here
    """
    prompt = f"<system>\n{system}\n</system>\n"
    for turn in history:
        prompt += f"<human>\n{turn['user'].strip()}\n</human>\n"
        prompt += f"<assistant>\n{turn['assistant'].strip()}\n</assistant>\n"
    prompt += f"<human>\n{new_user_msg.strip()}\n</human>\n<assistant>\n"
    return prompt


def extract_response(full_text: str, prompt: str) -> str:
    """
    Strip the prompt prefix and extract the assistant's reply,
    stopping at </assistant>.
    """
    # Remove the prompt portion
    if full_text.startswith(prompt):
        response = full_text[len(prompt):]
    else:
        # Fallback: grab text after last <assistant>\n
        parts = full_text.split("<assistant>\n")
        response = parts[-1] if parts else full_text

    # Cut at </assistant> stop tag
    if "</assistant>" in response:
        response = response.split("</assistant>")[0]

    # Also stop at <human> (shouldn't happen but safety net)
    if "<human>" in response:
        response = response.split("<human>")[0]

    return response.strip()


# ── Model loading ───────────────────────────────────────────────────────────────
def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    print(_c(f"\n  Loading checkpoint: {checkpoint_path}", C.DIM))

    # Add workspace to path so model.py is importable
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import NovaMind256M, NovaMindConfig  # noqa: E402

    # ── Load checkpoint ──────────────────────────────────────────────────────
    t0 = time.time()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config (saved as dict in ckpt["config"])
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        config = NovaMindConfig.from_dict(ckpt["config"])
    else:
        print(_c("  ⚠  No config in checkpoint — using defaults.", C.YELLOW))
        config = NovaMindConfig()

    # ── Build model ──────────────────────────────────────────────────────────
    model = NovaMind256M(config)
    state = ckpt["model_state_dict"]

    # Strip DataParallel "module." prefix if present
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    # Strip torch.compile "_orig_mod." prefix if present (saved under compile())
    if all(k.startswith("_orig_mod.") for k in state):
        state = {k[10:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    elapsed = time.time() - t0
    phase  = ckpt.get("phase", "unknown")
    step   = ckpt.get("step", "?")
    tokens = ckpt.get("tokens_seen", 0)
    loss   = ckpt.get("train_loss", "?")
    print(_c(f"  ✅  Model loaded in {elapsed:.1f}s", C.GREEN))
    print(_c(f"      phase={phase}  step={step:,}  "
             f"tokens={tokens/1e9:.2f}B  last_loss={loss}", C.DIM)
          if isinstance(step, int) else
          _c(f"      phase={phase}  step={step}", C.DIM))

    # ── Tokenizer ────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tok_name = None
    tokenizer = None
    for name in ["NousResearch/Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf", "gpt2"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            tok_name = name
            break
        except Exception:
            pass
    if tokenizer is None:
        raise RuntimeError("No tokenizer available. Run: pip install transformers")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(_c(f"  ✅  Tokenizer: {tok_name}  (vocab={tokenizer.vocab_size:,})", C.GREEN))
    return model, tokenizer, config, tok_name


# ── Generation ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: torch.device, args) -> tuple[str, float]:
    """Generate a response and return (response_text, elapsed_seconds)."""
    # Resolve stop token ids
    stop_ids = []
    for tag in ["</assistant>", "</s>"]:
        try:
            ids = tokenizer.encode(tag, add_special_tokens=False)
            stop_ids.extend(ids)
        except Exception:
            pass
    stop_ids.append(tokenizer.eos_token_id)
    stop_ids = list(set(i for i in stop_ids if i is not None))

    input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                 add_special_tokens=False).to(device)

    t0 = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        stop_token_ids=stop_ids,
    )
    elapsed = time.time() - t0

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    response  = extract_response(full_text, prompt)
    response = response.replace("</ass", "").strip()

    return response, elapsed, output_ids.shape[1] - input_ids.shape[1]


# ── CLI ─────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Chat with NovaMind SFT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", "-c",
                   default="checkpoints/novamind_sft_final.pt",
                   help="Path to .pt checkpoint (default: checkpoints/novamind_sft_final.pt)")
    p.add_argument("--temp",       type=float, default=0.7,  help="Sampling temperature (default: 0.7)")
    p.add_argument("--top_k",      type=int,   default=50,   help="Top-k sampling (default: 50)")
    p.add_argument("--top_p",      type=float, default=0.9,  help="Nucleus sampling p (default: 0.9)")
    p.add_argument("--rep_penalty",type=float, default=1.2,  help="Repetition penalty (default: 1.2)")
    p.add_argument("--max_tokens", type=int,   default=400,  help="Max new tokens per reply (default: 400)")
    p.add_argument("--system",     type=str,   default=SYS_PROMPT,
                   help="Override system prompt")
    p.add_argument("--no_history", action="store_true",
                   help="Disable multi-turn: each message is a fresh prompt")
    p.add_argument("--cpu",        action="store_true",
                   help="Force CPU inference (slow but works anywhere)")
    return p.parse_args()


# ── Main REPL ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        # Search relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path  = os.path.join(script_dir, ckpt_path)

    if not os.path.exists(ckpt_path):
        print(_c(f"\n  ❌  Checkpoint not found: {ckpt_path}", C.RED))
        print(_c("  Run: modal volume get novamind-vol /checkpoints/novamind_sft_final.pt "
                 "./checkpoints/novamind_sft_final.pt\n", C.DIM))
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    banner()
    model, tokenizer, config, tok_name = load_model_and_tokenizer(ckpt_path, device)
    info_block(ckpt_path, str(device), config, tok_name, args)

    # ── Conversation state ────────────────────────────────────────────────────
    history: list[dict] = []   # [{"user": ..., "assistant": ...}, ...]

    print(_c("  ━" * 26, C.CYAN))
    print()

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        # Prompt
        try:
            user_input = input(_c("  You  › ", C.GREEN, C.BOLD)).strip()
        except (KeyboardInterrupt, EOFError):
            print(_c("\n\n  Goodbye! 👋\n", C.CYAN))
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        cmd = user_input.lower()

        if cmd in ("/exit", "/quit", "exit", "quit"):
            print(_c("\n  Goodbye! 👋\n", C.CYAN))
            break

        if cmd == "/reset":
            history.clear()
            print(_c("\n  ✓  Conversation history cleared.\n", C.YELLOW))
            continue

        if cmd == "/info":
            info_block(ckpt_path, str(device), config, tok_name, args)
            continue

        if cmd == "/help":
            print(_c("""
  Commands:
    /reset        — clear conversation history
    /info         — show model and generation settings
    /help         — show this message
    /exit, /quit  — quit
""", C.DIM))
            continue

        # ── Generate ──────────────────────────────────────────────────────────
        hist_for_prompt = [] if args.no_history else history
        prompt = build_prompt(hist_for_prompt, user_input, system=args.system)

        print(_c("\n  NovaMind › ", C.CYAN, C.BOLD), end="", flush=True)

        try:
            response, elapsed, n_new = generate(model, tokenizer, prompt, device, args)
        except Exception as e:
            print(_c(f"\n  ❌  Generation error: {e}\n", C.RED))
            continue

        if not response:
            response = "[No response generated — try a different prompt]"

        # Print response (wrapped for readability)
        print()
        print(wrap_text(response))

        # Stats footer
        tok_per_sec = n_new / elapsed if elapsed > 0 else 0
        print(_c(f"\n  {n_new} tokens  ·  {elapsed:.1f}s  ·  {tok_per_sec:.0f} tok/s", C.GREY))
        print()

        # Update history (only if multi-turn is on)
        if not args.no_history:
            history.append({"user": user_input, "assistant": response})

            # Trim history to avoid exceeding context length.
            # Keep at most 6 turns (12 halves) — generous for 2048 context.
            if len(history) > 6:
                history = history[-6:]


if __name__ == "__main__":
    main()
