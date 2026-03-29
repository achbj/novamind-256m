import os
import torch
import argparse
from transformers import AutoTokenizer
from model import NovaMind256M, NovaMindConfig

def main():
    parser = argparse.ArgumentParser(description="Chat with NovaMind-256M")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/novamind_sft_final.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # 1. Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"🖥️  Using device: {device} ({dtype})")

    # 2. Load tokenizer
    print("🔤 Loading Llama-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    # 3. Load checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 4. Build model
    print("🏗️  Building model...")
    config = NovaMindConfig(**ckpt["config"])
    model = NovaMind256M(config).to(device).to(dtype)

    # Handle DataParallel or torch.compile prefix mismatch
    state = ckpt["model_state_dict"]
    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("_orig_mod."):
            k = k[10:]
        clean_state[k] = v
    
    model.load_state_dict(clean_state)
    model.eval()
    print("✅ Model loaded and ready!")

    # 5. Extract stop tokens
    try:
        stop_ids = [tokenizer.encode("</assistant>", add_special_tokens=False)[0]]
    except BaseException:
        stop_ids = None

    # 6. Chat loop
    print("\n" + "="*50)
    print("💬 NovaMind-256M Chat (type 'quit' or 'exit' to stop)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue

            # Format the prompt using the chat template used during training
            prompt = f"<human>\n{user_input}\n</human>\n<assistant>\n"
            
            # Encode
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Generate
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=torch.cuda.is_available()):
                    out_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        stop_token_ids=stop_ids,
                    )
            
            # Decode
            full_response = tokenizer.decode(out_ids[0], skip_special_tokens=False)
            
            # Extract just the assistant's response
            answer = full_response.split("<assistant>\n")[-1].split("</assistant>")[0].strip()
            
            print(f"\nNovaMind: {answer}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break

if __name__ == "__main__":
    main()
