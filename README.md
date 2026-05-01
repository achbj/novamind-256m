# NovaMind-256M: Training a 3.3B Token Conversational LLM on Modal

I've built and trained **NovaMind-256M**, a decoder-only conversational language model with ~252M parameters. This project covers the entire pipeline: from custom architecture design and data preparation to large-scale training on H100 GPUs using [Modal](https://modal.com).

## Why I built this

I wanted to understand how LLMs actually work under the hood, not just call an API. So I read a bunch of papers, picked the best ideas from the top models, and combined them into something I could actually train myself without spending thousands of dollars on cloud compute.

The result is ~256 million parameters. Small by industry standards, but big enough to get real results.

---

![My Model Architecture](image/Architecture.svg)

## 🏗️ The Architecture

I designed NovaMind-256M with modern efficiency in mind, combining established LLM techniques with a few unique twists:

- **Grouped Query Attention (GQA):** 16 Query heads sharing 4 KV heads (4:1 ratio). This dramatically cuts down my KV cache memory footprint during long chats.
- **SwiGLU FFN:** I used SwiGLU instead of standard GELU for better parameter efficiency.
- **Pre-RMSNorm:** Ensures training stays stable across all 24 layers.
- **HiRoPE (Hierarchical Rotary Position Embedding):** This is one of the unique parts of my model. I adapted HiRoPE from code-specific research to work for conversations. I split the head dimensions into:
  - **Local stream (base=10k):** Handles fine-grained context within a single turn.
  - **Global stream (base=500k):** Tracks coarse context across the entire dialogue history.
- **Tag-Aware Loss Curriculum:** Another unique experiment. I don't treat every token equally. I use a dynamic weighting scheme for `<think>`, `<assistant>`, and `<human>` tags that shifts as the model progresses through different training phases.
- **Weight Tying:** I tied the embedding and LM head weights, saving about 33M parameters.

## 🚀 Training Journey

I trained the model on a total of **3.368 Billion tokens** using Modal's H100 GPU infrastructure. I split the training into two distinct phases to optimize for knowledge and personality.

### Phase 1: Knowledge Foundation (Pretraining)

- **The Data:** I used a heavy mix of **Wikipedia EN** for factual depth and **TinyStories** to help a smaller model like this develop better reasoning and narrative coherence.
- **Cost:** Total of **$22.80** (this includes all the CPU-based data preparation and the H100 pretraining run).

![Phase 1 Training Loss](plots/phase1b_loss.png)

### Phase 2: Conversational Polish (SFT)

- **The Data:** A curated instruction set including **Alpaca, OASST1, Dolly, DailyDialog**, and my own custom **Identity Seeds** to anchor the NovaMind persona.
- **Cost:** Only **$1.21** on a Modal H100.

![Phase 2 SFT Loss](plots/phase2_sft_loss.png)

> [!TIP]
> Modal gives $30 of free credit every month. Since the entire training only cost me about $24.01, I basically trained a 3.3B token model for free.

## 💬 Chatting with NovaMind

I wrote a rich terminal interface to interact with the final SFT checkpoint. It supports multi-turn history (keeping the last 6 turns), auto-detects if you're on a Mac (MPS) or GPU (CUDA), and handles stop tokens so the model doesn't hallucinate runaway text.

```bash
# How I start the chat
python chat.py
```

![My Chat Interface](image/chat.png)

## 📁 What's Inside

- `model.py`: My core architecture (`NovaMind256M` & `NovaMindConfig`).
- `train.py`: The heart of the training loop.
- `modal_novamind.py`: My Modal deployment config for remote GPU execution.
- `prepare_data_*.py`: How I tokenized the 3.3B tokens on CPU.
- `chat.py`: The interactive REPL I use for testing.
- `plots/`: Where I keep my training visualizations.

## 🛠️ How to use it

`make sure you have novamind_sft_final.pt` in `checkpoints/` folder`

### Run the REPL

```bash
python chat.py --temp 0.7 --max_tokens 400
```
