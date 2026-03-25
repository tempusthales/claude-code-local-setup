# claude-code-local-setup

Claude Code is a powerful agentic coding tool, but by default every request routes through
Anthropic's API. That means cloud latency, token costs, and a hard dependency on an internet
connection.

This repo documents how to cut that dependency entirely. By pointing a single environment
variable at a local `llama-server` process, Claude Code works the same way it always has,
except the model runs on your own hardware.

## Important: know before you go

Claude Code sends very large prompts. Every request includes your codebase context, tool
definitions, and conversation history, often 10,000 tokens or more per request. Local models
have to process all of that before generating a single token of response.

**This setup works well if you have:**
- 24 GB+ VRAM (NVIDIA) or 32 GB+ unified memory (Apple Silicon)
- A use case that includes local chat inference, offline work, or fine-tuning experiments

**If you have less VRAM than that**, responses will take minutes per request for agentic
coding tasks. The Anthropic API with Claude Sonnet will serve you better for day-to-day
Claude Code use.

You can always switch between local and Anthropic's API with one command:

```fish
set -Ux ANTHROPIC_BASE_URL "http://localhost:8001"   # use local model
set -e ANTHROPIC_BASE_URL                             # switch back to Anthropic
```

## What each guide covers

- Installing build dependencies and (on Linux) CUDA
- Building `llama.cpp` from source with GPU acceleration
- Downloading a quantized GGUF model using `aria2c`
- Running `llama-server` in the background via `tmux`
- Installing and configuring Claude Code to route requests to your local server
- Fixing a KV cache bug in recent Claude Code versions that makes local inference much slower

## Platform guides

- [`arch_setup.md`](./arch_setup.md) — Arch Linux, CachyOS, and Arch-based distros. NVIDIA GPU with CUDA or CPU-only.
- [`macos_setup.md`](./macos_setup.md) — macOS on Apple Silicon. Metal acceleration, no CUDA required.

## Model

Both guides use **Qwen3.5-35B-A3B**, a Mixture of Experts model. Despite the 35B parameter
count, only 3B parameters are active during any single inference pass, which is why it runs
in much less VRAM than a traditional dense 35B model. The Q4_K_XL quant is around 22 GB on
disk.

## Hardware requirements

| | Arch Linux | macOS |
|---|---|---|
| GPU | NVIDIA 24 GB+ VRAM recommended | Apple Silicon 32 GB+ unified memory recommended |
| Minimum (chat only) | 12 GB VRAM + 32 GB RAM | 16 GB unified memory |
| Storage | 25 GB free for model | 25 GB free for model |
