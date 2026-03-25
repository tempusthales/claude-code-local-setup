# claude-code-local-setup

Claude Code is a powerful agentic coding tool, but by default every request routes through
Anthropic's API. That means cloud latency, token costs, and a hard dependency on an internet
connection.

This repo documents how to cut that dependency entirely. By pointing a single environment
variable at a local `llama-server` process, Claude Code works the same way it always has,
except the model runs on your own hardware. No API key required. No usage bill at the end
of the month.

## What each guide covers

Both guides walk through the full setup end to end:

- Installing build dependencies and (on Linux) CUDA
- Building `llama.cpp` from source with GPU acceleration
- Downloading a quantized GGUF model using `aria2c` for reliable large file transfers
- Choosing the right model size for your hardware, with three options explained
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
| GPU | NVIDIA 12 GB+ VRAM (CUDA) | Apple Silicon (Metal, automatic) |
| Recommended | 24 GB VRAM | 24 GB unified memory |
| Minimum | 12 GB VRAM + 32 GB RAM | 16 GB unified memory |
| Storage | 25 GB free for model | 25 GB free for model |
