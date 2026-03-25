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

- Building `llama.cpp` from source with GPU acceleration
- Downloading quantized GGUF models using `aria2c` for reliable large file transfers
- Choosing the right model size for your hardware (three options with trade-off explanations)
- Running `llama-server` in the background via `tmux` so it does not take over your terminal
- Installing Unsloth Studio as a local web chat UI via Docker
- Installing and configuring Claude Code to route requests to your local server
- Fixing a KV cache bug in recent Claude Code versions that causes local inference to run much slower than it should

## Platform guides

- [`arch_setup.md`](./arch_setup.md) — Arch Linux, CachyOS, and Arch-based distros with NVIDIA GPU (CUDA) or CPU-only setups
- [`macos_setup.md`](./macos_setup.md) — macOS on Apple Silicon using Metal acceleration

## Model used

The guides use **Qwen3.5-35B-A3B**, a Mixture of Experts model. Despite the 35B parameter
count, only 3B parameters are active during any inference pass, which is why it fits in
much less VRAM than a traditional dense 35B model. The Q4_K_XL quant lands around 22 GB
on disk and runs well on a 24 GB GPU or unified memory Mac. For 12 GB GPUs and 16 GB Macs,
the guides include options for running the same model with GPU and RAM sharing, or switching
to the smaller 9B variant that fits fully on device.

## Requirements at a glance

| | Arch Linux | macOS |
|---|---|---|
| GPU | NVIDIA (CUDA) or CPU-only | Apple Silicon (Metal, automatic) |
| Recommended VRAM / unified memory | 24 GB | 24 GB |
| Minimum | 12 GB VRAM + 32 GB RAM | 16 GB unified memory |
| Docker | `pacman -S docker` + nvidia-container-toolkit | Docker Desktop |
