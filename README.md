# claude-code-local-setup
Claude Code is a powerful agentic coding tool, but by default every request routes through Anthropic's API. That means cloud latency, token costs, and a hard dependency on an internet connection.
This repo documents how to cut that dependency entirely. By pointing a single environment variable at a local llama-server process, Claude Code works the same way it always has — except the model runs on your own hardware. No API key required. No usage bill at the end of the month.
The guides cover building llama.cpp from source, downloading quantized GGUF models from Hugging Face, configuring Claude Code to talk to your local server, and fixing a KV cache bug that ships with recent Claude Code versions that causes local inference to run 90% slower than it should.
Two platform guides are included:

arch-linux.md — Arch Linux, CachyOS, and Arch-based distros with NVIDIA GPU or CPU-only setups
macos.md — macOS on Apple Silicon and Intel, using Metal acceleration

The examples use Qwen3.5-35B-A3B and GLM-4.7-Flash, both of which fit comfortably in 24 GB of VRAM or unified memory. Any model with an OpenAI-compatible API endpoint works the same way — swap the model path and alias and the rest of the setup is identical.

