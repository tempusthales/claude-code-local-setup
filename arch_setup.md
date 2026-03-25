# Running Local LLMs with Claude Code on Arch Linux (CachyOS)

> Connect Claude Code to a local LLM using `llama.cpp`. No API bills. No cloud. No Anthropic account required.

---

## Read this first

This setup works, but there is an important limitation to understand before you invest time in it.

**Claude Code sends very large prompts.** Every request includes your codebase context, tool definitions, and conversation history, often 10,000-20,000 tokens or more. Local models have to process all of that on your hardware before generating a single token of response. On a consumer GPU with 12-16 GB VRAM, that means minutes per request, not seconds.

**Who this works well for:**
- 24 GB+ VRAM (RTX 3090, 4090, 5090) where the 35B model fits fully on GPU
- Apple Silicon with 32 GB+ unified memory
- Anyone who wants local chat inference or fine-tuning experiments
- Testing prompts offline before using the real API

**Who should just use the Anthropic API:**
- Anyone with less than 24 GB VRAM
- Anyone who needs Claude Code to be responsive for day-to-day coding

You can always switch between local and Anthropic's API instantly with one command:

```fish
set -Ux ANTHROPIC_BASE_URL "http://localhost:8001"   # use local model
set -e ANTHROPIC_BASE_URL                             # switch back to Anthropic
```

---

## How it works

```
Claude Code  →  llama-server (port 8001)  →  GGUF model on disk
```

---

## Prerequisites

- Arch Linux, CachyOS, or any Arch-based distro
- NVIDIA GPU (24 GB+ VRAM recommended for usable performance)
- Fish, bash, or zsh

---

## Part 1: Install dependencies

```bash
sudo pacman -S --needed base-devel cmake curl git pciutils aria2 tmux
```

---

## Part 2: Install CUDA (NVIDIA GPU only)

```bash
sudo pacman -S cuda
```

Add CUDA to your PATH so cmake can find `nvcc`:

**Fish (permanent):**
```fish
fish_add_path /opt/cuda/bin
exec fish
```

**Bash/Zsh (permanent):**
```bash
export PATH=/opt/cuda/bin:$PATH
```

Verify it worked:
```bash
nvcc --version
```

> If `sudo pacman -S cuda` does not find it, try `yay -S cuda` instead.

---

## Part 3: Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp/
```

**CPU-only (no GPU):** replace `-DGGML_CUDA=ON` with `-DGGML_CUDA=OFF`.

> If cmake cannot find CUDA, add `-DCUDAToolkit_ROOT=/opt/cuda` to the cmake command.

---

## Part 4: Download a model

The guide uses **Qwen3.5-35B-A3B**, a Mixture of Experts (MoE) model. Despite the "35B" name, only 3B parameters are active during any single inference pass. The Q4_K_XL quant is around 22 GB on disk.

Check your GPU VRAM:
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

| VRAM | Expected performance |
|---|---|
| 24 GB+ | Good. Model fits fully on GPU. Responses in seconds. |
| 12-16 GB | Poor for agentic coding. Minutes per request due to RAM offloading. Fine for chat. |
| CPU only | Very slow. Not recommended for Claude Code. |

> The `unsloth/` prefix in the URLs below is a Hugging Face account name, not software you need to install.

### Qwen3.5-35B-A3B (24 GB+ VRAM recommended)

```bash
mkdir -p /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF
aria2c -x 16 -s 16 \
    -d /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF \
    -o Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
```

### Qwen3.5-9B (12 GB VRAM, fits on GPU but still slow for agentic use)

```bash
mkdir -p /mnt/llm-storage/llm-models/huggingface/Qwen3.5-9B-GGUF
aria2c -x 16 -s 16 \
    -d /mnt/llm-storage/llm-models/huggingface/Qwen3.5-9B-GGUF \
    -o Qwen3.5-9B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf"
```

> aria2c resumes automatically if the download is interrupted. Re-run the same command to resume.

---

## Part 5: Start llama-server

Run llama-server in a tmux session so it stays running in the background. Replace `YOUR_USERNAME` with your actual username.

### Qwen3.5-35B-A3B (24 GB+ VRAM)

```bash
tmux new-session -d -s llama -c /home/YOUR_USERNAME/GitHub \
'bash -c "./llama.cpp/llama-server \
    --model /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-35B-A3B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on --ctx-size 131072 \
    2>&1 | tee /tmp/llama-server.log"'
```

### Qwen3.5-9B (12 GB VRAM)

```bash
tmux new-session -d -s llama -c /home/YOUR_USERNAME/GitHub \
'bash -c "./llama.cpp/llama-server \
    --model /mnt/llm-storage/llm-models/huggingface/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-9B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on --ctx-size 131072 \
    2>&1 | tee /tmp/llama-server.log"'
```

Watch the log until you see `server is listening on http://127.0.0.1:8001`:

```bash
tail -f /tmp/llama-server.log
```

Press `Ctrl+C` to stop watching the log. The server keeps running.

**tmux quick reference:**
- Reattach to the server session: `tmux attach -t llama`
- Detach without stopping it: `Ctrl+B` then `D`
- Stop the server: `tmux kill-session -t llama`

---

## Part 6: Install Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Verify:
```bash
claude --version
```

---

## Part 7: Configure Claude Code

### 7.1 Point Claude Code at your local server

**Fish (persistent):**
```fish
set -Ux ANTHROPIC_BASE_URL "http://localhost:8001"
set -Ux ANTHROPIC_API_KEY "sk-no-key-required"
```

**Bash/Zsh (persistent, add to `~/.bashrc` or `~/.zshrc`):**
```bash
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

> To switch back to Anthropic's real API: `set -e ANTHROPIC_BASE_URL` (Fish) or `unset ANTHROPIC_BASE_URL` (Bash/Zsh).

### 7.2 Fix the KV cache bug

Claude Code prepends a header that breaks the local model's KV cache and makes inference much slower. This must be set inside the settings file, not as an environment variable.

**Fish:**
```fish
mkdir -p ~/.claude
echo '{
  "promptSuggestionEnabled": false,
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  },
  "attribution": { "commit": "", "pr": "" },
  "plansDirectory": "./plans",
  "prefersReducedMotion": true,
  "terminalProgressBarEnabled": false,
  "effortLevel": "high"
}' > ~/.claude/settings.json
```

**Bash/Zsh:**
```bash
mkdir -p ~/.claude && cat > ~/.claude/settings.json << 'EOF'
{
  "promptSuggestionEnabled": false,
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  },
  "attribution": { "commit": "", "pr": "" },
  "plansDirectory": "./plans",
  "prefersReducedMotion": true,
  "terminalProgressBarEnabled": false,
  "effortLevel": "high"
}
EOF
```

### 7.3 Handle the sign-in prompt (if it appears)

If Claude Code asks you to log in on first run, add these two keys to `~/.claude.json`:

```json
{
  "hasCompletedOnboarding": true,
  "primaryApiKey": "sk-dummy-key"
}
```

---

## Part 8: Run Claude Code

```bash
cd ~/your-project
claude --model Qwen3.5-35B-A3B
```

If prompted whether to use an existing API key, select **No**.

**Skip permission prompts (autonomous mode, use carefully):**
```bash
claude --model Qwen3.5-35B-A3B --dangerously-skip-permissions
```

---

## Part 9: VS Code / Cursor Integration

Install the Claude Code extension:

- Press `Ctrl+Shift+X`, search **Claude Code**, click **Install**
- Direct link: [marketplace.visualstudio.com/items?itemName=anthropic.claude-code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)

The extension picks up `ANTHROPIC_BASE_URL` from your environment automatically. If it still prompts for sign-in, add `"claudeCode.disableLoginPrompt": true` to your VS Code `settings.json`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unable to connect to API` | llama-server is not running. Check with `tail -20 /tmp/llama-server.log` |
| Slow responses | Expected on 12-16 GB VRAM. See the note at the top of this guide. |
| `missing ANTHROPIC_API_KEY` | `set -x ANTHROPIC_API_KEY "sk-no-key-required"` |
| Sign-in loop on first launch | Add `hasCompletedOnboarding` and `primaryApiKey` to `~/.claude.json` |
| Model output is garbled | Update llama.cpp and re-download the GGUF |
| Out of VRAM | Reduce `--ctx-size` (try halving it) |
| Download interrupted | Re-run the same `aria2c` command, it resumes automatically |
| cmake cannot find CUDA | Add `-DCUDAToolkit_ROOT=/opt/cuda` to the cmake command |
| duplicate session: llama | Run `tmux kill-session -t llama` first, then retry |

---

## References

- [Unsloth Claude Code Guide](https://unsloth.ai/docs/basics/claude-code)
- [Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Claude Code docs](https://code.claude.com/docs)
