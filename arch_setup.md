# Running Local LLMs with Claude Code on Arch Linux (CachyOS)

> This guide walks you through connecting open-source LLMs to Claude Code entirely locally using `llama.cpp` and quantized GGUFs on Arch Linux / CachyOS. No API bills. No cloud.

---

## Overview

Claude Code normally routes requests to Anthropic's servers. By setting a single environment variable (`ANTHROPIC_BASE_URL`), you redirect it to a local `llama-server` process instead. The guide uses **Qwen3.5-35B-A3B** or **GLM-4.7-Flash**, both of which fit in 24 GB VRAM and handle agentic coding tasks well.

---

## Prerequisites

- Arch Linux, CachyOS, or any Arch-based distro
- NVIDIA GPU with 24 GB VRAM (e.g. RTX 4090, RTX 5070), or CPU-only with enough RAM
- `yay` or `paru` AUR helper installed
- Fish shell, bash, or zsh

---

## Part 1: Build llama.cpp

### 1.1 Install dependencies

```bash
sudo pacman -S --needed base-devel cmake curl git pciutils
```

> `base-devel` covers everything `build-essential` provides on Debian/Ubuntu, plus development headers. `libcurl` ships with the `curl` package on Arch. No separate `-dev` package needed.

### 1.2 Install CUDA (NVIDIA GPU only, skip for CPU-only)

```bash
sudo pacman -S cuda
```

Then add CUDA to your PATH so cmake can find `nvcc`:

**Fish (permanent):**

```fish
fish_add_path /opt/cuda/bin
```

**Bash/Zsh (permanent, add to `~/.bashrc` or `~/.zshrc`):**

```bash
export PATH=/opt/cuda/bin:$PATH
```

Open a new terminal after this, or run `source ~/.bashrc` / `exec fish` to reload your shell before continuing.

> CachyOS ships CUDA in the standard repos. If `sudo pacman -S cuda` does not find it, try `yay -S cuda` from the AUR instead.

### 1.3 Clone and build

**With NVIDIA GPU (CUDA):**

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp/
```

**CPU-only (no GPU):**

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=OFF
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp/
```

> If cmake still cannot find CUDA after installing it, pass the path explicitly:
> ```bash
> cmake llama.cpp -B llama.cpp/build \
>     -DBUILD_SHARED_LIBS=OFF \
>     -DGGML_CUDA=ON \
>     -DCUDAToolkit_ROOT=/opt/cuda
> ```

---

## Part 2: Choose Your Model

**A note on Qwen3.5-35B-A3B and VRAM:** Despite the "35B" in the name, this is a Mixture of Experts (MoE) model. The "A3B" means only 3 billion parameters are active during any single inference pass. The other parameters are organized into expert layers that sit idle until needed. This is why the model fits in far less VRAM than a traditional dense 35B model would require. The Q4_K_XL quant lands around 22 GB on disk, not the 40+ GB you would expect from a dense 35B model.

Pick one of the three options below based on your GPU's VRAM. If you are unsure, check your GPU specs first:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

| VRAM | Recommended option |
|---|---|
| 24 GB+ | Option 1: Qwen3.5-35B-A3B Q4_K_XL (best quality, fully on GPU) |
| 12 GB | Option 2: Qwen3.5-35B-A3B Q4_K_XL split across GPU and RAM (good quality, manageable speed due to MoE) |
| 12 GB (speed priority) | Option 3: Qwen3.5-9B (fits fully on GPU, fastest) |

Install aria2c for fast, reliable downloads:

```bash
sudo pacman -S aria2
```

> aria2c downloads using 16 parallel connections and resumes automatically if interrupted. It is much more reliable than the `hf` CLI for large files.

> **Note:** The download URLs below point to the `unsloth` Hugging Face account, where these optimized GGUF model files are hosted.

### Option 1: Qwen3.5-35B-A3B Q4_K_XL (24 GB+ VRAM, best quality)

The full-quality quant. Fits entirely in VRAM on a 24 GB card. Best output for agentic coding tasks.

```bash
mkdir -p /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF
aria2c -x 16 -s 16 \
    -d /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF \
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
```

### Option 2: Qwen3.5-35B-A3B Q4_K_XL (12 GB VRAM, GPU+RAM split)

Same Q4_K_XL quant as Option 1. On a 12 GB card, llama.cpp offloads the inactive expert layers to RAM while keeping the active compute on the GPU. Because only 3B parameters are active at any time (MoE), the performance penalty from offloading is much smaller than it would be with a dense model of the same size. You need at least 32 GB system RAM. The download command is identical to Option 1 since it is the same file.

```bash
mkdir -p /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF
aria2c -x 16 -s 16 \
    -d /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF \
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
```

> `--fit on` in the `llama-server` command (Part 3) handles the GPU/RAM split automatically. You do not need to configure anything else.

### Option 3: Qwen3.5-9B (12 GB VRAM, fastest)

The 9B model fits entirely within 12 GB VRAM with room to spare for context. Noticeably less capable than the 35B variants but responds quickly, which makes it practical for fast iteration during coding sessions.

```bash
mkdir -p /mnt/llm-storage/llm-models/huggingface/Qwen3.5-9B-GGUF
aria2c -x 16 -s 16 \
    -d /mnt/llm-storage/llm-models/huggingface/Qwen3.5-9B-GGUF \
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf"
```

---

## Part 3: Start llama-server

Run this in a dedicated terminal or inside `tmux`. Keep it running while you use Claude Code.

> **KV cache note:** `q8_0` reduces VRAM usage. Do **not** use `f16` KV cache with Qwen3.5. Multiple reports show accuracy degradation. Use `bf16` if you want full precision and have the VRAM headroom.

> **Disable thinking mode** (faster for agentic coding tasks, add to any command below):
> ```bash
> --chat-template-kwargs "{\"enable_thinking\": false}"
> ```

### Option 1: Qwen3.5-35B-A3B Q4_K_XL (24 GB+ VRAM)

```bash
./llama.cpp/llama-server \
    --model /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias "Qwen3.5-35B-A3B" \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 131072
```

### Option 2: Qwen3.5-35B-A3B Q4_K_XL (12 GB VRAM, GPU+RAM split)

Same file as Option 1. The `--fit on` flag splits layers across GPU and RAM automatically. Because this is a MoE model with only 3B active parameters, the active compute stays on the GPU and the idle expert layers sit in RAM. Speed is slower than a full VRAM setup but more manageable than offloading a dense model of the same size. Reduce `--ctx-size` if you run low on RAM.

```bash
./llama.cpp/llama-server \
    --model /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias "Qwen3.5-35B-A3B" \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 65536
```

### Option 3: Qwen3.5-9B (12 GB VRAM, fastest)

```bash
./llama.cpp/llama-server \
    --model /mnt/llm-storage/llm-models/huggingface/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf \
    --alias "Qwen3.5-9B" \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 131072
```

### GLM-4.7-Flash (alternative, 24 GB VRAM)

An alternative to Qwen3.5 if you want to try a different model architecture. Also fits in 24 GB VRAM.

```bash
mkdir -p /mnt/llm-storage/llm-models/huggingface/GLM-4.7-Flash-GGUF
aria2c -x 16 -s 16 \
    -d /mnt/llm-storage/llm-models/huggingface/GLM-4.7-Flash-GGUF \
    "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
```

```bash
./llama.cpp/llama-server \
    --model /mnt/llm-storage/llm-models/huggingface/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
    --alias "GLM-4.7-Flash" \
    --temp 1.0 \
    --top-p 0.95 \
    --min-p 0.01 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --batch-size 4096 --ubatch-size 1024 \
    --ctx-size 131072
```

---

## Part 4: Install Unsloth Studio (optional chat UI)

Unsloth Studio is a web UI for chatting with and fine-tuning local models. It runs in Docker and works with the GGUF you already downloaded.

### 4.1 Install prerequisites

```bash
sudo pacman -S docker nvidia-container-toolkit
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

### 4.2 Start Unsloth Studio

```bash
docker run -d \
    -e JUPYTER_PASSWORD="yourpassword" \
    -p 8888:8888 -p 8000:8000 -p 2222:22 \
    -v (pwd)/work:/workspace/work \
    -v /mnt/llm-storage:/mnt/llm-storage \
    --gpus all \
    --name unsloth-studio \
    unsloth/unsloth
```

> The `-v /mnt/llm-storage:/mnt/llm-storage` line mounts your model storage into the container so Unsloth Studio can access the GGUF you already downloaded without re-downloading it.

### 4.3 Open the UI

Open `http://localhost:8000` in your browser. You will see the Unsloth Studio welcome screen.

> `http://localhost:8888` is JupyterLab, not the Studio UI. Use port 8000.

### 4.4 Load your model

On the welcome screen, click **Skip to Chat** in the bottom left. This takes you straight to the chat interface, skipping the fine-tuning wizard.

In the model selector at the top, click **Select model**, then search for `Qwen3.5-35B-A3B`. When prompted for a local path, enter:

```
/mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

### 4.5 Managing the container

```bash
docker stop unsloth-studio    # stop without removing
docker start unsloth-studio   # start again later
docker rm unsloth-studio      # remove entirely (stop first)
```

> Unsloth Studio runs its own internal llama-server. If you already have a llama-server running on port 8001 from Part 3, stop it first to avoid conflicts:
> ```bash
> tmux kill-session -t llama
> ```

---

## Part 5: Install Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

> This installs Claude Code to `~/.local/bin` or a similar user-local path. Make sure that path is on your `$PATH`.

Verify it installed:

```bash
claude --version
```

---

## Part 6: Configure Claude Code

### 6.1 Start llama-server

Claude Code needs llama-server running to route requests to your local model. Run it in a tmux session so it does not take over the terminal:

```bash
tmux new-session -d -s llama -c ~/GitHub 'bash -c "./llama.cpp/llama-server \
    --model /mnt/llm-storage/llm-models/huggingface/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-35B-A3B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on --ctx-size 65536 \
    2>&1 | tee /tmp/llama-server.log"'
```

Watch the log until you see `server is listening on http://127.0.0.1:8001`:

```bash
tail -f /tmp/llama-server.log
```

### 6.2 Point Claude Code at your local server

**Fish shell (session only):**

```fish
set -x ANTHROPIC_BASE_URL "http://localhost:8001"
set -x ANTHROPIC_API_KEY "sk-no-key-required"
```

**Fish shell (persistent, survives new terminals):**

```fish
set -Ux ANTHROPIC_BASE_URL "http://localhost:8001"
set -Ux ANTHROPIC_API_KEY "sk-no-key-required"
```

**Bash/Zsh (session only):**

```bash
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

**Bash/Zsh (persistent, add to `~/.bashrc` or `~/.zshrc`):**

```bash
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

> To switch back to Anthropic's real API:
> ```fish
> set -e ANTHROPIC_BASE_URL   # Fish
> ```
> ```bash
> unset ANTHROPIC_BASE_URL    # Bash/Zsh
> ```

### 6.3 Fix the KV Cache invalidation bug (important, 90% speed impact)

Claude Code prepends an attribution header that invalidates the local model's KV cache, making inference much slower. Fix it by editing `~/.claude/settings.json`.

> `export CLAUDE_CODE_ATTRIBUTION_HEADER=0` does **not** work. You must set it inside the settings file.

Create or update `~/.claude/settings.json`:

```json
{
  "promptSuggestionEnabled": false,
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  },
  "attribution": {
    "commit": "",
    "pr": ""
  },
  "plansDirectory": "./plans",
  "prefersReducedMotion": true,
  "terminalProgressBarEnabled": false,
  "effortLevel": "high"
}
```

**Fish one-liner to write the file:**

```fish
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

### 6.4 Handle the sign-in prompt (if it appears)

If Claude Code asks you to log in on first run, add these two keys to `~/.claude.json`:

```json
{
  "hasCompletedOnboarding": true,
  "primaryApiKey": "sk-dummy-key"
}
```

---

## Part 7: Run Claude Code

Navigate to your project directory, then launch with the model alias:

```bash
cd ~/your-project
claude --model Qwen3.5-35B-A3B
```

**Skip permission prompts (full autonomous mode, use carefully):**

```bash
claude --model Qwen3.5-35B-A3B --dangerously-skip-permissions
```

> This lets Claude Code execute commands, write files, and run code without asking. Only use it in a sandboxed project directory.

### Example prompt to test the setup

```
You can only work in the cwd project/. Do not search for CLAUDE.md - this is it.
Create a Python virtual environment via `python -m venv venv` then activate it.
Install dependencies and write a simple script that fetches and prints the top 5
results from the Hacker News API. You have access to 1 GPU.
```

---

## Part 8: VS Code / Cursor Integration

Install the Claude Code extension:

- **VS Code:** [marketplace.visualstudio.com/items?itemName=anthropic.claude-code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)
- **Cursor:** `cursor:extension/anthropic.claude-code`
- Or press `Ctrl+Shift+X`, search **Claude Code**, and click **Install**

The extension picks up `ANTHROPIC_BASE_URL` from your environment. If it still prompts for sign-in, add `"claudeCode.disableLoginPrompt": true` to your VS Code `settings.json`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unable to connect to API (ConnectionRefused)` | `llama-server` is not running, or wrong port. Check the server terminal. |
| Claude Code is very slow (90% slower) | `CLAUDE_CODE_ATTRIBUTION_HEADER` not set in `~/.claude/settings.json`. See Part 5.2. |
| `missing ANTHROPIC_API_KEY` error | `set -x ANTHROPIC_API_KEY "sk-no-key-required"` |
| Sign-in loop on first launch | Add `hasCompletedOnboarding` and `primaryApiKey` to `~/.claude.json` |
| Model output loops or is garbled | Update llama.cpp (a bug in the KV cache calculation was fixed; re-download GGUFs too) |
| Out of VRAM | Reduce `--ctx-size` (e.g. `65536` instead of `131072`) |
| Download stalls or is slow | Use `aria2c -x 16 -s 16 -d <dir> <url>` for parallel chunk downloading. If the URL has expired, grab a fresh one from the Hugging Face model page. |

---

## Running Multiple Models

You can run multiple models on different ports simultaneously and switch between them:

```bash
# Terminal 1: Qwen3.5 on port 8001
./llama.cpp/llama-server --model Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --alias "Qwen3.5-35B-A3B" --port 8001 ...

# Terminal 2: GLM-4.7-Flash on port 8002
./llama.cpp/llama-server --model GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf --alias "GLM-4.7-Flash" --port 8002 ...
```

Then point Claude Code at whichever port you want:

```fish
# Fish: switch quickly
set -x ANTHROPIC_BASE_URL "http://localhost:8001"   # switch to Qwen3.5
set -x ANTHROPIC_BASE_URL "http://localhost:8002"   # switch to GLM
```

---

## References

- [Unsloth Claude Code Guide](https://unsloth.ai/docs/basics/claude-code)
- [Unsloth Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)
- [Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [GLM-4.7-Flash Guide](https://unsloth.ai/docs/models/glm-4.7-flash)
- [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Claude Code docs](https://code.claude.com/docs)
