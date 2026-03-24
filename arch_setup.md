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

## Part 2: Download a Model

Install the Hugging Face download tools:

```bash
pip install huggingface_hub hf_transfer
```

> **Note:** The `unsloth/` prefix in the download paths below is a Hugging Face account name. That is where these optimized GGUF model files are hosted. It is not something you need to install or sign up for.

### Option A: Qwen3.5-35B-A3B (recommended for agentic coding)

Uses the **UD-Q4_K_XL** quant, a dynamically quantized GGUF with the best accuracy/size balance. Fits ~24 GB VRAM.

```bash
hf download unsloth/Qwen3.5-35B-A3B-GGUF \
    --local-dir unsloth/Qwen3.5-35B-A3B-GGUF \
    --include "*UD-Q4_K_XL*"
```

> Want a smarter but slower model? Use `Qwen3.5-27B` instead. Drop `35B-A3B` in the repo name and include filter above. It runs at roughly half the token speed.

### Option B: GLM-4.7-Flash (fast, also 24 GB)

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id = "unsloth/GLM-4.7-Flash-GGUF",
    local_dir = "unsloth/GLM-4.7-Flash-GGUF",
    allow_patterns = ["*UD-Q4_K_XL*"],
)
```

---

## Part 3: Start llama-server

Run this in a dedicated terminal or inside `tmux`. Keep it running while you use Claude Code.

### Qwen3.5-35B-A3B

```bash
./llama.cpp/llama-server \
    --model unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
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

> **KV cache note:** `q8_0` reduces VRAM usage. Do **not** use `f16` KV cache with Qwen3.5. Multiple reports show accuracy degradation. Use `bf16` if you want full precision and have the VRAM headroom.

> **Disable thinking mode** (faster for agentic coding tasks):
> ```bash
> --chat-template-kwargs "{\"enable_thinking\": false}"
> ```

### GLM-4.7-Flash

```bash
./llama.cpp/llama-server \
    --model unsloth/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
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

## Part 4: Install Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

> This installs Claude Code to `~/.local/bin` or a similar user-local path. Make sure that path is on your `$PATH`.

---

## Part 5: Configure Claude Code

### 5.1 Point Claude Code at your local server

You need to set `ANTHROPIC_BASE_URL` so Claude Code talks to `llama-server` instead of Anthropic's API.

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

> If you ever need to switch back to Anthropic's real API, unset the variable:
> ```fish
> set -e ANTHROPIC_BASE_URL   # Fish
> ```
> ```bash
> unset ANTHROPIC_BASE_URL    # Bash/Zsh
> ```

### 5.2 Fix the KV Cache invalidation bug (important, 90% speed impact)

Claude Code prepends an attribution header that invalidates the local model's KV cache, making inference roughly 90% slower. Fix it by editing `~/.claude/settings.json`.

> **Note:** `export CLAUDE_CODE_ATTRIBUTION_HEADER=0` does **not** work. You must set it inside the settings file.

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

### 5.3 Handle the sign-in prompt (if it appears)

If Claude Code asks you to log in on first run, add these two keys to `~/.claude.json`:

```json
{
  "hasCompletedOnboarding": true,
  "primaryApiKey": "sk-dummy-key"
}
```

---

## Part 6: Run Claude Code

Navigate to your project directory, then launch with the model alias that matches your running `llama-server`:

```bash
# GLM-4.7-Flash
cd ~/your-project
claude --model GLM-4.7-Flash

# Qwen3.5-35B-A3B
claude --model Qwen3.5-35B-A3B
```

**Skip permission prompts (full autonomous mode, use carefully):**

```bash
claude --model GLM-4.7-Flash --dangerously-skip-permissions
```

> This lets Claude Code execute commands, write files, and run code without asking. Only use it in a sandboxed project directory where that is acceptable.

### Example prompt to test the setup

```
You can only work in the cwd project/. Do not search for CLAUDE.md - this is it.
Create a Python virtual environment via `python -m venv venv` then activate it.
Install dependencies and write a simple script that fetches and prints the top 5
results from the Hacker News API. You have access to 1 GPU.
```

---

## Part 7: VS Code / Cursor Integration

Install the Claude Code extension directly:

- **VS Code:** [marketplace.visualstudio.com/items?itemName=anthropic.claude-code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)
- **Cursor:** `cursor:extension/anthropic.claude-code`
- Or press `Ctrl+Shift+X`, search **Claude Code**, and click **Install**

The extension picks up `ANTHROPIC_BASE_URL` from your environment, so the same Fish `set -Ux` approach works here too.

If it still prompts for sign-in, add `"claudeCode.disableLoginPrompt": true` to your VS Code `settings.json`.

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
| Downloads stall on Hugging Face | Try `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`, or download via the Hugging Face web UI. |

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
