# Running Local LLMs with Claude Code on macOS (Apple Silicon)

> This guide walks you through connecting open-source LLMs to Claude Code entirely locally using `llama.cpp` and quantized GGUFs on Apple Silicon. No API bills. No cloud.

---

## Overview

Claude Code normally routes requests to Anthropic's servers. By setting `ANTHROPIC_BASE_URL`, you redirect it to a local `llama-server` process instead. Apple Silicon uses Metal for GPU acceleration automatically. No CUDA, no drivers, no extra setup required.

---

## Prerequisites

- macOS 13 Ventura or later (Apple Silicon)
- [Homebrew](https://brew.sh) installed
- Xcode Command Line Tools: `xcode-select --install`
- 24 GB+ unified memory recommended for the 35B-class models (16 GB works with smaller quants)

---

## Part 1: Build llama.cpp

### 1.1 Install dependencies

```bash
brew install cmake curl git aria2 tmux
```

> Metal GPU support is on by default on Apple Silicon. The build picks it up automatically. No extra flags needed.

### 1.2 Clone and build

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=OFF
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp/
```

> `-DGGML_CUDA=OFF` is correct on macOS. Metal acceleration is picked up automatically. You do not need to pass any additional Metal flag.

---

## Part 2: Choose Your Model

**A note on Qwen3.5-35B-A3B and memory:** Despite the "35B" in the name, this is a Mixture of Experts (MoE) model. The "A3B" means only 3 billion parameters are active during any single inference pass. The other parameters are organized into expert layers that sit idle until needed. This is why the model fits in far less memory than a traditional dense 35B model would require. The Q4_K_XL quant lands around 22 GB on disk.

Pick one of the three options below based on your unified memory:

```bash
system_profiler SPHardwareDataType | grep Memory
```

| Unified Memory | Recommended option |
|---|---|
| 24 GB+ | Option 1: Qwen3.5-35B-A3B Q4_K_XL (best quality, fully in memory) |
| 16 GB | Option 2: Qwen3.5-35B-A3B Q4_K_XL split across GPU and RAM (good quality, manageable speed due to MoE) |
| 16 GB (speed priority) | Option 3: Qwen3.5-9B (fits fully in memory, fastest) |

> **Note:** The `unsloth/` prefix in the download URLs below is a Hugging Face account name. That is where these optimized GGUF model files are hosted.

Create a folder to store your models:

```bash
mkdir -p ~/models
```

### Option 1: Qwen3.5-35B-A3B Q4_K_XL (24 GB+ unified memory, best quality)

The full-quality quant. Fits entirely in unified memory on a 24 GB Mac. Best output for agentic coding tasks.

```bash
aria2c -x 16 -s 16 \
    -d ~/models \
    -o Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
```

### Option 2: Qwen3.5-35B-A3B Q4_K_XL (16 GB unified memory, memory split)

Same Q4_K_XL quant as Option 1. On a 16 GB Mac, llama.cpp offloads the inactive expert layers to swap while keeping active compute in unified memory. Because only 3B parameters are active at any time (MoE), the performance penalty is smaller than it would be with a dense model. The download is identical to Option 1.

```bash
aria2c -x 16 -s 16 \
    -d ~/models \
    -o Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
```

> `--fit on` in the `llama-server` command (Part 3) handles the memory split automatically.

### Option 3: Qwen3.5-9B (16 GB unified memory, fastest)

The 9B model fits entirely within 16 GB unified memory with room to spare for context. Noticeably less capable than the 35B variants but responds quickly, which makes it practical for fast iteration during coding sessions.

```bash
aria2c -x 16 -s 16 \
    -d ~/models \
    -o Qwen3.5-9B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf"
```

---

## Part 3: Start llama-server

Run llama-server in a tmux session so it does not take over your terminal:

> **KV cache note:** `q8_0` reduces memory usage. Do **not** use `f16` KV cache with Qwen3.5. Multiple reports show accuracy degradation. Use `bf16` if you want full precision and have the memory headroom.

> **Disable thinking mode** (faster for agentic coding tasks, add to any command below):
> ```bash
> --chat-template-kwargs "{\"enable_thinking\": false}"
> ```

### Option 1: Qwen3.5-35B-A3B Q4_K_XL (24 GB+ unified memory)

```bash
tmux new-session -d -s llama 'bash -c "./llama.cpp/llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-35B-A3B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 131072 \
    2>&1 | tee /tmp/llama-server.log"'
```

### Option 2: Qwen3.5-35B-A3B Q4_K_XL (16 GB unified memory, memory split)

Same file as Option 1. The `--fit on` flag handles the memory split automatically. Reduce `--ctx-size` if the server fails to start.

```bash
tmux new-session -d -s llama 'bash -c "./llama.cpp/llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-35B-A3B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 65536 \
    2>&1 | tee /tmp/llama-server.log"'
```

### Option 3: Qwen3.5-9B (16 GB unified memory, fastest)

```bash
tmux new-session -d -s llama 'bash -c "./llama.cpp/llama-server \
    --model ~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-9B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 131072 \
    2>&1 | tee /tmp/llama-server.log"'
```

Watch the log until you see `server is listening on http://127.0.0.1:8001`:

```bash
tail -f /tmp/llama-server.log
```

To reattach to the tmux session: `tmux attach -t llama`
To detach without stopping it: `Ctrl+B` then `D`

---

## Part 4: Install Unsloth Studio (optional chat UI)

Unsloth Studio is a web UI for chatting with and fine-tuning local models. It runs via Docker Desktop on macOS.

> **Note:** On Apple Silicon, Unsloth Studio supports GGUF chat inference. GPU training is not yet available on macOS (MLX support is coming).

### 4.1 Install Docker Desktop

Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/). Open it and wait for it to finish starting before continuing.

### 4.2 Start Unsloth Studio

```bash
docker run -d \
    -e JUPYTER_PASSWORD="yourpassword" \
    -p 8888:8888 -p 8000:8000 -p 2222:22 \
    -v "$(pwd)/work:/workspace/work" \
    -v "$HOME/models:/root/models" \
    --name unsloth-studio \
    unsloth/unsloth
```

> The `-v "$HOME/models:/root/models"` line mounts your models folder into the container so Unsloth Studio can access the GGUF you already downloaded.

> macOS does not use `--gpus all` since Docker Desktop on macOS does not support GPU passthrough to containers. The container runs in CPU/chat-only mode.

### 4.3 Open the UI

Open `http://localhost:8000` in your browser.

> `http://localhost:8888` is JupyterLab, not the Studio UI. Use port 8000.

### 4.4 Load your model

On the welcome screen, click **Skip to Chat** in the bottom left. In the model selector, search for your model or point to a local path:

```
/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

### 4.5 Managing the container

```bash
docker stop unsloth-studio    # stop without removing
docker start unsloth-studio   # start again later
docker rm unsloth-studio      # remove entirely (stop first)
```

> Unsloth Studio runs its own internal llama-server. Stop your existing tmux llama-server session first to avoid port conflicts:
> ```bash
> tmux kill-session -t llama
> ```

---

## Part 5: Install Claude Code

**Option A: Homebrew:**

```bash
brew install --cask claude-code
```

**Option B: Install script:**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Verify it installed:

```bash
claude --version
```

---

## Part 6: Configure Claude Code

### 6.1 Start llama-server

Claude Code needs llama-server running to route requests to your local model. Use the tmux command from Part 3 that matches your memory configuration. Once started, confirm it is ready:

```bash
tail -f /tmp/llama-server.log
```

Wait for `server is listening on http://127.0.0.1:8001` before continuing.

### 6.2 Point Claude Code at your local server

**Zsh (macOS default, session only):**

```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

**Zsh (persistent, add to `~/.zshrc`):**

```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

Then reload: `source ~/.zshrc`

**Fish shell (session only):**

```fish
set -x ANTHROPIC_BASE_URL "http://localhost:8001"
set -x ANTHROPIC_API_KEY "sk-no-key-required"
```

**Fish shell (persistent):**

```fish
set -Ux ANTHROPIC_BASE_URL "http://localhost:8001"
set -Ux ANTHROPIC_API_KEY "sk-no-key-required"
```

> To switch back to Anthropic's real API:
> ```zsh
> unset ANTHROPIC_BASE_URL    # Zsh/Bash
> ```
> ```fish
> set -e ANTHROPIC_BASE_URL   # Fish
> ```

### 6.3 Fix the KV Cache invalidation bug (important, 90% speed impact)

Claude Code prepends an attribution header that invalidates the local model's KV cache, making inference much slower. Fix it by editing `~/.claude/settings.json`.

> `export CLAUDE_CODE_ATTRIBUTION_HEADER=0` does **not** work. The setting must live inside the JSON config file.

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

**One-liner (Zsh/Bash):**

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

**One-liner (Fish):**

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

> This lets Claude Code execute commands, write files, and run code without confirmation. Only use it in a sandboxed project directory.

### Example prompt to test the setup

```
You can only work in the cwd project/. Do not search for CLAUDE.md - this is it.
Create a Python virtual environment via `python -m venv venv` then activate it.
Install dependencies and write a simple script that fetches and prints the top 5
results from the Hacker News API.
```

---

## Part 8: VS Code / Cursor Integration

Install the Claude Code extension:

- **VS Code:** [marketplace.visualstudio.com/items?itemName=anthropic.claude-code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)
- **Cursor:** `cursor:extension/anthropic.claude-code`
- Or press `Cmd+Shift+X`, search **Claude Code**, and click **Install**

The extension inherits `ANTHROPIC_BASE_URL` from your shell environment. If it still prompts for sign-in, add `"claudeCode.disableLoginPrompt": true` to your VS Code `settings.json`.

---

## Apple Silicon Context Size Reference

Adjust `--ctx-size` based on your Mac's unified memory:

| Unified Memory | Recommended `--ctx-size` |
|---|---|
| 16 GB | 32768 (32K) |
| 24 GB | 65536–131072 (64K–128K) |
| 36 GB | 131072 (128K) |
| 64 GB+ | 262144 (256K) |

Reducing context size frees memory and improves token speed.

---

## Running Multiple Models

You can serve multiple models on different ports simultaneously:

```bash
# Terminal 1: Qwen3.5 on port 8001
tmux new-session -d -s qwen 'bash -c "./llama.cpp/llama-server --model ~/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --alias Qwen3.5-35B-A3B --port 8001 ..."'

# Terminal 2: Qwen3.5-9B on port 8002
tmux new-session -d -s qwen9b 'bash -c "./llama.cpp/llama-server --model ~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf --alias Qwen3.5-9B --port 8002 ..."'
```

Switch between them:

```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"   # switch to 35B
export ANTHROPIC_BASE_URL="http://localhost:8002"   # switch to 9B
```

Or add a shell function to `~/.zshrc`:

```zsh
cclocal() {
  local port=${1:-8001}
  ANTHROPIC_BASE_URL="http://localhost:${port}" \
  ANTHROPIC_API_KEY="sk-no-key-required" \
  claude "${@:2}"
}
```

Then: `cclocal` or `cclocal 8002`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unable to connect to API (ConnectionRefused)` | `llama-server` is not running, or wrong port. Run `tail -f /tmp/llama-server.log` to check. |
| Claude Code is very slow | `CLAUDE_CODE_ATTRIBUTION_HEADER` not set in `~/.claude/settings.json`. See Part 6.3. |
| `missing ANTHROPIC_API_KEY` error | `export ANTHROPIC_API_KEY="sk-no-key-required"` |
| Sign-in loop on first launch | Add `hasCompletedOnboarding` and `primaryApiKey` to `~/.claude.json` |
| Model output loops or is garbled | Update llama.cpp and re-download the GGUF files (a KV cache bug was patched) |
| Out of unified memory / server won't start | Reduce `--ctx-size` (try halving it). See the context size table above. |
| Download stalls or is slow | Use `aria2c -x 16 -s 16 -o <filename> <url>`. If the URL has expired, grab a fresh one from the Hugging Face model page. |
| `cmake` not found | Run `xcode-select --install` then `brew install cmake` |
| Unsloth Studio shows no GPU | Expected on macOS. Docker Desktop does not support GPU passthrough. Chat inference still works via CPU/Metal through llama-server. |

---

## References

- [Unsloth Claude Code Guide](https://unsloth.ai/docs/basics/claude-code)
- [Unsloth Studio Install Guide](https://unsloth.ai/docs/new/studio/install)
- [Unsloth Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)
- [Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Claude Code docs](https://code.claude.com/docs)
- [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
