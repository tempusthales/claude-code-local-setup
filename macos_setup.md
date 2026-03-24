# Running Local LLMs with Claude Code on macOS

> This guide walks you through connecting open-source LLMs to Claude Code entirely locally using `llama.cpp` and quantized GGUFs on macOS. Works on both Apple Silicon (M-series) and Intel Macs. No API bills. No cloud.

---

## Overview

Claude Code normally routes requests to Anthropic's servers. By setting `ANTHROPIC_BASE_URL`, you redirect it to a local `llama-server` process instead. macOS uses Apple Metal for GPU acceleration — no CUDA needed. A 24 GB unified memory Mac (M2 Pro, M3 Pro, M4 Pro, or better) can run the recommended models comfortably.

---

## Prerequisites

- macOS 13 Ventura or later (Apple Silicon or Intel)
- [Homebrew](https://brew.sh) installed
- 24 GB+ unified memory for the recommended 35B-class models (smaller quants work on 16 GB)
- Xcode Command Line Tools: `xcode-select --install`

---

## Part 1 — Build llama.cpp

### 1.1 Install dependencies

```bash
brew install cmake curl git
```

> Metal GPU support is on by default on macOS — you do not need any extra flags. The build will automatically accelerate on Apple Silicon and Intel Macs with a supported GPU.

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

> Setting `-DGGML_CUDA=OFF` is correct on macOS. Metal acceleration is picked up automatically by the build system — you do not need to pass any additional Metal flag.

---

## Part 2 — Download a Model

Install the Hugging Face download tools:

```bash
pip3 install huggingface_hub hf_transfer
```

> **Note:** The `unsloth/` prefix in the download paths below is a Hugging Face account name — that is where these optimized GGUF model files are hosted. It is not something you need to install or sign up for.

> If you use a virtual environment manager like `pyenv` or `conda`, activate your environment first.

### Option A: Qwen3.5-35B-A3B (recommended for agentic coding)

Uses the **UD-Q4_K_XL** quant — a dynamically quantized GGUF with the best accuracy/size balance. Requires ~24 GB unified memory.

```bash
hf download unsloth/Qwen3.5-35B-A3B-GGUF \
    --local-dir unsloth/Qwen3.5-35B-A3B-GGUF \
    --include "*UD-Q4_K_XL*"
```

> **16 GB Mac?** Use the 2-bit quant instead (lower accuracy but fits): `--include "*UD-Q2_K_XL*"`
>
> **Need a smarter model?** `Qwen3.5-27B` is stronger but runs at roughly half the token speed. Substitute `27B` for `35B-A3B` in the repo name and include pattern.

### Option B: GLM-4.7-Flash (fast, also ~24 GB)

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

## Part 3 — Start llama-server

Run this in a dedicated terminal window or `tmux` pane. Leave it running while you use Claude Code.

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

> **KV cache:** `q8_0` saves unified memory. Avoid `f16` KV cache with Qwen3.5 — reports show accuracy degradation. Use `bf16` if you want full precision and have the memory headroom.

> **Disable thinking mode** (faster for agentic tasks):
> ```bash
> --chat-template-kwargs "{\"enable_thinking\": false}"
> ```

> **Watching model load:** First startup loads the full model into unified memory (20+ GB). Watch Activity Monitor → Memory tab — you will see the footprint climb for 10–30 seconds before the server is ready.

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

## Part 4 — Install Claude Code

**Option A — Homebrew:**

```bash
brew install --cask claude-code
```

**Option B — install script:**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

---

## Part 5 — Configure Claude Code

### 5.1 Point Claude Code at your local server

**Zsh (macOS default) — session only:**

```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

**Zsh — persistent (add to `~/.zshrc`):**

```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

Then reload: `source ~/.zshrc`

**Fish shell — session only:**

```fish
set -x ANTHROPIC_BASE_URL "http://localhost:8001"
set -x ANTHROPIC_API_KEY "sk-no-key-required"
```

**Fish shell — persistent:**

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

### 5.2 Fix the KV Cache invalidation bug (important — 90% speed impact)

Claude Code prepends an attribution header that breaks the local model's KV cache, causing inference to run roughly 90% slower. Fix it by editing `~/.claude/settings.json`.

> `export CLAUDE_CODE_ATTRIBUTION_HEADER=0` does **not** work — the setting must live inside the JSON config file.

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

**One-liner to write the file (Zsh/Bash):**

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

### 5.3 Handle the sign-in prompt (if it appears)

If Claude Code asks you to log in on first run, add these two keys to `~/.claude.json`:

```json
{
  "hasCompletedOnboarding": true,
  "primaryApiKey": "sk-dummy-key"
}
```

---

## Part 6 — Run Claude Code

Navigate to your project directory, then launch with the model alias that matches your running `llama-server`:

```bash
# GLM-4.7-Flash
cd ~/your-project
claude --model GLM-4.7-Flash

# Qwen3.5-35B-A3B
claude --model Qwen3.5-35B-A3B
```

**Skip permission prompts (full autonomous mode — use carefully):**

```bash
claude --model GLM-4.7-Flash --dangerously-skip-permissions
```

> This lets Claude Code execute commands, write files, and run code without confirmation. Only use it in a sandboxed project directory.

### Example prompt to test the setup

```
You can only work in the cwd project/. Do not search for CLAUDE.md - this is it.
Create a Python virtual environment via `python -m venv venv` then activate it.
Install dependencies and write a simple script that fetches and prints the top 5
results from the Hacker News API. You have access to 1 GPU.
```

---

## Part 7 — VS Code / Cursor Integration

Install the Claude Code extension:

- **VS Code:** [marketplace.visualstudio.com/items?itemName=anthropic.claude-code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)
- **Cursor:** `cursor:extension/anthropic.claude-code`
- Or press `Cmd+Shift+X`, search **Claude Code**, and click **Install**

The extension inherits `ANTHROPIC_BASE_URL` from your shell environment. If it still prompts for sign-in, add `"claudeCode.disableLoginPrompt": true` to your VS Code `settings.json`.

---

## Running Multiple Models

You can serve multiple models on different ports simultaneously:

```bash
# Terminal 1 — Qwen3.5 on port 8001
./llama.cpp/llama-server --model Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --alias "Qwen3.5-35B-A3B" --port 8001 ...

# Terminal 2 — GLM-4.7-Flash on port 8002
./llama.cpp/llama-server --model GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf --alias "GLM-4.7-Flash" --port 8002 ...
```

Switch between them by changing the env var:

```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"   # switch to Qwen3.5
export ANTHROPIC_BASE_URL="http://localhost:8002"   # switch to GLM
```

Or with a shell function in `~/.zshrc`:

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

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unable to connect to API (ConnectionRefused)` | `llama-server` is not running, or wrong port. Check that terminal. |
| Claude Code is very slow (90% slower) | `CLAUDE_CODE_ATTRIBUTION_HEADER` not set in `~/.claude/settings.json` — see Part 5.2. |
| `missing ANTHROPIC_API_KEY` error | `export ANTHROPIC_API_KEY="sk-no-key-required"` |
| Sign-in loop on first launch | Add `hasCompletedOnboarding` and `primaryApiKey` to `~/.claude.json` |
| Model output loops or is garbled | Update llama.cpp and re-download the GGUF files (a KV cache bug was patched) |
| Out of unified memory / server won't start | Reduce `--ctx-size` (try halving it) |
| Downloads stall on Hugging Face | Try `pip3 install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`, or download via the Hugging Face web UI. |
| `cmake` not found | Run `xcode-select --install` then `brew install cmake` |
| `hf` command not found | `pip3 install huggingface_hub hf_transfer` and ensure `~/Library/Python/3.x/bin` is on `$PATH` |

---

## References

- [Unsloth Claude Code Guide](https://unsloth.ai/docs/basics/claude-code)
- [Unsloth Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)
- [Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [GLM-4.7-Flash Guide](https://unsloth.ai/docs/models/glm-4.7-flash)
- [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Claude Code docs](https://code.claude.com/docs)
