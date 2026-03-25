# Running Local LLMs with Claude Code on macOS (Apple Silicon)

> Connect Claude Code to a local LLM using `llama.cpp`. No API bills. No cloud. No Anthropic account required.

---

## Read this first

This setup works, but there is an important limitation to understand before you invest time in it.

**Claude Code sends very large prompts.** Every request includes your codebase context, tool definitions, and conversation history, often 10,000-20,000 tokens or more. Local models have to process all of that on your hardware before generating a single token of response. On a Mac with less than 32 GB unified memory, that means minutes per request, not seconds.

**Who this works well for:**
- 32 GB+ unified memory (M2 Max, M3 Max, M4 Max or better)
- Anyone who wants local chat inference or fine-tuning experiments
- Testing prompts offline before using the real API

**Who should just use the Anthropic API:**
- Anyone with less than 32 GB unified memory
- Anyone who needs Claude Code to be responsive for day-to-day coding

You can always switch between local and Anthropic's API instantly:

```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"   # use local model
unset ANTHROPIC_BASE_URL                              # switch back to Anthropic
```

---

## How it works

Claude Code normally routes every request to Anthropic's servers. By setting one environment variable (`ANTHROPIC_BASE_URL`), you redirect it to a local `llama-server` process running on your own hardware instead.

```
Claude Code  →  llama-server (port 8001)  →  GGUF model on disk
```

Apple Silicon uses Metal for GPU acceleration automatically. No CUDA, no drivers, no extra setup required.

---

## Prerequisites

- macOS 13 Ventura or later (Apple Silicon)
- [Homebrew](https://brew.sh) installed
- Xcode Command Line Tools: `xcode-select --install`
- 24 GB+ unified memory recommended (16 GB works with smaller quants)

---

## Part 1: Install dependencies

```bash
brew install cmake curl git aria2 tmux
```

> Metal GPU support is on by default on Apple Silicon. The build picks it up automatically. No extra flags needed.

---

## Part 2: Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=OFF
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp/
```

> `-DGGML_CUDA=OFF` is correct on macOS. Metal acceleration is picked up automatically.

---

## Part 3: Download a model

The guide uses **Qwen3.5-35B-A3B**, a Mixture of Experts (MoE) model. Despite the "35B" name, only 3B parameters are active during any single inference pass. This is why it fits in far less memory than a traditional dense 35B model. The Q4_K_XL quant is around 22 GB on disk.

Check your unified memory first:
```bash
system_profiler SPHardwareDataType | grep Memory
```

| Unified Memory | Expected performance |
|---|---|
| 32 GB+ | Good. Model fits fully in unified memory. Responses in seconds. |
| 24 GB | Marginal. May offload some layers. Responses slower than ideal. |
| 16 GB | Poor for agentic coding. Minutes per request. Fine for chat. |

Create a folder to store your models:
```bash
mkdir -p ~/models
```

> The `unsloth/` prefix in the URLs below is a Hugging Face account name, not software you need to install.

### Options 1 and 2: Qwen3.5-35B-A3B (same file for both)

```bash
aria2c -x 16 -s 16 \
    -d ~/models \
    -o Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
```

### Option 3: Qwen3.5-9B

```bash
aria2c -x 16 -s 16 \
    -d ~/models \
    -o Qwen3.5-9B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf"
```

> aria2c resumes automatically if the download is interrupted. Re-run the same command to resume.

---

## Part 4: Start llama-server

Run llama-server in a tmux session so it stays running in the background without taking over your terminal.

### Option 1: Qwen3.5-35B-A3B (24 GB+ unified memory)

```bash
tmux new-session -d -s llama \
'bash -c "./llama.cpp/llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-35B-A3B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on --ctx-size 131072 \
    2>&1 | tee /tmp/llama-server.log"'
```

### Option 2: Qwen3.5-35B-A3B (16 GB unified memory, memory split)

Same file, reduced context size. The `--fit on` flag automatically offloads inactive expert layers while keeping active compute in unified memory. Because only 3B parameters are active at any time (MoE), the speed penalty is manageable.

```bash
tmux new-session -d -s llama \
'bash -c "./llama.cpp/llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --alias Qwen3.5-35B-A3B \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
    --port 8001 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on --ctx-size 65536 \
    2>&1 | tee /tmp/llama-server.log"'
```

### Option 3: Qwen3.5-9B (16 GB unified memory, fastest)

```bash
tmux new-session -d -s llama \
'bash -c "./llama.cpp/llama-server \
    --model ~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
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

> First startup loads the full model into unified memory. Watch Activity Monitor (Memory tab) — the footprint climbs for 10-30 seconds before the server is ready.

> The first request after startup takes longer than usual while the model warms up its KV cache. Subsequent requests are faster.

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

Verify:
```bash
claude --version
```

---

## Part 6: Configure Claude Code

### 6.1 Point Claude Code at your local server

**Zsh (macOS default, persistent — add to `~/.zshrc`):**
```zsh
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY="sk-no-key-required"
```

Then reload: `source ~/.zshrc`

**Fish (persistent):**
```fish
set -Ux ANTHROPIC_BASE_URL "http://localhost:8001"
set -Ux ANTHROPIC_API_KEY "sk-no-key-required"
```

> To switch back to Anthropic's real API: `unset ANTHROPIC_BASE_URL` (Zsh) or `set -e ANTHROPIC_BASE_URL` (Fish).

### 6.2 Fix the KV cache bug

Claude Code prepends a header that breaks the local model's KV cache and makes inference much slower. This must be set inside the settings file, not as an environment variable.

**Zsh/Bash:**
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

### 6.3 Handle the sign-in prompt (if it appears)

If Claude Code asks you to log in on first run, add these two keys to `~/.claude.json`:

```json
{
  "hasCompletedOnboarding": true,
  "primaryApiKey": "sk-dummy-key"
}
```

---

## Part 7: Run Claude Code

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

## Part 8: VS Code / Cursor Integration

Install the Claude Code extension:

- Press `Cmd+Shift+X`, search **Claude Code**, click **Install**
- Direct link: [marketplace.visualstudio.com/items?itemName=anthropic.claude-code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)

The extension picks up `ANTHROPIC_BASE_URL` from your environment automatically. If it still prompts for sign-in, add `"claudeCode.disableLoginPrompt": true` to your VS Code `settings.json`.

---

## Unified memory context size reference

| Unified Memory | Chip examples | Recommended `--ctx-size` | Claude Code suitability |
|---|---|---|---|
| 16 GB | M5, M4 (base) | 16384 (16K) | Chat only. Too slow for agentic use. |
| 24 GB | M5, M4 Pro (base) | 32768 (32K) | Marginal. Expect slow responses. |
| 32 GB | M5, M4 Max (base) | 65536 (64K) | Borderline. Workable for short sessions. |
| 48 GB | M5 Pro, M4 Pro | 131072 (128K) | Good. Recommended minimum for Claude Code. |
| 64 GB | M5 Pro, M4 Max, M5 Max (base) | 131072 (128K) | Great. Comfortable headroom. |
| 128 GB | M5 Max, M4 Max | 262144 (256K) | Excellent. Room for larger models too. |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unable to connect to API` | llama-server is not running. Check with `tail -20 /tmp/llama-server.log` |
| Slow responses | Confirm `CLAUDE_CODE_ATTRIBUTION_HEADER` is set in `~/.claude/settings.json` |
| `missing ANTHROPIC_API_KEY` | `export ANTHROPIC_API_KEY="sk-no-key-required"` |
| Sign-in loop on first launch | Add `hasCompletedOnboarding` and `primaryApiKey` to `~/.claude.json` |
| Model output is garbled | Update llama.cpp and re-download the GGUF |
| Out of unified memory | Reduce `--ctx-size` (try halving it). See table above. |
| Download interrupted | Re-run the same `aria2c` command, it resumes automatically |
| duplicate session: llama | Run `tmux kill-session -t llama` first, then retry |
| `cmake` not found | Run `xcode-select --install` then `brew install cmake` |

---

## References

- [Unsloth Claude Code Guide](https://unsloth.ai/docs/basics/claude-code)
- [Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Claude Code docs](https://code.claude.com/docs)
