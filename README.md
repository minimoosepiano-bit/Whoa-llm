# Whoa-llm

Local, memory-efficient LLM fine-tuning with a Gradio web UI. Works with any
Hugging Face causal LM and dataset. Supports:

- **SFT** in three modes — full fine-tune, LoRA, QLoRA
- **GRPO** — RL-style post-training driven by reward functions (no preference
  pairs needed)
- **Memory offloading** to CPU and disk for low-VRAM machines
- **Unsloth** acceleration for supported model families
  (Llama / Mistral / Qwen / Gemma / Phi / Mixtral / Falcon / Yi / DeepSpeed
  / Solar / TinyLlama), with a Hugging Face fallback for everything else

## Install

```bash
# Base CLI (no ML deps yet)
pip install -e .

# Add the training stack (CUDA wheels for bitsandbytes on Linux)
pip install -e '.[core]'

# Optional: Unsloth for faster training on supported families
pip install -e '.[unsloth]'

# Optional: dev tools
pip install -e '.[dev]'
```

You'll also need a Hugging Face token if you plan to use gated models:

```bash
export HF_TOKEN=hf_xxx
```

## Quickstart — Web UI

```bash
whoa-llm ui
# → http://127.0.0.1:7860
```

The UI has five tabs:

| Tab | What it does |
|------|-------------|
| **Model** | Pick a HF model id, see hardware summary, click **Estimate memory** to get a method/quant/offload recommendation without downloading weights |
| **Dataset** | Pick a HF dataset, choose a format (alpaca / sharegpt / completion / raw_text / chat_template), preview 5 rows tokenised with the model's tokenizer |
| **Training** | All training/LoRA/memory hyperparameters |
| **Run** | Start, cancel, live loss plot, log tail, **Save config as YAML**, **Resume run** from a previous output directory |
| **GRPO** | Pick reward functions (built-ins or pasted Python), set generation knobs, watch sample (prompt, completion, reward) triples evolve |
| **Export** | Merge LoRA → base, generate model card, convert to GGUF, push to Hub |

## Quickstart — CLI

```bash
# Headless run from a YAML
whoa-llm train examples/configs/sft_qlora.yaml

# Resume from a previous output directory
whoa-llm resume outputs/llama-3.2-1b-alpaca-qlora

# Hardware / preset summary
whoa-llm info
```

### Example configs

- `examples/configs/sft_qlora.yaml` — Llama-3.2-1B QLoRA on Alpaca
- `examples/configs/sft_lora_cpu.yaml` — Tiny GPT-2 LoRA on CPU (sanity test)
- `examples/configs/sft_full_tiny.yaml` — Full fine-tune of SmolLM2-135M
- `examples/configs/grpo_format.yaml` — GRPO on GSM8K with a
  `<answer>\d+</answer>` format reward

## Memory presets

The recommender (`whoa-llm info` for hardware, or **Estimate memory** in the
UI) picks one of four regimes from your free VRAM, the model size, and the
sequence length:

| Free VRAM vs. model | Recommended setup | Notes |
|---|---|---|
| ≥ 2× bf16 weights + activations | Full or LoRA at bf16, no offload | Best throughput |
| ≥ 4-bit weights + activations + 1 GB | **QLoRA** 4-bit on-GPU | Default for consumer GPUs |
| Less than that, with a GPU | QLoRA 4-bit + CPU/disk offload | Slower; `max_memory` cap on GPU + CPU |
| No CUDA GPU | LoRA at fp32 on CPU | Viable only for ≤1B models |

`bf16` is recommended on Ampere+ (compute capability ≥ 8.0); pre-Ampere falls
back to `fp16`. The recommender outputs a `MemoryRecommendation` with a
human-readable `rationale` and a `breakdown` of the byte budget.

## GRPO reward authoring

A reward function takes per-step batches of *prompts* and *completions* and
returns a list of floats:

```python
def my_reward(prompts, completions, **kwargs):
    return [1.0 if "hello" in str(c).lower() else 0.0 for c in completions]
```

Two ways to use a custom reward:

1. **Paste in the UI** — the GRPO tab has a code box that writes your
   function to `~/.whoa_llm/user_rewards/<name>.py` and makes it available
   under the same name in the multi-select.
2. **Reference by dotted path** in a YAML config:
   ```yaml
   rewards:
     - name: mypkg.rewards.my_reward
       kwargs: { ... }
       weight: 1.0
   ```

### Built-in rewards

| Name | Description |
|---|---|
| `length_reward` | 1.0 at `target_tokens`, falls off linearly to 0.0 |
| `regex_format_reward` | 1.0 when `re.search(pattern, completion)` matches |
| `contains_reward` | Fraction of `keywords` that appear in the completion |

## Engine fallback

`whoa-llm` selects the training engine automatically:

1. **Unsloth** if the model id matches a known family
   (Llama / Mistral / Qwen / Gemma / Phi / Mixtral / Falcon / Yi /
   DeepSeek / Solar / TinyLlama / Vicuna) **and** `unsloth` is installed.
   Loading is faster and uses less memory.
2. **Hugging Face** otherwise. Supports CPU/disk offload, `bitsandbytes`
   4-bit/8-bit quantisation, and any model `transformers` can load.

If Unsloth raises during loading we automatically fall back to the HF engine.
Force one or the other with the `Engine` dropdown in the UI or the
`engine: unsloth | hf | auto` key in YAML.

## Troubleshooting

- **`OutOfMemoryError`** — let the `OOMRecoveryCallback` retry once
  (it halves batch size and doubles grad-accum), or click **Estimate
  memory** in the UI and apply the recommendation. For 7B+ on consumer
  GPUs, use QLoRA (`method: qlora` + `quantization: 4bit`).
- **`bitsandbytes` install fails** — it only ships wheels for Linux+CUDA.
  On macOS/Windows or CPU-only boxes, set `quantization: none` and use
  `method: lora` (or `full` for tiny models).
- **`Unsloth not found`** — the engine registry will fall back to HF.
  To force Unsloth, install it: `pip install -e '.[unsloth]'`.
- **`apply_chat_template not available`** — pick a different `format`
  (e.g. `alpaca`) or use a chat-capable tokenizer.
- **HF Hub 401/403** — set `HF_TOKEN` and make sure you've accepted the
  model's licence on huggingface.co.
- **GRPO loss looks weird** — try lowering `beta` (KL coefficient),
  raising `num_generations` to ≥4, and verify your reward function with
  the live (prompt, completion, reward) panel in the GRPO tab.

## Repository layout

```
whoa_llm/
  cli.py                    # Typer entrypoint (info, ui, train, resume)
  config.py                 # Settings + HF login helper
  hardware.py               # GPU/RAM detection + memory recommender
  engines/
    registry.py             # pick_engine()
    hf_engine.py            # load_model_and_tokenizer() via transformers
    unsloth_engine.py       # same signature via FastLanguageModel
  data/
    hf_datasets.py          # load + preview HF datasets
    formatting.py           # alpaca/sharegpt/completion/raw_text/chat_template
  training/
    sft.py                  # SFTConfig + run_sft (full/LoRA/QLoRA)
    grpo.py                 # GRPOConfig + run_grpo
    callbacks.py            # MetricsCallback + OOMRecoveryCallback
    resume.py               # find_latest_checkpoint() + resume_run()
    rewards/
      builtin.py            # length / regex / contains
      loader.py             # resolve_reward()
  ui/
    app.py                  # Gradio Blocks (5 tabs)
    state.py                # RunState + RewardSampleBuffer
    export.py               # merge_adapter / push_to_hub / convert_to_gguf
                            # / model card
examples/configs/           # 4 ready-to-run YAML recipes
tests/                      # 145 tests; --run-network unlocks HF tests
```

## Contributing

Run the suite locally:

```bash
pip install -e '.[core,dev]'
python -m pytest -q                  # offline-only, ~10s wall time
python -m pytest --run-network       # adds tests that hit HF Hub
ruff check whoa_llm tests
```

## License

MIT — see `LICENSE`.
