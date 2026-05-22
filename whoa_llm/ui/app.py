"""Gradio app: 5-tab UI for the whoa-llm SFT workflow.

Tabs:
  * **Model**    — pick a HF model, see hardware, estimate memory.
  * **Dataset**  — pick an HF dataset, format, preview.
  * **Training** — method (full/lora/qlora) + LoRA + train + memory knobs.
  * **Run**      — start/cancel, live loss plot, log tail, summary.
  * **Export**   — merge LoRA → base, save safetensors, push to Hub.

The app is built around a single :class:`~whoa_llm.ui.state.RunState`
instance shared across tabs.  ``gr.State`` holds the per-session config
draft as a dict that is materialised into a :class:`SFTConfig` when the
user starts a run.

Heavy ML imports are deferred to the moment the user clicks a button,
so the UI loads even on machines without torch installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gradio as gr
import yaml

from whoa_llm.ui.state import RunState

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _hardware_summary() -> str:
    from whoa_llm.hardware import detect
    return detect().summary()


def _estimate_memory_handler(model_id: str, method: str, seq_len: int):
    """Return (markdown rationale, breakdown table) for the Estimate button."""
    if not model_id.strip():
        return "Enter a Hugging Face model id first.", []
    from whoa_llm.hardware import estimate_memory

    rec = estimate_memory(model_id, method=method, seq_len=int(seq_len))
    if rec is None:
        return (
            "Could not read the model config (offline, gated, or unsupported).\n"
            "Fill the memory knobs manually below.",
            [],
        )
    md = f"### Recommendation\n\n{rec.rationale}\n\n"
    md += f"**Method**: `{rec.method}` &nbsp; **Quant**: `{rec.quantization}` &nbsp; "
    md += f"**Precision**: `{rec.precision}` &nbsp; **CPU offload**: `{rec.cpu_offload}`"
    rows = [[k, v] for k, v in rec.breakdown.items()]
    return md, rows


def _dataset_preview_handler(name: str, config: str, split: str, fmt: str, model_id: str):
    """Load 5 rows of the dataset and render them with the formatter applied."""
    from whoa_llm.data.formatting import make_formatting_func
    from whoa_llm.data.hf_datasets import load_dataset_split, preview

    if not name.strip():
        return "Enter a dataset id first.", []

    try:
        ds = load_dataset_split(
            name.strip(), config=config.strip() or None, split=split.strip() or "train",
            max_samples=5,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Dataset load failed: {exc}", []

    cols = list(ds.column_names) if hasattr(ds, "column_names") else []
    rows = preview(ds, 5)

    formatted = ""
    if model_id.strip():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_id.strip())
            fn = make_formatting_func(fmt, tok)
            formatted_examples = [fn(r)[:400] for r in rows]
            formatted = "\n\n---\n\n".join(formatted_examples)
        except Exception as exc:  # noqa: BLE001
            formatted = f"(Could not load tokenizer for {model_id!r}: {exc})"

    table = [[json.dumps(r, default=str)[:200] for r in [row]] for row in rows]
    md = f"**Columns**: `{cols}`\n\n### Formatted ({fmt})\n\n```\n{formatted}\n```"
    return md, table


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------
def _build_config(form: dict[str, Any]) -> Any:
    """Turn the flattened UI dict into an :class:`SFTConfig`."""
    from whoa_llm.training.sft import (
        DatasetConfig,
        LoRAConfig,
        MemoryConfig,
        SFTConfig,
        TrainConfig,
    )

    cfg = SFTConfig(
        model_id=form["model_id"],
        engine=form.get("engine", "auto"),
        method=form["method"],
        precision=form.get("precision", "bf16"),
        seed=int(form.get("seed", 42)),
        output_dir=form.get("output_dir") or "outputs/sft",
        run_name=form.get("run_name") or None,
        dataset=DatasetConfig(
            name=form["dataset_name"],
            config=form.get("dataset_config") or None,
            split=form.get("dataset_split", "train"),
            format=form.get("dataset_format", "alpaca"),
            max_samples=int(form["max_samples"]) if form.get("max_samples") else None,
        ),
        lora=LoRAConfig(
            r=int(form.get("lora_r", 16)),
            alpha=int(form.get("lora_alpha", 32)),
            dropout=float(form.get("lora_dropout", 0.05)),
        ),
        train=TrainConfig(
            epochs=float(form.get("epochs", 1)),
            learning_rate=float(form.get("learning_rate", 2e-4)),
            per_device_train_batch_size=int(form.get("batch_size", 2)),
            gradient_accumulation_steps=int(form.get("grad_accum", 4)),
            max_seq_length=int(form.get("max_seq_length", 1024)),
            max_steps=int(form.get("max_steps", -1)),
            logging_steps=int(form.get("logging_steps", 5)),
            save_steps=int(form.get("save_steps", 200)),
            warmup_ratio=float(form.get("warmup_ratio", 0.03)),
            optim=form.get("optim", "paged_adamw_8bit"),
            gradient_checkpointing=bool(form.get("gradient_checkpointing", True)),
        ),
        memory=MemoryConfig(
            quantization=form.get("quantization", "none"),
            cpu_offload=bool(form.get("cpu_offload", False)),
            disk_offload_dir=form.get("disk_offload_dir") or None,
            attn_impl=form.get("attn_impl") or None,
        ),
    )
    return cfg.normalised()


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------
def _build_model_tab(form: gr.State):
    with gr.Tab("Model"):
        gr.Markdown("### Pick a model and check it'll fit")
        with gr.Row():
            model_id = gr.Textbox(
                label="HF model id",
                value="meta-llama/Llama-3.2-1B",
                placeholder="e.g. mistralai/Mistral-7B-v0.1",
            )
            engine = gr.Dropdown(
                label="Engine", choices=["auto", "unsloth", "hf"], value="auto",
            )
        hw_md = gr.Markdown(f"```\n{_hardware_summary()}\n```")
        with gr.Row():
            method = gr.Dropdown(
                label="Method", choices=["lora", "qlora", "full"], value="lora",
            )
            seq_len_est = gr.Slider(
                label="Seq length (for estimate)", minimum=128, maximum=8192,
                step=128, value=1024,
            )
        est_btn = gr.Button("Estimate memory", variant="primary")
        est_md = gr.Markdown()
        est_table = gr.Dataframe(headers=["metric", "value"], interactive=False)

        est_btn.click(
            _estimate_memory_handler,
            inputs=[model_id, method, seq_len_est],
            outputs=[est_md, est_table],
        )

        # Persist to shared form state.
        for w, key in [(model_id, "model_id"), (engine, "engine"), (method, "method")]:
            w.change(_update_form(key), inputs=[form, w], outputs=form)

    return {"model_id": model_id, "engine": engine, "method": method}


def _build_dataset_tab(form: gr.State, model_id_w: gr.Textbox):
    with gr.Tab("Dataset"):
        gr.Markdown("### Hugging Face dataset")
        with gr.Row():
            ds_name = gr.Textbox(label="Dataset id", value="tatsu-lab/alpaca")
            ds_config = gr.Textbox(label="Config (optional)", value="")
            ds_split = gr.Textbox(label="Split", value="train[:1000]")
        with gr.Row():
            ds_format = gr.Dropdown(
                label="Format",
                choices=["alpaca", "sharegpt", "completion", "raw_text", "chat_template"],
                value="alpaca",
            )
            max_samples = gr.Number(label="Max samples (optional)", value=None, precision=0)
        preview_btn = gr.Button("Preview")
        preview_md = gr.Markdown()
        preview_table = gr.Dataframe(headers=["row"], interactive=False, wrap=True)

        preview_btn.click(
            _dataset_preview_handler,
            inputs=[ds_name, ds_config, ds_split, ds_format, model_id_w],
            outputs=[preview_md, preview_table],
        )

        for w, key in [
            (ds_name, "dataset_name"),
            (ds_config, "dataset_config"),
            (ds_split, "dataset_split"),
            (ds_format, "dataset_format"),
            (max_samples, "max_samples"),
        ]:
            w.change(_update_form(key), inputs=[form, w], outputs=form)


def _build_training_tab(form: gr.State):
    with gr.Tab("Training"):
        gr.Markdown("### Hyperparameters & memory")
        with gr.Row():
            epochs = gr.Number(label="Epochs", value=1.0)
            max_steps = gr.Number(label="Max steps (-1 = use epochs)", value=-1, precision=0)
            lr = gr.Number(label="Learning rate", value=2e-4)
        with gr.Row():
            bs = gr.Number(label="Per-device batch size", value=2, precision=0)
            ga = gr.Number(label="Gradient accumulation", value=4, precision=0)
            seq = gr.Number(label="Max seq length", value=1024, precision=0)
        with gr.Row():
            warmup = gr.Number(label="Warmup ratio", value=0.03)
            log_steps = gr.Number(label="Logging steps", value=5, precision=0)
            save_steps = gr.Number(label="Save steps", value=200, precision=0)
        with gr.Row():
            optim = gr.Dropdown(
                label="Optimiser",
                choices=["paged_adamw_8bit", "adamw_torch", "adamw_torch_fused", "adamw_bnb_8bit"],
                value="paged_adamw_8bit",
            )
            precision = gr.Dropdown(label="Precision", choices=["bf16", "fp16", "fp32"], value="bf16")
            grad_ckpt = gr.Checkbox(label="Gradient checkpointing", value=True)

        gr.Markdown("#### LoRA")
        with gr.Row():
            lora_r = gr.Slider(label="r", minimum=1, maximum=128, step=1, value=16)
            lora_alpha = gr.Slider(label="alpha", minimum=1, maximum=256, step=1, value=32)
            lora_dropout = gr.Slider(label="dropout", minimum=0, maximum=0.5, step=0.01, value=0.05)

        gr.Markdown("#### Memory")
        with gr.Row():
            quant = gr.Dropdown(label="Quantization", choices=["none", "4bit", "8bit"], value="none")
            cpu_off = gr.Checkbox(label="CPU offload", value=False)
            attn = gr.Dropdown(
                label="Attention impl",
                choices=["", "sdpa", "flash_attention_2", "eager"], value="",
            )

        gr.Markdown("#### Output")
        with gr.Row():
            output_dir = gr.Textbox(label="Output dir", value="outputs/sft")
            run_name = gr.Textbox(label="Run name (optional)")
            seed = gr.Number(label="Seed", value=42, precision=0)

        mapping = [
            (epochs, "epochs"), (max_steps, "max_steps"), (lr, "learning_rate"),
            (bs, "batch_size"), (ga, "grad_accum"), (seq, "max_seq_length"),
            (warmup, "warmup_ratio"), (log_steps, "logging_steps"), (save_steps, "save_steps"),
            (optim, "optim"), (precision, "precision"), (grad_ckpt, "gradient_checkpointing"),
            (lora_r, "lora_r"), (lora_alpha, "lora_alpha"), (lora_dropout, "lora_dropout"),
            (quant, "quantization"), (cpu_off, "cpu_offload"), (attn, "attn_impl"),
            (output_dir, "output_dir"), (run_name, "run_name"), (seed, "seed"),
        ]
        for w, key in mapping:
            w.change(_update_form(key), inputs=[form, w], outputs=form)


def _build_run_tab(form: gr.State, run_state: RunState):
    with gr.Tab("Run"):
        gr.Markdown("### Start / monitor a training run")
        with gr.Row():
            start_btn = gr.Button("Start training", variant="primary")
            cancel_btn = gr.Button("Cancel")
            save_yaml_btn = gr.Button("Save config as YAML")
        status_md = gr.Markdown("Idle.")
        loss_plot = gr.LinePlot(
            value=None,
            x="step",
            y="loss",
            title="Training loss",
            height=320,
        )
        log_box = gr.Textbox(label="Log", lines=12, interactive=False, max_lines=20)
        saved_yaml = gr.Textbox(label="Saved YAML path", interactive=False)
        timer = gr.Timer(POLL_INTERVAL_S)

        def _start(form_state):
            if run_state.is_running:
                return "A run is already in progress."
            try:
                cfg = _build_config(form_state)
            except Exception as exc:  # noqa: BLE001
                return f"Invalid config: {exc}"
            ok = run_state.start(cfg)
            return "Run started." if ok else "Could not start a run."

        def _cancel():
            run_state.cancel()
            return "Cancellation requested."

        def _save_yaml(form_state):
            try:
                cfg = _build_config(form_state)
            except Exception as exc:  # noqa: BLE001
                return f"Invalid config: {exc}"
            path = Path(cfg.output_dir) / "config.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
            return str(path)

        def _tick():
            snap = run_state.metrics.snapshot()
            if snap:
                # gr.LinePlot accepts a list-of-dicts.
                plot_data = [
                    {"step": p["step"], "loss": p["loss"]}
                    for p in snap if p.get("loss") is not None
                ]
            else:
                plot_data = None
            if run_state.is_running:
                latest = run_state.metrics.latest or {}
                status = f"Running — step {latest.get('step', '?')} loss {latest.get('loss', '?')}"
            elif run_state.last_error:
                status = f"❌ Error: {run_state.last_error}"
            elif run_state.last_summary:
                status = f"✅ Done — {run_state.last_summary}"
            else:
                status = "Idle."
            return status, plot_data, run_state.log.text()

        start_btn.click(_start, inputs=[form], outputs=[status_md])
        cancel_btn.click(_cancel, outputs=[status_md])
        save_yaml_btn.click(_save_yaml, inputs=[form], outputs=[saved_yaml])
        timer.tick(_tick, outputs=[status_md, loss_plot, log_box])


def _build_export_tab():
    with gr.Tab("Export"):
        gr.Markdown("### Merge LoRA + push to Hub")
        with gr.Row():
            base_model = gr.Textbox(label="Base model id")
            adapter_dir = gr.Textbox(label="Adapter directory")
            out_dir = gr.Textbox(label="Output directory")
        merge_btn = gr.Button("Merge & save")
        merge_status = gr.Markdown()

        with gr.Row():
            repo_id = gr.Textbox(label="HF repo (user/name)")
            private = gr.Checkbox(label="Private", value=True)
        push_btn = gr.Button("Push to Hub")
        push_status = gr.Markdown()

        def _do_merge(base, adapter, out):
            from whoa_llm.ui.export import merge_adapter
            try:
                p = merge_adapter(base, adapter, out)
                return f"Merged → `{p}`"
            except Exception as exc:  # noqa: BLE001
                return f"Merge failed: {exc}"

        def _do_push(local, repo, priv):
            from whoa_llm.ui.export import push_to_hub
            try:
                url = push_to_hub(local, repo, private=priv)
                return f"Pushed → {url}"
            except Exception as exc:  # noqa: BLE001
                return f"Push failed: {exc}"

        merge_btn.click(_do_merge, inputs=[base_model, adapter_dir, out_dir], outputs=[merge_status])
        push_btn.click(_do_push, inputs=[out_dir, repo_id, private], outputs=[push_status])


def _update_form(key: str):
    """Return a Gradio handler that writes ``value`` into ``form_state[key]``."""
    def _h(form_state, value):
        form_state = dict(form_state or {})
        form_state[key] = value
        return form_state
    return _h


# ---------------------------------------------------------------------------
# Top-level Blocks
# ---------------------------------------------------------------------------
def build_app(run_state: RunState | None = None) -> gr.Blocks:
    run_state = run_state or RunState()
    with gr.Blocks(title="Whoa-llm: Local LLM Fine-tuning") as app:
        gr.Markdown("# Whoa-llm — Local LLM Fine-tuning")
        form: gr.State = gr.State({})

        model_widgets = _build_model_tab(form)
        _build_dataset_tab(form, model_widgets["model_id"])
        _build_training_tab(form)
        _build_run_tab(form, run_state)
        _build_export_tab()

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    """Entry point used by the CLI."""
    app = build_app()
    app.queue().launch(server_name=host, server_port=port, share=share)
