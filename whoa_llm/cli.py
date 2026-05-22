"""Typer CLI entrypoints. The Gradio app and training runners are imported
lazily so ``whoa-llm --help`` works without the heavy ML stack installed."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="whoa-llm",
    help="Local, memory-efficient LLM fine-tuning (SFT/LoRA/QLoRA/GRPO).",
    no_args_is_help=True,
)


@app.command()
def info() -> None:
    """Print detected hardware and the recommended training preset."""
    from whoa_llm.hardware import detect

    typer.echo(detect().summary())


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Host to bind the Gradio server to."),
    port: int = typer.Option(7860, help="Port to serve the Gradio app on."),
    share: bool = typer.Option(False, help="Create a public Gradio share link."),
) -> None:
    """Launch the Gradio web UI. (Implemented in Phase 5.)"""
    try:
        from whoa_llm.ui.app import launch
    except ImportError as e:  # pragma: no cover - phase-5 gate
        raise typer.BadParameter(
            "UI dependencies not installed. Install with: pip install -e .[core]"
        ) from e
    launch(host=host, port=port, share=share)


@app.command()
def train(config: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Run a headless SFT job from a YAML config."""
    import yaml

    from whoa_llm.training.sft import SFTConfig, run_sft

    data = yaml.safe_load(config.read_text())
    cfg = SFTConfig.model_validate(data)
    summary = run_sft(cfg)
    typer.echo(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":  # pragma: no cover
    app()
