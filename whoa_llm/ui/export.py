"""Model export helpers used by the Gradio Export tab.

Two operations:

* :func:`merge_adapter` — load a base model + a LoRA adapter from
  ``adapter_dir``, call ``merge_and_unload``, save the merged weights as
  ``safetensors`` to ``output_dir``.
* :func:`push_to_hub` — upload an already-merged or adapter-only directory
  to the Hugging Face Hub.

Both functions import torch / transformers / peft lazily.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def find_gguf_converter() -> str | None:
    """Return the path to llama.cpp's ``convert_hf_to_gguf.py`` if available.

    Looks on ``$PATH`` and falls back to common locations.  Returns ``None``
    when nothing is found; callers should show an install hint.
    """
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        found = shutil.which(name)
        if found:
            return found
    # Common manual-install paths.
    for candidate in (
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def convert_to_gguf(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    quant: str = "q4_k_m",
) -> Path:
    """Convert an HF model directory to GGUF using llama.cpp's converter.

    Parameters
    ----------
    model_dir:
        Directory containing the *merged* HF model (not a LoRA adapter).
    out_path:
        Destination ``.gguf`` file.
    quant:
        Quantisation type passed through as ``--outtype``.

    Raises
    ------
    FileNotFoundError
        If the llama.cpp converter isn't on ``$PATH``.
    """
    converter = find_gguf_converter()
    if converter is None:
        raise FileNotFoundError(
            "Could not find convert_hf_to_gguf.py from llama.cpp on $PATH. "
            "Clone https://github.com/ggerganov/llama.cpp and add the repo "
            "to PATH, or place convert_hf_to_gguf.py somewhere we look "
            "(~/llama.cpp/ or /opt/llama.cpp/)."
        )

    model_dir = Path(model_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", converter, str(model_dir),
        "--outfile", str(out_path),
        "--outtype", quant,
    ]
    logger.info("Running GGUF conversion: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def merge_adapter(
    base_model_id: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "bf16",
    trust_remote_code: bool = False,
) -> Path:
    """Merge a LoRA adapter into its base and save the result.

    Returns the output directory path.
    """
    from peft import PeftModel

    from whoa_llm.engines.hf_engine import load_model_and_tokenizer

    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model %s for merge…", base_model_id)
    base, tok = load_model_and_tokenizer(
        base_model_id,
        quantization="none",
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )

    logger.info("Attaching adapter from %s", adapter_dir)
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    logger.info("Saving merged model to %s", output_dir)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tok.save_pretrained(str(output_dir))
    return output_dir


def push_to_hub(
    local_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str = "Upload fine-tuned model",
) -> str:
    """Upload *local_dir* to ``repo_id`` on the Hub. Returns the repo URL."""
    from huggingface_hub import HfApi, create_repo

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    create_repo(repo_id, private=private, exist_ok=True)
    api = HfApi()
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, commit_message=commit_message)
    return f"https://huggingface.co/{repo_id}"


def write_model_card(
    output_dir: str | Path,
    *,
    base_model: str,
    method: str,
    dataset: str,
    hyperparams: dict,
) -> Path:
    """Write a minimal README.md model card next to merged weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "README.md"
    hp_lines = "\n".join(f"  - **{k}**: `{v}`" for k, v in hyperparams.items())
    body = f"""---
base_model: {base_model}
library_name: peft
tags:
  - whoa-llm
  - {method}
---

# Fine-tuned {base_model}

- **Method**: `{method}`
- **Dataset**: `{dataset}`
- **Hyperparameters**:
{hp_lines}

Produced by [whoa-llm](https://github.com/minimoosepiano-bit/whoa-llm).
"""
    path.write_text(body)
    return path


def card_from_config(output_dir: str | Path, config_dict: dict) -> Path:
    """Convenience: build a model card from a saved ``config.yaml`` dict."""
    train = config_dict.get("train", {})
    lora = config_dict.get("lora", {})
    hyperparams = {
        "epochs": train.get("epochs"),
        "learning_rate": train.get("learning_rate"),
        "batch_size": train.get("per_device_train_batch_size"),
        "grad_accum": train.get("gradient_accumulation_steps"),
        "lora_r": lora.get("r"),
        "lora_alpha": lora.get("alpha"),
        "precision": config_dict.get("precision"),
        "quantization": config_dict.get("memory", {}).get("quantization"),
    }
    return write_model_card(
        output_dir,
        base_model=config_dict.get("model_id", "unknown"),
        method=config_dict.get("method", "lora"),
        dataset=(config_dict.get("dataset") or {}).get("name", "unknown"),
        hyperparams={k: v for k, v in hyperparams.items() if v is not None},
    )
