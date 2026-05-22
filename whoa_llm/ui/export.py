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
from pathlib import Path

logger = logging.getLogger(__name__)


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
    body = f"""# Fine-tuned {base_model}

- **Method**: {method}
- **Dataset**: {dataset}
- **Hyperparameters**: `{hyperparams}`

Produced by [whoa-llm](https://github.com/minimoosepiano-bit/whoa-llm).
"""
    path.write_text(body)
    return path
