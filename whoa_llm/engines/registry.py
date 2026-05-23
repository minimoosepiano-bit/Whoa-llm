"""Engine registry: decide whether to use Unsloth or the plain HF stack.

The registry checks the model_id against a set of known Unsloth-compatible
model families.  If Unsloth is installed *and* the model matches, we route to
the Unsloth engine; otherwise we fall back to the HF engine.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lowercase prefixes of model IDs / names known to work with Unsloth.
# A model matches if any prefix appears anywhere in its lowercased id.
_UNSLOTH_FAMILIES: set[str] = {
    "llama",
    "mistral",
    "mixtral",
    "qwen",
    "gemma",
    "phi",
    "tinyllama",
    "falcon",
    "yi",
    "deepseek",
    "vicuna",
    "solar",
}


def _unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401

        return True
    except ImportError:
        return False


def pick_engine(
    model_id: str,
    *,
    force: str | None = None,
) -> str:
    """Return ``"unsloth"`` or ``"hf"`` for the given *model_id*.

    Parameters
    ----------
    model_id:
        A Hugging Face model ID such as ``"meta-llama/Llama-3.2-1B"``.
    force:
        Override the auto-selection.  Pass ``"unsloth"`` or ``"hf"`` to
        skip the heuristic entirely.  ``None`` (default) auto-detects.

    Returns
    -------
    str
        ``"unsloth"`` when Unsloth is installed and the model family is
        supported; ``"hf"`` otherwise.
    """
    if force is not None:
        engine = force.lower().strip()
        if engine not in {"unsloth", "hf"}:
            raise ValueError(f"force must be 'unsloth' or 'hf', got {force!r}")
        if engine == "unsloth" and not _unsloth_available():
            logger.warning("Unsloth forced but not installed — falling back to HF engine.")
            return "hf"
        return engine

    lower_id = model_id.lower()
    family_match = any(family in lower_id for family in _UNSLOTH_FAMILIES)

    if family_match and _unsloth_available():
        logger.debug("Engine selected: unsloth (model=%s)", model_id)
        return "unsloth"

    if family_match:
        logger.debug(
            "Model %s matches an Unsloth family but Unsloth is not installed; using HF.",
            model_id,
        )
    else:
        logger.debug("Engine selected: hf (model=%s — no Unsloth family match)", model_id)

    return "hf"
