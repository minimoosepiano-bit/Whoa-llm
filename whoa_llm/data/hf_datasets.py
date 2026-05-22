"""Thin wrappers around ``datasets`` for loading and previewing HF datasets."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_dataset_split(
    name: str,
    *,
    config: str | None = None,
    split: str = "train",
    streaming: bool = False,
    max_samples: int | None = None,
) -> Any:
    """Load a Hugging Face dataset split.

    Parameters
    ----------
    name:
        HF dataset id, e.g. ``"tatsu-lab/alpaca"``.
    config:
        Optional dataset configuration / subset name.
    split:
        Dataset split, e.g. ``"train"``, ``"train[:1000]"``.
    streaming:
        If True, returns an ``IterableDataset`` without downloading all data.
    max_samples:
        If set, truncates the dataset to this many rows (ignored when streaming).

    Returns
    -------
    datasets.Dataset or datasets.IterableDataset
    """
    from datasets import load_dataset

    logger.info("Loading dataset %s (config=%s, split=%s, streaming=%s)", name, config, split, streaming)
    ds = load_dataset(name, config, split=split, streaming=streaming)

    if max_samples is not None and not streaming:
        ds = ds.select(range(min(max_samples, len(ds))))  # type: ignore[arg-type]
        logger.info("Truncated dataset to %d samples", len(ds))  # type: ignore[arg-type]

    return ds


def list_columns(dataset: Any) -> list[str]:
    """Return column names for a loaded dataset (works for both Dataset and IterableDataset)."""
    if hasattr(dataset, "column_names"):
        cols = dataset.column_names
        if isinstance(cols, dict):
            # DatasetDict — pick first split
            return next(iter(cols.values()))
        return list(cols)
    return []


def preview(dataset: Any, n: int = 5) -> list[dict[str, Any]]:
    """Return the first *n* rows of *dataset* as a list of dicts."""
    if hasattr(dataset, "take"):
        # IterableDataset
        return [row for row in dataset.take(n)]  # type: ignore[attr-defined]
    return [dataset[i] for i in range(min(n, len(dataset)))]  # type: ignore[arg-type]
