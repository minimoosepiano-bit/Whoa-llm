"""Shared pytest fixtures and CLI plumbing."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.requires_network (need internet/HF Hub).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_net = pytest.mark.skip(reason="needs --run-network (uses internet)")
    for item in items:
        if "requires_network" in item.keywords:
            item.add_marker(skip_net)
