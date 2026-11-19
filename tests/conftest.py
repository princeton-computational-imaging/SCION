"""Repo-wide test configuration.

Applies the ``gpu`` marker (registered in pyproject) to every test that
carries a CUDA-availability skipif, so ``pytest -m "not gpu"`` /
``-m gpu`` select correctly without each file declaring the marker.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        for mark in item.iter_markers("skipif"):
            reason = str(mark.kwargs.get("reason", ""))
            if "cuda" in reason.lower() or "gpu" in reason.lower():
                item.add_marker(pytest.mark.gpu)
                break
