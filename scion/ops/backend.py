"""Backend resolution for SCION hot-path ops.

Each registered op resolves through tiers CUDA > Triton > Python, with
the resolution cached per op-name. Per-op env-var overrides force a
specific tier (useful for parity tests):
    SCION_BACKEND_<op_name>=cuda|triton|python
"""
from __future__ import annotations

import os
from typing import Callable, Dict

from scion import _cuda, _triton
from scion.ops import _ref


# Registered ops: {name: {"cuda": fn, "triton": fn, "python": fn}}.
# Built at import time from each tier module's OPS dict + _ref namespace.
_REGISTRY: Dict[str, Dict[str, Callable]] = {}


def _build_registry() -> None:
    """Populate _REGISTRY from the three tier modules."""
    # Reference (Python) tier: introspect scion.ops._ref for
    # callables -- every public function exported is a candidate.
    py_table = {
        name: getattr(_ref, name)
        for name in getattr(_ref, "__all__", [])
        if callable(getattr(_ref, name, None))
    }

    all_op_names = set(py_table) | set(_cuda.OPS) | set(_triton.OPS)
    for name in all_op_names:
        _REGISTRY[name] = {
            "cuda":   _cuda.OPS.get(name),
            "triton": _triton.OPS.get(name),
            "python": py_table.get(name),
        }


_build_registry()


def _resolve(op_name: str) -> Callable:
    """Pick the highest-priority available tier for op_name, honoring
    the env override if set."""
    if op_name not in _REGISTRY:
        raise KeyError(
            f"unknown op {op_name!r}; known: {sorted(_REGISTRY)}"
        )
    env_key = f"SCION_BACKEND_{op_name}"
    override = os.environ.get(env_key)
    table = _REGISTRY[op_name]
    if override:
        fn = table.get(override)
        if fn is None:
            raise RuntimeError(
                f"env {env_key}={override} but no {override} backend "
                f"registered for op {op_name!r}"
            )
        return fn
    for tier in ("cuda", "triton", "python"):
        fn = table.get(tier)
        if fn is not None:
            return fn
    raise RuntimeError(f"no backend available for op {op_name!r}")


# Cache the resolution per op-name; flush via reset_backend_cache().
_RESOLVED: Dict[str, Callable] = {}


def dispatch(op_name: str, *args, **kwargs):
    fn = _RESOLVED.get(op_name)
    if fn is None:
        fn = _resolve(op_name)
        _RESOLVED[op_name] = fn
    return fn(*args, **kwargs)


def resolved_tiers() -> Dict[str, str]:
    """``{op: tier}`` of every op dispatched since the last reset (run.json ``backend_tiers``)."""
    return {op: next(t for t, f in _REGISTRY[op].items() if f is fn) for op, fn in _RESOLVED.items()}


def reset_backend_cache() -> None:
    """Drop the resolved-callable cache. Useful in tests that toggle env vars."""
    _RESOLVED.clear()
