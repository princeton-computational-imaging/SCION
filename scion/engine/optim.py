"""Optimizer rebinding after an instance resize."""

from __future__ import annotations

from typing import Any, Dict, Optional


def rebind_params(
    model: Any,
    optimizers: Any,
    strategy_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """Apply pending resize moves on the model and rebind optimizer state.

    An instance resize (growth / culling) replaces the instance tensors
    with ones of the new shape. The optimizer is still holding references
    to the *old* ``Parameter`` objects in ``param_groups`` and keyed by
    the old ones in ``state``. This function:

    1. Detects a pending resize via ``model._instance_size_changed``.
    2. Calls :meth:`CodebookMixin.replace_instance_params` to swap
       parameters on the model (which also rebinds the optimizer state).
    3. Recomputes assembly indices so downstream renders pick up the
       new sizes.

    When ``strategy_state`` is provided, the per-instance ``grad_ema``
    accumulator (read by :class:`DefaultInstanceStrategy`) is cleared on
    instance resize — its row indexing no longer matches.

    Args:
        model: :class:`CodebookModel`.
        optimizers: An object whose ``optimizers`` attribute is a
            ``Dict[str, torch.optim.Optimizer]`` (the Trainer's
            ``_OptimizerShim``), OR a plain :class:`torch.optim.Optimizer`.
        strategy_state: Optional per-strategy state dict.

    Returns:
        True iff any rebinding happened.
    """
    if not getattr(model, "_instance_size_changed", False):
        return False

    model.replace_instance_params(optimizers)
    model._instance_size_changed = False
    if strategy_state is not None:
        strategy_state["grad_ema"] = {}
    if hasattr(model, "_cached_grad_ema"):
        model._cached_grad_ema = {}

    model.recompute_assembly_indices()
    return True


__all__ = ["rebind_params"]
