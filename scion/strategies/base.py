"""Base class for all codebook densification strategies.

Every strategy — regardless of level (splat, candidate, instance) — inherits
from CodebookStrategy.  This ensures any strategy is a drop-in replacement
in a CompositeStrategy slot.

Lifecycle hooks, called by :class:`scion.engine.trainer.Trainer`:
  step_pre_forward   → before the forward pass
  step_post_backward → after backward + optimizer step
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from scion.camera import Camera

if TYPE_CHECKING:
    from scion.models.codebook_base import CodebookMixin


class CodebookStrategy:
    """Base class for all codebook densification strategies."""

    # Name of the config/strategies/<level>/<name>.yaml whose defaults are
    # merged into the model config when this class is selected (None = none).
    config_name: Optional[str] = None

    def initialize_state(self, model: CodebookMixin) -> dict:
        """Called once after model init. Returns mutable state dict."""
        return {}

    def step_pre_forward(
        self,
        model: CodebookMixin,
        step: int,
        state: dict,
        *,
        camera: Optional[Camera] = None,
        batch: Optional[dict] = None,
    ) -> Dict[str, float]:
        """Structural changes before the forward pass.

        Args:
            model: The CodebookMixin instance.
            step: Current training step.
            state: Mutable strategy state (persists across steps).
            camera: ``scion.Camera`` from previous step (for MH
                re-rendering).
            batch: Training batch from previous step.

        Returns:
            Dict of metrics to log.
        """
        return {}

    def step_post_backward(
        self,
        model: CodebookMixin,
        step: int,
        state: dict,
        *,
        optimizers: Any = None,
        lr: float = 0.0,
    ) -> Dict[str, float]:
        """After ``loss.backward()``: accumulate per-row / per-instance signals.

        Args:
            model: The CodebookMixin instance.
            step: Current training step.
            state: Mutable strategy state (persists across steps).
            optimizers: Object whose ``.optimizers`` dict maps param-group
                name to its Adam (for reading Adam state).
            lr: Current ``splat_means`` learning rate.

        Returns:
            Dict of metrics to log.
        """
        return {}


def strategy_class(spec: str, key: str) -> type:
    """Resolve a ``"pkg.mod:Class"`` strategy spec from config ``key``; fail early and clearly."""
    import importlib

    mod, _, cls = spec.partition(":")
    try:
        obj = getattr(importlib.import_module(mod), cls)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"{key}={spec!r}: cannot import {cls!r} from {mod!r} ({e})") from e
    if not (isinstance(obj, type) and issubclass(obj, CodebookStrategy)):
        raise TypeError(f"{key}={spec!r}: {obj!r} is not a CodebookStrategy subclass")
    return obj
