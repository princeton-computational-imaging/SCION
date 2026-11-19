"""CompositeStrategy: wires one strategy per level (splat, instance)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from scion.camera import Camera
from scion.strategies.base import CodebookStrategy

if TYPE_CHECKING:
    from scion.models.codebook_base import CodebookMixin


class CompositeStrategy(CodebookStrategy):
    """Composes one strategy per level: splat and instance.

    Ordering within hooks:
      step_pre_forward:   instances first (resize), then splats (MH births)
      step_post_backward: splats (gradient caching), then instances

    The instance slot accepts a list for composing multiple instance
    strategies; canonical codebook stack passes a single instance.

    Canonical composition::

        CompositeStrategy(
            splat=MHSplatStrategy(),
            instance=DefaultInstanceStrategy(),
        )
    """

    def __init__(
        self,
        splat: Optional[CodebookStrategy] = None,
        instance: Optional[Union[CodebookStrategy, List[CodebookStrategy]]] = None,
    ):
        self.splat = splat
        if instance is None:
            self.instances: List[CodebookStrategy] = []
        elif isinstance(instance, list):
            self.instances = instance
        else:
            self.instances = [instance]

    def initialize_state(self, model: CodebookMixin) -> dict:
        state: dict = {}
        for s in self.all_strategies():
            state.update(s.initialize_state(model))
        return state

    def step_pre_forward(
        self,
        model: CodebookMixin,
        step: int,
        state: dict,
        *,
        camera: Optional[Camera] = None,
        batch: Optional[dict] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        # Instance strategies first (resize before splat births)
        for s in self.instances:
            metrics.update(
                s.step_pre_forward(model, step, state, camera=camera, batch=batch)
            )
        if self.splat:
            metrics.update(
                self.splat.step_pre_forward(
                    model, step, state, camera=camera, batch=batch
                )
            )
        return metrics

    def step_post_backward(
        self,
        model: CodebookMixin,
        step: int,
        state: dict,
        *,
        optimizers: Any = None,
        lr: float = 0.0,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.splat:
            metrics.update(
                self.splat.step_post_backward(
                    model, step, state, optimizers=optimizers, lr=lr
                )
            )
        for s in self.instances:
            metrics.update(
                s.step_post_backward(
                    model, step, state, optimizers=optimizers, lr=lr
                )
            )
        return metrics

    def all_strategies(self) -> List[CodebookStrategy]:
        """Return all non-None strategies in execution order."""
        result: List[CodebookStrategy] = list(self.instances)
        if self.splat:
            result.append(self.splat)
        return result
