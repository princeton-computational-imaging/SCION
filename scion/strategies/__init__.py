"""Pluggable density-control strategies for CodebookModel.

One strategy per level (splat, instance), all inheriting from
CodebookStrategy. CompositeStrategy wires them together.
"""

from scion.strategies.base import CodebookStrategy
from scion.strategies.splat import MHSplatStrategy
from scion.strategies.instance import DefaultInstanceStrategy
from scion.strategies.composite import CompositeStrategy

__all__ = [
    "CodebookStrategy",
    "MHSplatStrategy",
    "DefaultInstanceStrategy",
    "CompositeStrategy",
]
