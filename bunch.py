"""Lightweight drop-in replacement for the unmaintained ``bunch`` package."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class Bunch(dict):
    """Dictionary subclass that exposes keys as attributes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self) -> "Bunch":
        return Bunch(super().copy())

    def deepcopy(self) -> "Bunch":
        return Bunch(deepcopy(dict(self)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Bunch":
        return cls(**mapping)
