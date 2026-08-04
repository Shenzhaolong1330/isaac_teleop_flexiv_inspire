"""Small transport-independent registries for policy tensor mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionTensorMapping:
    schema_id: str
    input_dimension: int
    transform: Callable[[Sequence[float]], Sequence[float]]

    def __post_init__(self) -> None:
        if not str(self.schema_id).strip():
            raise ValueError("action mapping schema_id must be non-empty")
        if int(self.input_dimension) <= 0:
            raise ValueError("action mapping input_dimension must be positive")
        if not callable(self.transform):
            raise ValueError("action mapping transform must be callable")


class ActionMappingRegistry:
    """Map versioned policy actions into one station canonical action tensor."""

    def __init__(self, *, canonical_dimension: int) -> None:
        if int(canonical_dimension) <= 0:
            raise ValueError("canonical_dimension must be positive")
        self.canonical_dimension = int(canonical_dimension)
        self._mappings: dict[str, ActionTensorMapping] = {}

    @property
    def schema_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._mappings))

    def register(
        self,
        schema_id: str,
        *,
        input_dimension: int,
        transform: Callable[[Sequence[float]], Sequence[float]],
    ) -> "ActionMappingRegistry":
        mapping = ActionTensorMapping(schema_id, input_dimension, transform)
        if mapping.schema_id in self._mappings:
            raise ValueError(f"duplicate action mapping: {mapping.schema_id}")
        self._mappings[mapping.schema_id] = mapping
        return self

    def map(self, schema_id: str, value: Sequence[float]) -> np.ndarray:
        try:
            mapping = self._mappings[str(schema_id)]
        except KeyError as exc:
            raise ValueError(f"action schema has no canonical mapping: {schema_id}") from exc
        source = np.asarray(value, dtype=np.float64).reshape(-1)
        if source.shape != (mapping.input_dimension,) or not np.all(np.isfinite(source)):
            raise ValueError(
                f"action {mapping.schema_id} must contain "
                f"{mapping.input_dimension} finite values"
            )
        result = np.asarray(mapping.transform(source), dtype=np.float64).reshape(-1)
        if result.shape != (self.canonical_dimension,) or not np.all(np.isfinite(result)):
            raise ValueError(
                f"canonical action must contain {self.canonical_dimension} finite values"
            )
        return result
