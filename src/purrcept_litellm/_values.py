"""Small dependency-neutral helpers for LiteLLM's object-or-dict values."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from types import MappingProxyType
from typing import TypeVar, cast

from purrcept_core.models import JsonValue

ValueT = TypeVar("ValueT")
MISSING = object()


def read(value: object, key: str, default: ValueT) -> object | ValueT:
    """Read one field from a mapping or SDK value object."""

    if isinstance(value, Mapping):
        return cast(Mapping[object, object], value).get(key, default)
    return getattr(value, key, default)


def sequence(value: object) -> tuple[object, ...]:
    """Normalize a non-string sequence-like SDK field."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(cast(Sequence[object], value))
    return ()


def non_empty_string(value: object) -> str | None:
    """Return only non-empty strings."""

    return value if isinstance(value, str) and value else None


def integer(value: object, *, default: int = 0) -> int:
    """Return a non-negative integer SDK count or a safe default."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def freeze_json_object(
    value: Mapping[str, JsonValue],
    *,
    field_name: str,
) -> Mapping[str, JsonValue]:
    """Take a deep immutable JSON snapshot."""

    frozen = _freeze_json(value, path=field_name, active_ids=set())
    return cast(Mapping[str, JsonValue], frozen)


def thaw_json(value: JsonValue) -> object:
    """Create ordinary dict/list containers suitable for LiteLLM."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [thaw_json(item) for item in value]
    return value


def json_object(value: object) -> dict[str, JsonValue]:
    """Project supported SDK metadata into a JSON object."""

    projected = json_value(value)
    if isinstance(projected, Mapping):
        return dict(cast(Mapping[str, JsonValue], projected))
    return {}


def json_value(value: object) -> JsonValue | None:
    """Project an SDK value to JSON, dropping unsupported values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for raw_key, raw_item in cast(Mapping[object, object], value).items():
            if not isinstance(raw_key, str):
                continue
            item = json_value(raw_item)
            if item is not None or raw_item is None:
                result[raw_key] = item
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result_items: list[JsonValue] = []
        for raw_item in cast(Sequence[object], value):
            item = json_value(raw_item)
            if item is not None or raw_item is None:
                result_items.append(item)
        return result_items
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = cast(Callable[[], object], model_dump)()
        return json_value(dumped)
    return None


def _freeze_json(
    value: object,
    *,
    path: str,
    active_ids: set[int],
) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers.")
        return value
    if isinstance(value, Mapping):
        container_id = id(cast(object, value))
        if container_id in active_ids:
            raise ValueError(f"{path} must not contain a reference cycle.")
        active_ids.add(container_id)
        result: dict[str, JsonValue] = {}
        try:
            for raw_key, raw_item in cast(Mapping[object, object], value).items():
                if not isinstance(raw_key, str):
                    raise TypeError(f"{path} must contain only string object keys.")
                result[raw_key] = _freeze_json(
                    raw_item,
                    path=f"{path}.{raw_key}",
                    active_ids=active_ids,
                )
        finally:
            active_ids.remove(container_id)
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        container_id = id(cast(object, value))
        if container_id in active_ids:
            raise ValueError(f"{path} must not contain a reference cycle.")
        active_ids.add(container_id)
        try:
            return tuple(
                _freeze_json(
                    item,
                    path=f"{path}[{index}]",
                    active_ids=active_ids,
                )
                for index, item in enumerate(cast(Sequence[object], value))
            )
        finally:
            active_ids.remove(container_id)
    raise TypeError(f"{path} contains unsupported JSON value of type {type(value).__name__}.")


__all__: list[str] = []
