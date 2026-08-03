"""Immutable runtime-owned LiteLLM backend configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import cast

from purrcept_core.models import JsonValue

from ._values import freeze_json_object

PROVIDER_OPTIONS_KEY = "purrcept_litellm"


def _empty_options() -> Mapping[str, JsonValue]:
    return {}


@dataclass(frozen=True, slots=True)
class LiteLLMConfig:
    """Credentials, endpoint selection, and provider-wide LiteLLM options."""

    api_key: str | None = field(default=None, repr=False, kw_only=True)
    api_base: str | None = field(default=None, kw_only=True)
    api_version: str | None = field(default=None, kw_only=True)
    timeout: float | None = field(default=None, kw_only=True)
    allow_strict_prompt_cache: bool = field(default=False, kw_only=True)
    default_options: Mapping[str, JsonValue] = field(
        default_factory=_empty_options,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        _validate_optional_string(self.api_key, field_name="api_key")
        _validate_optional_string(self.api_base, field_name="api_base")
        _validate_optional_string(self.api_version, field_name="api_version")
        _validate_timeout(self.timeout)
        if not isinstance(cast(object, self.allow_strict_prompt_cache), bool):
            raise TypeError("allow_strict_prompt_cache must be a bool.")
        if not isinstance(cast(object, self.default_options), Mapping):
            raise TypeError("default_options must be a mapping.")
        object.__setattr__(
            self,
            "default_options",
            freeze_json_object(self.default_options, field_name="default_options"),
        )


def _validate_optional_string(value: object, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None.")
    if not value:
        raise ValueError(f"{field_name} must not be empty.")


def _validate_timeout(value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout must be a number or None.")
    if not isfinite(value):
        raise ValueError("timeout must be finite.")
    if value <= 0:
        raise ValueError("timeout must be greater than zero.")


__all__ = ["PROVIDER_OPTIONS_KEY", "LiteLLMConfig"]
