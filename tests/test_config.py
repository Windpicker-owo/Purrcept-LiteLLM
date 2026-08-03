from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from purrcept_core.models import JsonValue

from purrcept_litellm import LiteLLMBackend, LiteLLMConfig


def test_config_is_frozen_and_takes_a_deep_snapshot() -> None:
    nested: dict[str, JsonValue] = {"headers": {"x-tenant": "alpha"}, "tags": ["one"]}
    config = LiteLLMConfig(
        api_key="secret",
        api_base="https://example.invalid/v1",
        api_version="2026-07-28",
        timeout=30,
        allow_strict_prompt_cache=True,
        default_options=nested,
    )
    nested["headers"] = {"changed": True}

    assert config.api_key == "secret"
    assert config.api_base == "https://example.invalid/v1"
    assert config.api_version == "2026-07-28"
    assert config.timeout == 30
    assert config.allow_strict_prompt_cache is True
    assert config.default_options["headers"] == {"x-tenant": "alpha"}
    assert config.default_options["tags"] == ("one",)
    assert "secret" not in repr(config)
    with pytest.raises(TypeError):
        cast(dict[str, object], config.default_options)["new"] = True
    with pytest.raises(FrozenInstanceError):
        config.timeout = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"api_key": cast(str, 1)}, TypeError, "api_key"),
        ({"api_key": ""}, ValueError, "api_key"),
        ({"api_base": cast(str, 1)}, TypeError, "api_base"),
        ({"api_base": ""}, ValueError, "api_base"),
        ({"api_version": cast(str, 1)}, TypeError, "api_version"),
        ({"api_version": ""}, ValueError, "api_version"),
        ({"timeout": cast(float, True)}, TypeError, "timeout"),
        ({"timeout": float("inf")}, ValueError, "finite"),
        ({"timeout": 0}, ValueError, "greater than zero"),
        (
            {"allow_strict_prompt_cache": cast(bool, 1)},
            TypeError,
            "allow_strict_prompt_cache",
        ),
        (
            {"default_options": cast(dict[str, JsonValue], object())},
            TypeError,
            "mapping",
        ),
        (
            {"default_options": {"bad": float("nan")}},
            ValueError,
            "finite JSON",
        ),
        (
            {"default_options": cast(dict[str, JsonValue], {1: "bad"})},
            TypeError,
            "string object keys",
        ),
        (
            {"default_options": cast(dict[str, JsonValue], {"bad": object()})},
            TypeError,
            "unsupported JSON",
        ),
    ],
)
def test_config_validates_public_boundaries(
    kwargs: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        LiteLLMConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_reference_cycles() -> None:
    cyclic: dict[str, JsonValue] = {}
    cyclic["self"] = cyclic

    with pytest.raises(ValueError, match="reference cycle"):
        LiteLLMConfig(default_options=cyclic)


def test_backend_validates_completion_callable() -> None:
    with pytest.raises(TypeError, match="asynchronous callable"):
        LiteLLMBackend(completion=cast(object, 1))  # type: ignore[arg-type]


def test_backend_exposes_its_immutable_config() -> None:
    backend = LiteLLMBackend(api_base="https://example.invalid")

    assert backend.config.api_base == "https://example.invalid"
    assert backend.callbacks == ()
