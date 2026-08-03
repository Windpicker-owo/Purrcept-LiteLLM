from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
)
from purrcept_core.models import (
    ModelAuthenticationError,
    ModelContextWindowError,
    ModelError,
    ModelRateLimitError,
    ModelRequestError,
    ModelUnavailableError,
)

from purrcept_litellm import LiteLLMBackend, map_litellm_error
from purrcept_litellm.errors import (
    _is_instance,  # pyright: ignore[reportPrivateUsage]
    _retry_after,  # pyright: ignore[reportPrivateUsage]
    cast_header,
)

from .helpers import CaptureCompletion


def test_error_mapping_preserves_existing_model_errors() -> None:
    original = ModelRequestError("already mapped")

    assert map_litellm_error(original, model="model") is original


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            lambda: ContextWindowExceededError("too large", "m", "fixture"),
            ModelContextWindowError,
        ),
        (
            lambda: AuthenticationError("bad key", "fixture", "m"),
            ModelAuthenticationError,
        ),
        (
            lambda: BadRequestError("bad", "m", "fixture"),
            ModelRequestError,
        ),
        (
            lambda: NotFoundError("missing", "m", "fixture"),
            ModelRequestError,
        ),
        (
            lambda: UnsupportedParamsError("unsupported", "fixture", "m"),
            ModelRequestError,
        ),
        (
            lambda: ContentPolicyViolationError("blocked", "m", "fixture"),
            ModelRequestError,
        ),
        (
            lambda: Timeout("timeout", "m", "fixture"),
            ModelUnavailableError,
        ),
        (
            lambda: APIConnectionError("offline", "fixture", "m"),
            ModelUnavailableError,
        ),
        (
            lambda: ServiceUnavailableError("down", "fixture", "m"),
            ModelUnavailableError,
        ),
        (
            lambda: InternalServerError("broken", "fixture", "m"),
            ModelUnavailableError,
        ),
        (
            lambda: APIError(503, "broken", "fixture", "m"),
            ModelUnavailableError,
        ),
        (
            lambda: APIError(409, "conflict", "fixture", "m"),
            ModelError,
        ),
        (
            lambda: ValueError("unknown"),
            ModelError,
        ),
        (
            lambda: ValueError(""),
            ModelError,
        ),
    ],
)
def test_litellm_errors_map_to_provider_neutral_taxonomy(
    factory: Callable[[], Exception],
    expected: type[ModelError],
) -> None:
    raw = factory()

    mapped = map_litellm_error(raw, model="requested")

    assert type(mapped) is expected
    assert mapped.model == "requested"
    assert mapped.provider == ("fixture" if not isinstance(raw, ValueError) else "litellm")
    assert str(mapped)


def test_rate_limit_maps_retry_after_from_direct_value_or_header() -> None:
    direct = RateLimitError("slow", "fixture", "model")
    direct.__dict__["retry_after"] = 2
    mapped_direct = map_litellm_error(direct, model="requested")

    response = httpx.Response(
        429,
        headers={"Retry-After": "3.5"},
        request=httpx.Request("POST", "https://example.invalid"),
    )
    header = RateLimitError("slow", "fixture", "model", response=response)
    mapped_header = map_litellm_error(header, model="requested")

    assert isinstance(mapped_direct, ModelRateLimitError)
    assert mapped_direct.retry_after_seconds == 2
    assert isinstance(mapped_header, ModelRateLimitError)
    assert mapped_header.retry_after_seconds == 3.5


@pytest.mark.parametrize("value", [-1, float("inf"), "later", True])
def test_invalid_retry_after_values_are_ignored(value: object) -> None:
    error = RateLimitError("slow", "fixture", "model")
    error.__dict__["retry_after"] = value

    mapped = map_litellm_error(error, model="requested")

    assert isinstance(mapped, ModelRateLimitError)
    assert mapped.retry_after_seconds is None


def test_invalid_retry_after_headers_are_ignored() -> None:
    response = httpx.Response(
        429,
        headers={"retry-after": "later"},
        request=httpx.Request("POST", "https://example.invalid"),
    )
    error = RateLimitError("slow", "fixture", "model", response=response)

    mapped = map_litellm_error(error, model="requested")

    assert isinstance(mapped, ModelRateLimitError)
    assert mapped.retry_after_seconds is None
    assert cast_header({"Retry-After": 3}, "retry-after") is None
    assert cast_header({1: "3"}, "retry-after") is None
    assert _retry_after(ValueError("no response")) is None
    assert not _is_instance(ValueError("unknown"), "MissingLiteLLMError")


async def test_backend_maps_completion_errors_with_the_original_cause() -> None:
    raw = AuthenticationError("bad key", "fixture", "provider-model")
    backend = LiteLLMBackend(completion=CaptureCompletion(raw))

    with pytest.raises(ModelAuthenticationError) as caught:
        from purrcept_core.models import Message, ModelRequest

        await backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="requested",
        )

    assert caught.value.__cause__ is raw
