"""Map LiteLLM's OpenAI-compatible failures to Purrcept model errors."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import cast

import litellm
from purrcept_core.models import (
    ModelAuthenticationError,
    ModelContextWindowError,
    ModelError,
    ModelRateLimitError,
    ModelRequestError,
    ModelUnavailableError,
)

from ._values import non_empty_string, read


def map_litellm_error(error: Exception, *, model: str) -> ModelError:
    """Return the most specific provider-neutral error for a LiteLLM failure."""

    if isinstance(error, ModelError):
        return error
    message = str(error).strip() or type(error).__name__
    provider = _provider(error)

    if _is_instance(error, "ContextWindowExceededError"):
        return ModelContextWindowError(message, provider=provider, model=model)
    if _is_instance(error, "AuthenticationError"):
        return ModelAuthenticationError(message, provider=provider, model=model)
    if _is_instance(error, "RateLimitError"):
        return ModelRateLimitError(
            message,
            retry_after_seconds=_retry_after(error),
            provider=provider,
            model=model,
        )
    if _is_instance(
        error,
        "BadRequestError",
        "ContentPolicyViolationError",
        "NotFoundError",
        "UnsupportedParamsError",
    ):
        return ModelRequestError(message, provider=provider, model=model)
    if _is_instance(
        error,
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
    ):
        return ModelUnavailableError(message, provider=provider, model=model)
    if _is_instance(error, "APIError"):
        status = read(error, "status_code", None)
        if isinstance(status, int) and status >= 500:
            return ModelUnavailableError(message, provider=provider, model=model)
    return ModelError(message, provider=provider, model=model)


def _is_instance(error: Exception, *names: str) -> bool:
    classes: list[type[object]] = []
    for name in names:
        candidate = getattr(litellm, name, None)
        if isinstance(candidate, type):
            classes.append(candidate)
    return bool(classes) and isinstance(error, tuple(classes))


def _provider(error: Exception) -> str:
    direct = non_empty_string(read(error, "llm_provider", None))
    if direct is not None:
        return direct
    return "litellm"


def _retry_after(error: Exception) -> float | None:
    direct = _number(read(error, "retry_after", None))
    if direct is not None:
        return direct
    response = read(error, "response", None)
    headers = read(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    value = cast_header(cast(Mapping[object, object], headers), "retry-after")
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if isfinite(parsed) and parsed >= 0 else None


def cast_header(headers: Mapping[object, object], name: str) -> str | None:
    """Read a case-insensitive string header."""

    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == name and isinstance(value, str):
            return value
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) and number >= 0 else None


__all__ = ["map_litellm_error"]
