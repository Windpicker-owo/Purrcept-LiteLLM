"""Build request-scoped LiteLLM OpenTelemetry instrumentation.

This module owns the small, stable configuration surface exposed to runtimes.
It deliberately does not configure a process-wide OpenTelemetry provider or
LiteLLM callback registry. :func:`create_opentelemetry_callback` imports the
optional OpenTelemetry stack only when observability is enabled, creates a
private provider through LiteLLM's built-in callback, and returns that callback
for injection through :class:`purrcept_litellm.LiteLLMBackend`.

The callback owns a batch span processor and therefore has a lifecycle. Runtime
shutdown should call ``force_flush`` and then ``shutdown`` after model work has
stopped. ``SPAN_ONLY`` intentionally exports prompts and completions as span
attributes; ``NO_CONTENT`` retains operational metadata without message bodies.
The derived callback also adds the small OpenInference surface required for LLM
viewers while leaving LiteLLM's GenAI attributes authoritative.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Literal, Protocol, cast
from urllib.parse import urlsplit

OpenTelemetryExporter = Literal["console", "otlp_http"]
OpenTelemetryCaptureMode = Literal["NO_CONTENT", "SPAN_ONLY"]

_INSTALL_MESSAGE = (
    "OpenTelemetry support is not installed. Install "
    "`purrcept_litellm[otel]` before enabling LiteLLM observability."
)
_COMPATIBILITY_MESSAGE = (
    "The installed LiteLLM version does not expose the required OpenTelemetry callback API."
)
_ATTRIBUTE_COMPATIBILITY_MESSAGE = (
    "The installed LiteLLM version does not expose the required OpenTelemetry "
    "set_attributes callback hook."
)


class LiteLLMOpenTelemetryError(RuntimeError):
    """Base error for creating the optional LiteLLM OpenTelemetry callback."""


class LiteLLMOpenTelemetryDependencyError(LiteLLMOpenTelemetryError):
    """Raised when the optional OpenTelemetry SDK or exporter is unavailable."""


class LiteLLMOpenTelemetryCompatibilityError(LiteLLMOpenTelemetryError):
    """Raised when LiteLLM lacks the callback surface this adapter requires."""


class LiteLLMOpenTelemetryCallback(Protocol):
    """Lifecycle surface added to LiteLLM's built-in callback instance.

    LiteLLM calls the concrete custom logger for each request. The owning
    runtime calls these two methods only during orderly shutdown, after no new
    request can use the callback.
    """

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Wait for completed spans to reach the configured exporter."""

        ...

    def shutdown(self) -> None:
        """Release the callback's private tracer provider and exporter."""

        ...


class _TracerProviderLifecycle(Protocol):
    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Flush completed spans within the caller's time budget."""

        ...

    def shutdown(self) -> None:
        """Release exporter resources."""

        ...


class _AttributeSpan(Protocol):
    """Minimal mutable span surface used before LiteLLM ends a request span."""

    def set_attribute(self, key: str, value: str) -> object:
        """Attach one string attribute to the active span."""

        ...


@dataclass(frozen=True, slots=True)
class LiteLLMOpenTelemetryConfig:
    """Immutable configuration for one private LiteLLM trace exporter.

    ``headers`` follows LiteLLM and the OTLP environment-variable syntax, for
    example ``"Authorization=Bearer%20token,x-tenant=alpha"``. It is excluded
    from the dataclass representation because authentication headers are
    secret-bearing configuration. Console export accepts neither an endpoint
    nor headers; OTLP/HTTP requires an explicit HTTP(S) endpoint.
    """

    exporter: OpenTelemetryExporter = field(default="console", kw_only=True)
    endpoint: str | None = field(default=None, kw_only=True)
    headers: str | None = field(default=None, repr=False, kw_only=True)
    service_name: str = field(default="purrcept", kw_only=True)
    environment: str | None = field(default=None, kw_only=True)
    capture_message_content: OpenTelemetryCaptureMode = field(
        default="NO_CONTENT",
        kw_only=True,
    )

    def __post_init__(self) -> None:
        _validate_exporter(self.exporter)
        _validate_optional_string(self.endpoint, field_name="endpoint")
        _validate_optional_string(self.headers, field_name="headers")
        _validate_required_string(self.service_name, field_name="service_name")
        _validate_optional_string(self.environment, field_name="environment")
        _validate_capture_mode(self.capture_message_content)

        if self.exporter == "console":
            if self.endpoint is not None or self.headers is not None:
                raise ValueError("console export does not accept endpoint or headers.")
            return

        if self.endpoint is None:
            raise ValueError("otlp_http export requires endpoint.")
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute HTTP(S) URL.")


def create_opentelemetry_callback(
    config: LiteLLMOpenTelemetryConfig,
) -> LiteLLMOpenTelemetryCallback:
    """Create an isolated instance of LiteLLM's built-in OTel callback.

    Imports are lazy so the base package remains usable without OpenTelemetry.
    The generated subclass suppresses LiteLLM Proxy registration and always
    asks constructor hooks for a new provider instead of consulting or setting
    process-global OpenTelemetry providers. The returned callback must be passed
    to ``LiteLLMBackend(callbacks=(callback,))``.

    Raises:
        TypeError: If ``config`` is not a :class:`LiteLLMOpenTelemetryConfig`.
        LiteLLMOpenTelemetryDependencyError: If the ``otel`` extra is absent.
        LiteLLMOpenTelemetryCompatibilityError: If LiteLLM's supported callback
            classes are missing.
    """

    if not isinstance(cast(object, config), LiteLLMOpenTelemetryConfig):
        raise TypeError("config must be a LiteLLMOpenTelemetryConfig.")

    integration = _load_optional_modules(config)
    try:
        callback_base = cast(type[object], integration.__dict__["OpenTelemetry"])
        callback_config_type = cast(
            Callable[..., object],
            integration.__dict__["OpenTelemetryConfig"],
        )
    except KeyError as error:
        raise LiteLLMOpenTelemetryCompatibilityError(_COMPATIBILITY_MESSAGE) from error

    callback_type = _isolated_callback_type(
        callback_base,
        capture_message_content=config.capture_message_content,
    )
    native_config = callback_config_type(
        exporter=config.exporter,
        endpoint=config.endpoint,
        headers=config.headers,
        enable_metrics=False,
        enable_events=False,
        service_name=config.service_name,
        deployment_environment=config.environment,
        skip_set_global=True,
        capture_message_content=config.capture_message_content,
    )
    try:
        callback_factory = cast(Callable[..., object], callback_type)
        callback = callback_factory(config=native_config)
    except ImportError as error:
        raise LiteLLMOpenTelemetryDependencyError(_INSTALL_MESSAGE) from error
    return cast(LiteLLMOpenTelemetryCallback, callback)


def _load_optional_modules(config: LiteLLMOpenTelemetryConfig) -> ModuleType:
    """Verify optional dependencies before importing LiteLLM's integration."""

    modules = ["opentelemetry.trace", "opentelemetry.sdk.trace"]
    if config.exporter == "otlp_http":
        modules.append("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    try:
        for module_name in modules:
            import_module(module_name)
        return import_module("litellm.integrations.opentelemetry")
    except ImportError as error:
        raise LiteLLMOpenTelemetryDependencyError(_INSTALL_MESSAGE) from error


def _isolated_callback_type(
    callback_base: type[object],
    *,
    capture_message_content: OpenTelemetryCaptureMode,
) -> type[object]:
    """Derive a callback with isolated ownership and OpenInference enrichment.

    LiteLLM's generic callback already emits the standard ``gen_ai.*`` message
    payloads. Phoenix can translate those messages, but its LLM-specific UI is
    selected by ``openinference.span.kind``. The derived method therefore calls
    LiteLLM first and adds only the missing classification, top-level
    input/output values, and tool schemas. It deliberately does not mirror the
    complete message sequence under ``llm.input_messages``; those messages
    remain authoritative in the GenAI attributes and Phoenix expands them during
    ingestion.
    """

    candidate = getattr(callback_base, "set_attributes", None)
    if not callable(candidate):
        raise LiteLLMOpenTelemetryCompatibilityError(_ATTRIBUTE_COMPATIBILITY_MESSAGE)
    base_set_attributes = cast(Callable[[object, object, object, object], None], candidate)

    def set_attributes(
        callback: object,
        span: object,
        kwargs: object,
        response_obj: object,
    ) -> None:
        """Preserve LiteLLM attributes before adding OpenInference UI semantics."""

        base_set_attributes(callback, span, kwargs, response_obj)
        try:
            _set_openinference_attributes(
                span,
                kwargs,
                response_obj,
                capture_message_content=capture_message_content,
            )
        except Exception:
            # LiteLLM's GenAI attributes are already committed above. A future
            # provider wrapper may expose Mapping-like methods that raise in
            # unexpected ways; optional UI enrichment must never leave the
            # request span unfinished or change the model-call outcome.
            return

    return type(
        "PurrceptIsolatedOpenTelemetry",
        (callback_base,),
        {
            "_get_or_create_provider": _create_private_provider,
            "_init_otel_logger_on_litellm_proxy": _skip_proxy_registration,
            "force_flush": _force_flush,
            "set_attributes": set_attributes,
            "shutdown": _shutdown,
        },
    )


def _set_openinference_attributes(
    span: object,
    kwargs: object,
    response_obj: object,
    *,
    capture_message_content: OpenTelemetryCaptureMode,
) -> None:
    """Make a generic LiteLLM span render as an LLM call in OpenInference UIs.

    Classification is safe without message capture and remains available for
    operational traces. Prompt summaries, response summaries, and advertised
    tools are content-bearing, so ``NO_CONTENT`` must stop before any of them
    are inspected or exported.
    """

    writable_span = cast(_AttributeSpan, span)
    _safe_set_attribute(writable_span, "openinference.span.kind", "LLM")

    if capture_message_content != "SPAN_ONLY":
        return

    request = _string_mapping(kwargs) or {}
    input_summary = _last_message_summary(request.get("messages"))
    if input_summary is not None:
        input_value, input_mime_type = input_summary
        _safe_set_attribute(writable_span, "input.value", input_value)
        _safe_set_attribute(writable_span, "input.mime_type", input_mime_type)

    output_summary = _first_response_summary(response_obj)
    if output_summary is not None:
        output_value, output_mime_type = output_summary
        _safe_set_attribute(writable_span, "output.value", output_value)
        _safe_set_attribute(writable_span, "output.mime_type", output_mime_type)

    optional_params = request.get("optional_params")
    optional_parameter_map = _string_mapping(optional_params)
    if optional_parameter_map is None:
        return
    tools = _object_sequence(optional_parameter_map.get("tools"))
    if tools is None:
        return
    for index, tool in enumerate(tools):
        tool_definition = _string_mapping(tool)
        if tool_definition is None:
            continue
        serialized_tool = _json_attribute(tool_definition)
        if serialized_tool is None:
            continue
        key = f"llm.tools.{index}.tool.json_schema"
        _safe_set_attribute(writable_span, key, serialized_tool)


def _last_message_summary(messages: object) -> tuple[str, str] | None:
    """Return the latest message content used as Phoenix's compact input row."""

    message_sequence = _object_sequence(messages)
    if not message_sequence:
        return None
    last_message = message_sequence[-1]
    message = _string_mapping(last_message)
    if message is None:
        return None

    content = message.get("content")
    if content is not None:
        return _openinference_value(content)
    tool_calls = message.get("tool_calls")
    return None if tool_calls is None else _openinference_value(tool_calls)


def _first_response_summary(response_obj: object) -> tuple[str, str] | None:
    """Return assistant text, or its tool calls when the response has no text."""

    choices = _object_sequence(_get_value(response_obj, "choices"))
    if not choices:
        return None
    message = _get_value(choices[0], "message")
    if message is None:
        return None

    content = _get_value(message, "content")
    if content is not None and content != "":
        return _openinference_value(content)
    tool_calls = _get_value(message, "tool_calls")
    if tool_calls is not None:
        return _openinference_value(tool_calls)
    return _openinference_value(content) if content == "" else None


def _openinference_value(value: object) -> tuple[str, str] | None:
    """Serialize one OpenInference summary without falling back to ``repr``."""

    if isinstance(value, str):
        return value, "text/plain"
    serialized = _json_attribute(value)
    if serialized is None:
        return None
    return serialized, "application/json"


def _json_attribute(value: object) -> str | None:
    """Serialize JSON-compatible LiteLLM values while supporting Pydantic models."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_model_json_value,
        )
    except (TypeError, ValueError):
        return None


def _model_json_value(value: object) -> object:
    """Expose a model's public JSON data or reject an unknown object shape."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _get_value(container: object, key: str) -> object:
    """Read mapping-like LiteLLM response objects without depending on their type."""

    mapping = _string_mapping(container)
    if mapping is not None:
        return mapping.get(key)
    getter = getattr(container, "get", None)
    if not callable(getter):
        return None
    try:
        return getter(key)
    except (KeyError, TypeError):
        return None


def _object_sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(Sequence[object], value)
    return None


def _string_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _safe_set_attribute(span: _AttributeSpan, key: str, value: str) -> None:
    """Keep optional UI enrichment from changing the model request outcome."""

    try:
        span.set_attribute(key, value)
    except Exception:
        # Telemetry backends are optional observers. LiteLLM's authoritative
        # GenAI attributes and, more importantly, the model result must survive
        # an exporter-specific attribute rejection.
        return


def _create_private_provider(
    _callback: object,
    provider: object,
    provider_name: str,
    get_existing_provider_fn: Callable[[], object],
    sdk_provider_class: type[object],
    create_new_provider_fn: Callable[[], object],
    set_provider_fn: Callable[[object], object],
    skip_set_global: bool = False,
) -> object:
    """Honor injection or create a provider without reading or setting globals."""

    del (
        provider_name,
        get_existing_provider_fn,
        sdk_provider_class,
        set_provider_fn,
        skip_set_global,
    )
    return provider if provider is not None else create_new_provider_fn()


def _skip_proxy_registration(_callback: object) -> None:
    """Keep the request-scoped callback out of LiteLLM Proxy registries."""


def _force_flush(callback: object, timeout_millis: int = 30_000) -> bool:
    """Flush the private tracer provider owned by ``callback``."""

    provider = cast(_TracerProviderLifecycle, vars(callback)["_tracer_provider"])
    result = provider.force_flush(timeout_millis)
    return bool(result)


def _shutdown(callback: object) -> None:
    """Shut down the private tracer provider owned by ``callback``."""

    provider = cast(_TracerProviderLifecycle, vars(callback)["_tracer_provider"])
    provider.shutdown()


def _validate_exporter(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("exporter must be a string.")
    if value not in {"console", "otlp_http"}:
        raise ValueError("exporter must be 'console' or 'otlp_http'.")


def _validate_capture_mode(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("capture_message_content must be a string.")
    if value not in {"NO_CONTENT", "SPAN_ONLY"}:
        raise ValueError("capture_message_content must be 'NO_CONTENT' or 'SPAN_ONLY'.")


def _validate_required_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _validate_optional_string(value: object, *, field_name: str) -> None:
    if value is None:
        return
    _validate_required_string(value, field_name=field_name)


__all__ = [
    "LiteLLMOpenTelemetryCallback",
    "LiteLLMOpenTelemetryCompatibilityError",
    "LiteLLMOpenTelemetryConfig",
    "LiteLLMOpenTelemetryDependencyError",
    "LiteLLMOpenTelemetryError",
    "OpenTelemetryCaptureMode",
    "OpenTelemetryExporter",
    "create_opentelemetry_callback",
]
