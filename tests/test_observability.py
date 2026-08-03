from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import ClassVar, NoReturn, cast

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from purrcept_core.models import Message, ModelError, ModelRequest, ModelRequestError

import purrcept_litellm.observability as observability
from purrcept_litellm import (
    LiteLLMBackend,
    LiteLLMOpenTelemetryCompatibilityError,
    LiteLLMOpenTelemetryConfig,
    LiteLLMOpenTelemetryDependencyError,
    create_opentelemetry_callback,
)
from purrcept_litellm.backend import (
    _RequestCallbackCompletion,  # pyright: ignore[reportPrivateUsage]
)

from .helpers import CaptureCompletion, text_response


class _FakeProvider:
    def __init__(self) -> None:
        self.flush_timeouts: list[int] = []
        self.shutdown_calls = 0

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.flush_timeouts.append(timeout_millis)
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeNativeConfig:
    def __init__(self, **kwargs: object) -> None:
        self.values = kwargs


class _FakeSpan:
    """Record semantic attributes in the order the callback publishes them."""

    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.writes: list[tuple[str, object]] = []

    def set_attribute(self, key: str, value: object) -> None:
        """Mirror the OpenTelemetry span method used by LiteLLM callbacks."""

        self.attributes[key] = value
        self.writes.append((key, value))


class _RejectingEnrichmentSpan(_FakeSpan):
    """Accept LiteLLM data while simulating a backend rejecting UI attributes."""

    def set_attribute(self, key: str, value: object) -> None:
        """Reject only the optional OpenInference namespace and summaries."""

        if key.startswith(("openinference.", "input.", "output.", "llm.tools.")):
            raise RuntimeError("attribute rejected")
        super().set_attribute(key, value)


class _JsonModel:
    """Provide the small Pydantic-compatible serialization surface we consume."""

    def model_dump(self, *, mode: str) -> dict[str, str]:
        assert mode == "json"
        return {"model": "value"}


class _GetterFailure:
    def __init__(self, error_type: type[Exception]) -> None:
        self._error_type = error_type

    def get(self, _key: str) -> NoReturn:
        """Model a response object whose compatibility getter is unusable."""

        raise self._error_type("unavailable")


class _HookCallback:
    async def async_log_success_event(self, *_args: object, **_kwargs: object) -> None:
        pass

    def log_failure_event(self, *_args: object, **_kwargs: object) -> None:
        pass


class _SyncSuccessCallback:
    """Return synchronously from hooks to exercise the compatibility wrapper."""

    def async_log_success_event(self, value: object) -> object:
        return value

    def log_failure_event(self, value: object) -> object:
        return value


class _OTLPHandler(BaseHTTPRequestHandler):
    """Capture OTLP protobuf bodies without involving an external collector."""

    payloads: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:
        """Store one complete OTLP request and acknowledge it."""

        length = int(self.headers.get("Content-Length", "0"))
        type(self).payloads.append(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Keep the offline regression test silent."""

        del format, args


class _FakeOpenTelemetry:
    proxy_registration_calls = 0
    global_provider_reads = 0
    global_provider_writes = 0

    def __init__(self, *, config: object) -> None:
        self.config = config
        injected = _FakeProvider()
        selected = self._get_or_create_provider(  # type: ignore[attr-defined]
            injected,
            "TracerProvider",
            self._read_global,
            _FakeProvider,
            _FakeProvider,
            self._write_global,
        )
        assert selected is injected
        self._tracer_provider = self._get_or_create_provider(  # type: ignore[attr-defined]
            None,
            "TracerProvider",
            self._read_global,
            _FakeProvider,
            _FakeProvider,
            self._write_global,
            True,
        )
        self._init_otel_logger_on_litellm_proxy()  # type: ignore[attr-defined]

    async def async_log_success_event(self, *_args: object, **_kwargs: object) -> None:
        pass

    def log_failure_event(self, *_args: object, **_kwargs: object) -> None:
        pass

    def set_attributes(
        self,
        span: _FakeSpan,
        kwargs: object,
        response_obj: object,
    ) -> None:
        """Stand in for LiteLLM's native GenAI attribute publication."""

        span.set_attribute("gen_ai.operation.name", "chat")
        native_config = cast(_FakeNativeConfig, self.config)
        if native_config.values["capture_message_content"] == "SPAN_ONLY":
            span.set_attribute("gen_ai.input.messages", "native-input")
            if response_obj is not None:
                span.set_attribute("gen_ai.output.messages", "native-output")

    def safe_set_attribute(
        self,
        span: _FakeSpan,
        key: str,
        value: object,
    ) -> None:
        """Provide LiteLLM's compatibility helper to the derived callback."""

        span.set_attribute(key, value)

    @classmethod
    def _read_global(cls) -> object:
        cls.global_provider_reads += 1
        return object()

    @classmethod
    def _write_global(cls, _provider: object) -> object:
        cls.global_provider_writes += 1
        return object()


def _integration_module(*, complete: bool = True) -> ModuleType:
    module = ModuleType("litellm.integrations.opentelemetry")
    if complete:
        module.OpenTelemetry = _FakeOpenTelemetry  # type: ignore[attr-defined]
        module.OpenTelemetryConfig = _FakeNativeConfig  # type: ignore[attr-defined]
    return module


def _install_fake_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    integration: ModuleType | None = None,
) -> list[str]:
    imports: list[str] = []
    selected = _integration_module() if integration is None else integration

    def fake_import(name: str) -> ModuleType:
        imports.append(name)
        if name == "litellm.integrations.opentelemetry":
            return selected
        return ModuleType(name)

    monkeypatch.setattr(observability, "import_module", fake_import)
    return imports


def test_config_is_frozen_validated_and_hides_headers() -> None:
    config = LiteLLMOpenTelemetryConfig(
        exporter="otlp_http",
        endpoint="http://127.0.0.1:4318/v1/traces",
        headers="Authorization=Bearer%20secret",
        service_name="purrcept-engine",
        environment="development",
        capture_message_content="SPAN_ONLY",
    )

    assert config.exporter == "otlp_http"
    assert config.headers == "Authorization=Bearer%20secret"
    assert "secret" not in repr(config)
    with pytest.raises(FrozenInstanceError):
        config.environment = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"exporter": cast(str, 1)}, TypeError, "exporter"),
        ({"exporter": "grpc"}, ValueError, "console.*otlp_http"),
        ({"endpoint": "http://collector"}, ValueError, "console export"),
        ({"headers": "token=secret"}, ValueError, "console export"),
        ({"exporter": "otlp_http"}, ValueError, "requires endpoint"),
        (
            {"exporter": "otlp_http", "endpoint": "collector:4318"},
            ValueError,
            "absolute HTTP",
        ),
        (
            {"exporter": "otlp_http", "endpoint": "ftp://collector/traces"},
            ValueError,
            "absolute HTTP",
        ),
        (
            {"exporter": "otlp_http", "endpoint": "http://"},
            ValueError,
            "absolute HTTP",
        ),
        ({"service_name": cast(str, 1)}, TypeError, "service_name"),
        ({"service_name": "  "}, ValueError, "service_name"),
        ({"environment": cast(str, 1)}, TypeError, "environment"),
        ({"environment": ""}, ValueError, "environment"),
        ({"endpoint": ""}, ValueError, "endpoint"),
        ({"headers": ""}, ValueError, "headers"),
        (
            {"capture_message_content": cast(str, 1)},
            TypeError,
            "capture_message_content",
        ),
        (
            {"capture_message_content": "SPAN_AND_EVENT"},
            ValueError,
            "NO_CONTENT.*SPAN_ONLY",
        ),
    ],
)
def test_config_rejects_ambiguous_or_unsupported_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        LiteLLMOpenTelemetryConfig(**kwargs)  # type: ignore[arg-type]


def test_factory_builds_isolated_callback_with_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOpenTelemetry.global_provider_reads = 0
    _FakeOpenTelemetry.global_provider_writes = 0
    imports = _install_fake_imports(monkeypatch)
    config = LiteLLMOpenTelemetryConfig(
        exporter="otlp_http",
        endpoint="https://collector.invalid/v1/traces",
        headers="authorization=secret",
        service_name="engine",
        environment="test",
        capture_message_content="SPAN_ONLY",
    )

    callback = create_opentelemetry_callback(config)

    assert imports == [
        "opentelemetry.trace",
        "opentelemetry.sdk.trace",
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "litellm.integrations.opentelemetry",
    ]
    assert _FakeOpenTelemetry.global_provider_reads == 0
    assert _FakeOpenTelemetry.global_provider_writes == 0
    native_config = cast(_FakeNativeConfig, cast(object, callback).__dict__["config"])
    assert native_config.values == {
        "exporter": "otlp_http",
        "endpoint": "https://collector.invalid/v1/traces",
        "headers": "authorization=secret",
        "enable_metrics": False,
        "enable_events": False,
        "service_name": "engine",
        "deployment_environment": "test",
        "skip_set_global": True,
        "capture_message_content": "SPAN_ONLY",
    }
    assert callback.force_flush(1234) is True
    provider = cast(_FakeProvider, cast(object, callback).__dict__["_tracer_provider"])
    assert provider.flush_timeouts == [1234]
    callback.shutdown()
    assert provider.shutdown_calls == 1


def test_callback_keeps_genai_attributes_and_adds_openinference_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phoenix receives concise display fields without replacing LiteLLM data."""

    messages = [
        {"role": "system", "content": "stable contract"},
        {"role": "user", "content": "latest observation"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "play_action",
                "description": "Commit one action program.",
                "parameters": {
                    "type": "object",
                    "properties": {"program": {"type": "string"}},
                    "required": ["program"],
                },
            },
        }
    ]
    response: dict[str, object] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I will wait.",
                }
            }
        ]
    }
    span = _FakeSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="SPAN_ONLY")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {"messages": messages, "optional_params": {"tools": tools}},
        response,
    )

    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.input.messages"] == "native-input"
    assert span.attributes["gen_ai.output.messages"] == "native-output"
    assert span.attributes["openinference.span.kind"] == "LLM"
    assert span.attributes["input.value"] == "latest observation"
    assert span.attributes["input.mime_type"] == "text/plain"
    assert span.attributes["output.value"] == "I will wait."
    assert span.attributes["output.mime_type"] == "text/plain"
    assert json.loads(cast(str, span.attributes["llm.tools.0.tool.json_schema"])) == tools[0]

    base_write = span.writes.index(("gen_ai.operation.name", "chat"))
    openinference_write = span.writes.index(("openinference.span.kind", "LLM"))
    assert base_write < openinference_write


def test_callback_summarizes_structured_input_and_tool_only_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-text content remains lossless while using an explicit JSON MIME type."""

    structured_content = [
        {"type": "text", "text": "What is shown?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "play_action", "arguments": '{"program":"pass"}'},
        }
    ]
    span = _FakeSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="SPAN_ONLY")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {
            "messages": [{"role": "user", "content": structured_content}],
            "optional_params": {},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                }
            ]
        },
    )

    assert span.attributes["input.mime_type"] == "application/json"
    assert json.loads(cast(str, span.attributes["input.value"])) == structured_content
    assert span.attributes["output.mime_type"] == "application/json"
    assert json.loads(cast(str, span.attributes["output.value"])) == tool_calls


def test_no_content_marks_llm_without_exporting_prompt_or_tool_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phoenix classification remains available when content capture is disabled."""

    secrets = {"prompt-secret", "response-secret", "tool-secret"}
    span = _FakeSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="NO_CONTENT")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {
            "messages": [{"role": "user", "content": "prompt-secret"}],
            "optional_params": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "tool-secret",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            },
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "response-secret",
                    }
                }
            ]
        },
    )

    assert span.attributes == {
        "gen_ai.operation.name": "chat",
        "openinference.span.kind": "LLM",
    }
    exported = json.dumps(span.attributes)
    assert all(secret not in exported for secret in secrets)


@pytest.mark.parametrize(
    ("messages", "response", "optional_params"),
    cast(
        list[tuple[object, object, object]],
        [
            ([], {"choices": []}, "not-a-mapping"),
            ([42], {"choices": [{}]}, {"tools": [42, {"invalid": object()}]}),
            ([{"role": "user"}], {"choices": [{"message": {}}]}, {}),
            (
                [{"role": "user", "content": object()}],
                {"choices": [{"message": {"content": object()}}]},
                {},
            ),
        ],
    ),
)
def test_callback_ignores_incomplete_or_unserializable_provider_shapes(
    monkeypatch: pytest.MonkeyPatch,
    messages: object,
    response: object,
    optional_params: object,
) -> None:
    """Provider drift must not turn optional trace enrichment into a request error."""

    span = _FakeSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="SPAN_ONLY")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {"messages": messages, "optional_params": optional_params},
        response,
    )

    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["openinference.span.kind"] == "LLM"
    assert "input.value" not in span.attributes
    assert "output.value" not in span.attributes
    assert not any(key.startswith("llm.tools.") for key in span.attributes)


def test_callback_uses_tool_calls_for_latest_input_and_preserves_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool history and an intentional empty completion remain visible summaries."""

    input_tool_calls = [{"function": {"name": "observed_action"}}]
    span = _FakeSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="SPAN_ONLY")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {
            "messages": [{"role": "assistant", "tool_calls": input_tool_calls}],
            "optional_params": {},
        },
        {"choices": [{"message": {"content": ""}}]},
    )

    assert json.loads(cast(str, span.attributes["input.value"])) == input_tool_calls
    assert span.attributes["input.mime_type"] == "application/json"
    assert span.attributes["output.value"] == ""
    assert span.attributes["output.mime_type"] == "text/plain"


def test_callback_serializes_model_values_and_skips_circular_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic-like values are supported while circular data is safely omitted."""

    circular: list[object] = []
    circular.append(circular)
    span = _FakeSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="SPAN_ONLY")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {
            "messages": [{"role": "user", "content": _JsonModel()}],
            "optional_params": {"tools": [{"circular": circular}]},
        },
        None,
    )

    assert json.loads(cast(str, span.attributes["input.value"])) == {"model": "value"}
    assert "llm.tools.0.tool.json_schema" not in span.attributes


@pytest.mark.parametrize(
    "response",
    [
        object(),
        _GetterFailure(KeyError),
        _GetterFailure(TypeError),
        _GetterFailure(RuntimeError),
    ],
)
def test_callback_tolerates_response_objects_without_a_readable_getter(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    """Unknown SDK response wrappers omit output summaries without failing."""

    span = _FakeSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="SPAN_ONLY")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {"messages": [], "optional_params": {}},
        response,
    )

    assert "output.value" not in span.attributes
    assert span.attributes["openinference.span.kind"] == "LLM"


def test_callback_survives_span_rejection_of_optional_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A telemetry backend rejection cannot replace a successful model result."""

    span = _RejectingEnrichmentSpan()
    _install_fake_imports(monkeypatch)
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(capture_message_content="SPAN_ONLY")
    )

    cast(_FakeOpenTelemetry, callback).set_attributes(
        span,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "optional_params": {"tools": [{"type": "function", "function": {"name": "act"}}]},
        },
        {"choices": [{"message": {"content": "hello back"}}]},
    )

    assert span.attributes == {
        "gen_ai.operation.name": "chat",
        "gen_ai.input.messages": "native-input",
        "gen_ai.output.messages": "native-output",
    }


def test_console_factory_does_not_require_otlp_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _install_fake_imports(monkeypatch)

    create_opentelemetry_callback(LiteLLMOpenTelemetryConfig())

    assert imports == [
        "opentelemetry.trace",
        "opentelemetry.sdk.trace",
        "litellm.integrations.opentelemetry",
    ]


def test_factory_reports_stable_missing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(_name: str) -> ModuleType:
        raise ModuleNotFoundError("provider-specific detail")

    monkeypatch.setattr(observability, "import_module", missing_import)

    with pytest.raises(
        LiteLLMOpenTelemetryDependencyError,
        match=r"purrcept_litellm\[otel\]",
    ) as caught:
        create_opentelemetry_callback(LiteLLMOpenTelemetryConfig())

    assert "provider-specific detail" not in str(caught.value)


def test_factory_translates_dependency_failure_during_callback_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenOpenTelemetry:
        def __init__(self, *, config: object) -> None:
            del config
            raise ImportError("late optional import")

        def set_attributes(self, *_args: object, **_kwargs: object) -> None:
            """Expose the supported hook so construction reaches the dependency error."""

    integration = _integration_module()
    integration.OpenTelemetry = BrokenOpenTelemetry  # type: ignore[attr-defined]
    _install_fake_imports(monkeypatch, integration=integration)

    with pytest.raises(LiteLLMOpenTelemetryDependencyError) as caught:
        create_opentelemetry_callback(LiteLLMOpenTelemetryConfig())

    assert "late optional import" not in str(caught.value)


def test_factory_rejects_wrong_config_and_incompatible_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="LiteLLMOpenTelemetryConfig"):
        create_opentelemetry_callback(cast(LiteLLMOpenTelemetryConfig, object()))

    _install_fake_imports(monkeypatch, integration=_integration_module(complete=False))
    with pytest.raises(LiteLLMOpenTelemetryCompatibilityError, match="LiteLLM version"):
        create_opentelemetry_callback(LiteLLMOpenTelemetryConfig())


def test_factory_reports_missing_attribute_extension_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail at construction when LiteLLM cannot host the semantic adapter."""

    class IncompatibleOpenTelemetry:
        def __init__(self, *, config: object) -> None:
            del config

    integration = _integration_module()
    integration.OpenTelemetry = IncompatibleOpenTelemetry  # type: ignore[attr-defined]
    _install_fake_imports(monkeypatch, integration=integration)

    with pytest.raises(
        LiteLLMOpenTelemetryCompatibilityError,
        match="set_attributes",
    ):
        create_opentelemetry_callback(LiteLLMOpenTelemetryConfig())


async def test_backend_passes_fresh_request_local_callbacks_on_success_and_failure() -> None:
    callback = object()
    completion = CaptureCompletion(text_response("ok"), RuntimeError("provider failed"))
    backend = LiteLLMBackend(callbacks=(callback,), completion=completion)
    request = ModelRequest((Message.user("hello"),))

    await backend.generate(request, model="model")
    with pytest.raises(ModelError):
        await backend.generate(request, model="model")

    assert backend.callbacks == (callback,)
    first_success = cast(list[object], completion.calls[0]["success_callback"])
    first_failure = cast(list[object], completion.calls[0]["failure_callback"])
    second_success = cast(list[object], completion.calls[1]["success_callback"])
    second_failure = cast(list[object], completion.calls[1]["failure_callback"])
    assert first_success == first_failure == second_success == second_failure == [callback]
    assert len({id(first_success), id(first_failure), id(second_success), id(second_failure)}) == 4


async def test_backend_selects_custom_logger_success_and_failure_hooks() -> None:
    callback = _HookCallback()
    completion = CaptureCompletion(text_response())
    backend = LiteLLMBackend(callbacks=(callback,), completion=completion)

    await backend.generate(ModelRequest((Message.user("hello"),)), model="model")

    success = cast(list[object], completion.calls[0]["success_callback"])
    failure = cast(list[object], completion.calls[0]["failure_callback"])
    assert success == [callback.async_log_success_event]
    assert failure == [callback.log_failure_event]


async def test_request_completion_covers_noncallable_and_sync_hooks() -> None:
    """Preserve LiteLLM identifiers while joining callable sync results."""

    identifier_completion = _RequestCallbackCompletion(("custom-logger",))
    assert identifier_completion.success_callbacks == ["custom-logger"]
    assert identifier_completion.failure_callbacks == ["custom-logger"]
    await identifier_completion.wait()

    callback = _SyncSuccessCallback()
    hook_completion = _RequestCallbackCompletion((callback,))
    success = cast(
        Callable[[object], Awaitable[object]],
        hook_completion.success_callbacks[0],
    )
    assert await success("complete") == "complete"
    await hook_completion.wait()
    failure = hook_completion.failure_callbacks[0]
    assert callable(failure)
    assert failure("failed") == "failed"


@pytest.mark.parametrize("raises", [False, True])
async def test_request_completion_joins_async_failure_hooks(raises: bool) -> None:
    """Await an async result returned through LiteLLM's sync failure path."""

    calls: list[str] = []

    async def callback(*_args: object, **_kwargs: object) -> None:
        calls.append("failure")
        if raises:
            raise RuntimeError("ignored callback failure")

    completion = _RequestCallbackCompletion((callback,))
    failure = completion.failure_callbacks[0]
    assert callable(failure)
    failure()
    await completion.wait()

    assert calls == ["failure"]


async def test_request_completion_signals_when_sync_failure_hook_raises() -> None:
    """Do not strand the provider result behind a failed callback barrier."""

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("sync callback failure")

    completion = _RequestCallbackCompletion((fail,))
    failure = completion.failure_callbacks[0]
    assert callable(failure)
    with pytest.raises(RuntimeError, match="sync callback failure"):
        failure()
    await completion.wait()


def test_terminal_event_loop_close_keeps_real_otlp_prompt_and_response() -> None:
    """Join LiteLLM's detached success worker before ``asyncio.run`` exits."""

    _OTLPHandler.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OTLPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(
            exporter="otlp_http",
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
            service_name="purrcept-terminal-test",
            environment="test",
            capture_message_content="SPAN_ONLY",
        )
    )
    backend = LiteLLMBackend(
        default_options={"mock_response": "terminal-response"},
        callbacks=(callback,),
    )
    streaming_backend = LiteLLMBackend(
        default_options={"mock_response": "stream-response"},
        callbacks=(callback,),
    )

    try:
        response = asyncio.run(
            backend.generate(
                ModelRequest((Message.user("terminal-prompt"),)),
                model="openai/offline-test",
            )
        )
        assert response.text == "terminal-response"
        streaming_response = asyncio.run(
            streaming_backend.generate(
                ModelRequest((Message.user("stream-prompt"),)),
                model="openai/offline-test",
                emit=lambda _event: None,
            )
        )
        assert streaming_response.text == "stream-response"
        assert callback.force_flush(5_000) is True
    finally:
        callback.shutdown()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    exported = ExportTraceServiceRequest()
    for payload in _OTLPHandler.payloads:
        exported.MergeFromString(payload)
    spans = [
        span
        for resource_spans in exported.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
        if span.name == "litellm_request"
    ]
    assert len(spans) == 2
    attributes = [
        {attribute.key: attribute.value.string_value for attribute in span.attributes}
        for span in spans
    ]
    combined_inputs = "\n".join(item["gen_ai.input.messages"] for item in attributes)
    combined_outputs = "\n".join(item["gen_ai.output.messages"] for item in attributes)
    assert "terminal-prompt" in combined_inputs
    assert "stream-prompt" in combined_inputs
    assert "terminal-response" in combined_outputs
    assert "stream-response" in combined_outputs
    assert {item["openinference.span.kind"] for item in attributes} == {"LLM"}
    assert {item["input.value"] for item in attributes} == {
        "stream-prompt",
        "terminal-prompt",
    }
    assert {item["output.value"] for item in attributes} == {
        "stream-response",
        "terminal-response",
    }


def test_real_otel_callback_does_not_mask_a_local_stream_sink_failure() -> None:
    """A partial LiteLLM stream cannot strand its sink error behind OTel."""

    callback = create_opentelemetry_callback(
        LiteLLMOpenTelemetryConfig(
            exporter="console",
            service_name="purrcept-local-failure-test",
            environment="test",
        )
    )
    backend = LiteLLMBackend(
        default_options={"mock_response": "stream-response"},
        callbacks=(callback,),
    )
    sink_error = RuntimeError("sink failed")

    def broken_sink(_event: object) -> NoReturn:
        raise sink_error

    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(
                asyncio.wait_for(
                    backend.generate(
                        ModelRequest((Message.user("stream-prompt"),)),
                        model="openai/offline-test",
                        emit=broken_sink,
                    ),
                    timeout=5,
                )
            )
        assert caught.value is sink_error
    finally:
        callback.shutdown()


@pytest.mark.parametrize("callbacks", [cast(tuple[object, ...], "otel"), (None,)])
def test_backend_rejects_invalid_callback_sequences(callbacks: tuple[object, ...]) -> None:
    with pytest.raises(TypeError, match="callbacks"):
        LiteLLMBackend(callbacks=callbacks)


@pytest.mark.parametrize(
    "reserved",
    ["callback", "callbacks", "success_callback", "failure_callback"],
)
async def test_request_cannot_override_runtime_callback_fields(reserved: str) -> None:
    backend = LiteLLMBackend(completion=CaptureCompletion(text_response()))
    request = ModelRequest(
        (Message.user("hello"),),
        provider_options={"purrcept_litellm": {reserved: "bad"}},
    )

    with pytest.raises(ModelRequestError, match="cannot override"):
        await backend.generate(request, model="model")
