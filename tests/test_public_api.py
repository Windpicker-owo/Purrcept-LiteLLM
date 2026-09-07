from __future__ import annotations

from purrcept_core.models import ModelBackend

import purrcept_litellm


def test_public_api_and_version() -> None:
    backend = purrcept_litellm.LiteLLMBackend()

    assert purrcept_litellm.__version__ == "0.1.1"
    assert isinstance(backend, ModelBackend)
    assert purrcept_litellm.PROVIDER_OPTIONS_KEY == "purrcept_litellm"
    assert purrcept_litellm.__all__ == [
        "PROVIDER_OPTIONS_KEY",
        "CompletionCallable",
        "LiteLLMBackend",
        "LiteLLMConfig",
        "LiteLLMOpenTelemetryCallback",
        "LiteLLMOpenTelemetryCompatibilityError",
        "LiteLLMOpenTelemetryConfig",
        "LiteLLMOpenTelemetryDependencyError",
        "LiteLLMOpenTelemetryError",
        "OpenTelemetryCaptureMode",
        "OpenTelemetryExporter",
        "__version__",
        "create_opentelemetry_callback",
        "map_litellm_error",
    ]
