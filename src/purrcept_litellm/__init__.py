"""LiteLLM provider integration for Purrcept Core."""

from .backend import CompletionCallable, LiteLLMBackend
from .config import PROVIDER_OPTIONS_KEY, LiteLLMConfig
from .errors import map_litellm_error
from .observability import (
    LiteLLMOpenTelemetryCallback,
    LiteLLMOpenTelemetryCompatibilityError,
    LiteLLMOpenTelemetryConfig,
    LiteLLMOpenTelemetryDependencyError,
    LiteLLMOpenTelemetryError,
    OpenTelemetryCaptureMode,
    OpenTelemetryExporter,
    create_opentelemetry_callback,
)

__version__ = "0.1.3"

__all__ = [
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
