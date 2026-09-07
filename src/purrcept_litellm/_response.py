"""Convert LiteLLM response objects and stream chunks to Purrcept values.

Visible assistant text, hidden model reasoning, tool calls, usage, metadata,
and optional continuation state are mapped independently. Reasoning fragments
are retained in the final assistant message so a later request can replay the
provider's complete turn. Readable reasoning also emits a distinct reasoning
channel; it never appears in assistant answer text-delta events.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from purrcept_core.models import (
    FinishReason,
    JsonValue,
    Message,
    MessageRole,
    ModelContinuation,
    ModelRequestError,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelStreamStarted,
    ReasoningBlock,
    ReasoningDelta,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageUpdate,
)

from ._values import (
    MISSING,
    integer,
    json_object,
    json_value,
    non_empty_string,
    read,
    sequence,
)

_CONTINUATION_FIELD = "purrcept_continuation"


def _empty_metadata() -> dict[str, JsonValue]:
    return {}


def _empty_text() -> list[str]:
    return []


def _empty_tools() -> dict[int, _StreamingTool]:
    return {}


def _empty_arguments() -> list[str]:
    return []


def convert_response(raw: object, *, requested_model: str) -> ModelResponse:
    """Convert one non-streaming LiteLLM response."""

    choice = _first_choice(raw)
    message = read(choice, "message", MISSING)
    if message is MISSING:
        raise _invalid_response("the first choice has no message")

    blocks: list[ReasoningBlock | TextBlock | ToolCallBlock] = []
    reasoning = _reasoning_text(message)
    if reasoning is not None:
        blocks.append(ReasoningBlock(reasoning))
    content = _response_text(read(message, "content", None))
    if content is not None:
        blocks.append(TextBlock(content))
    blocks.extend(_tool_calls(read(message, "tool_calls", None)))
    if not blocks:
        blocks.append(TextBlock(""))

    usage = _usage(read(raw, "usage", None))
    response_id = non_empty_string(read(raw, "id", None))
    response_model = non_empty_string(read(raw, "model", None)) or requested_model
    metadata, continuation = _provider_metadata(raw)
    return ModelResponse(
        Message(MessageRole.ASSISTANT, tuple(blocks)),
        usage=usage,
        finish_reason=_finish_reason(read(choice, "finish_reason", None)),
        model=response_model,
        response_id=response_id,
        continuation=continuation,
        provider_metadata=metadata,
    )


@dataclass(slots=True)
class StreamAccumulator:
    """Accumulate OpenAI-format LiteLLM chunks into one final response."""

    requested_model: str
    _finish: FinishReason = field(default=FinishReason.OTHER, init=False)
    _metadata: dict[str, JsonValue] = field(default_factory=_empty_metadata, init=False)
    _model: str | None = field(default=None, init=False)
    _response_id: str | None = field(default=None, init=False)
    _seen: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)
    _reasoning: list[str] = field(default_factory=_empty_text, init=False)
    _text: list[str] = field(default_factory=_empty_text, init=False)
    _tools: dict[int, _StreamingTool] = field(default_factory=_empty_tools, init=False)
    _usage_value: TokenUsage | None = field(default=None, init=False)
    _continuation: ModelContinuation | None = field(default=None, init=False)

    def ingest(self, chunk: object) -> tuple[ModelStreamEvent, ...]:
        """Consume one chunk and return ordered Purrcept events."""

        self._seen = True
        self._response_id = non_empty_string(read(chunk, "id", None)) or self._response_id
        self._model = non_empty_string(read(chunk, "model", None)) or self._model
        metadata, continuation = _provider_metadata(chunk)
        self._metadata.update(metadata)
        if continuation is not None:
            self._continuation = continuation

        events: list[ModelStreamEvent] = []
        if not self._started:
            self._started = True
            events.append(
                ModelStreamStarted(
                    model=self._model,
                    response_id=self._response_id,
                    provider_metadata=self._metadata,
                )
            )

        usage = _usage(read(chunk, "usage", None))
        if usage is not None:
            self._usage_value = usage
            events.append(UsageUpdate(usage))

        choices = sequence(read(chunk, "choices", ()))
        if not choices:
            return tuple(events)
        choice = choices[0]
        raw_finish = read(choice, "finish_reason", None)
        if raw_finish is not None:
            self._finish = _finish_reason(raw_finish)
        delta = read(choice, "delta", None)
        if delta is None:
            return tuple(events)

        reasoning = _reasoning_text(delta)
        if reasoning is not None:
            # Keep continuation state intact while exposing only the provider's
            # normalized readable text on a separate observation channel.
            self._reasoning.append(reasoning)
            events.append(ReasoningDelta(reasoning))

        content = read(delta, "content", None)
        if isinstance(content, str) and content:
            self._text.append(content)
            events.append(TextDelta(content))

        for position, raw_tool in enumerate(sequence(read(delta, "tool_calls", ()))):
            index = _tool_index(read(raw_tool, "index", position), fallback=position)
            tool = self._tools.setdefault(index, _StreamingTool(index))
            tool_id = non_empty_string(read(raw_tool, "id", None))
            function = read(raw_tool, "function", None)
            name = non_empty_string(read(function, "name", None)) if function is not None else None
            arguments = read(function, "arguments", "") if function is not None else ""
            arguments_delta = _arguments_fragment(arguments)
            if tool_id is not None:
                tool.tool_call_id = tool_id
            if name is not None:
                tool.name = name
            tool.arguments.append(arguments_delta)
            events.append(
                ToolCallDelta(
                    index,
                    arguments_delta,
                    tool_call_id=tool_id,
                    name=name,
                )
            )
        return tuple(events)

    def complete(self) -> ModelResponse:
        """Finalize a consumed stream."""

        if not self._seen:
            raise _invalid_response("the streaming response produced no chunks")
        blocks: list[ReasoningBlock | TextBlock | ToolCallBlock] = []
        reasoning = "".join(self._reasoning)
        if reasoning:
            blocks.append(ReasoningBlock(reasoning))
        text = "".join(self._text)
        if text:
            blocks.append(TextBlock(text))
        for index in sorted(self._tools):
            tool = self._tools[index]
            blocks.append(
                ToolCallBlock(
                    tool.tool_call_id or f"litellm-tool-{index}",
                    tool.require_name(),
                    arguments=_arguments_object("".join(tool.arguments)),
                )
            )
        if not blocks:
            blocks.append(TextBlock(""))
        return ModelResponse(
            Message(MessageRole.ASSISTANT, tuple(blocks)),
            usage=self._usage_value,
            finish_reason=self._finish,
            model=self._model or self.requested_model,
            response_id=self._response_id,
            continuation=self._continuation,
            provider_metadata=self._metadata,
        )


@dataclass(slots=True)
class _StreamingTool:
    index: int
    arguments: list[str] = field(default_factory=_empty_arguments)
    name: str | None = None
    tool_call_id: str | None = None

    def require_name(self) -> str:
        if self.name is None:
            raise _invalid_response(f"streamed tool call {self.index} has no function name")
        return self.name


def synthetic_stream_events(response: ModelResponse) -> tuple[ModelStreamEvent, ...]:
    """Create a valid stream lifecycle when a completion callable returns a final value."""

    events: list[ModelStreamEvent] = [
        ModelStreamStarted(
            model=response.model,
            response_id=response.response_id,
            provider_metadata=response.provider_metadata,
        )
    ]
    if response.text:
        events.append(TextDelta(response.text))
    for index, tool in enumerate(response.tool_calls):
        events.append(
            ToolCallDelta(
                index,
                json.dumps(
                    dict(tool.arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                tool_call_id=tool.id,
                name=tool.name,
            )
        )
    if response.usage is not None:
        events.append(UsageUpdate(response.usage))
    events.append(ModelStreamCompleted(response))
    return tuple(events)


def _first_choice(raw: object) -> object:
    choices = sequence(read(raw, "choices", ()))
    if not choices:
        raise _invalid_response("the response has no choices")
    return choices[0]


def _response_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    text: list[str] = []
    for item in sequence(value):
        if isinstance(item, str):
            text.append(item)
            continue
        item_type = read(item, "type", None)
        item_text = read(item, "text", None)
        if item_type in {"text", "output_text"} and isinstance(item_text, str):
            text.append(item_text)
    return "".join(text)


def _reasoning_text(message: object) -> str | None:
    """Read LiteLLM's normalized hidden-reasoning field from one message value."""

    value = read(message, "reasoning_content", None)
    if not isinstance(value, str):
        provider_fields = read(message, "provider_specific_fields", cast(object, {}))
        value = read(provider_fields, "reasoning_content", None)
    return value if isinstance(value, str) and value else None


def _tool_calls(value: object) -> tuple[ToolCallBlock, ...]:
    calls: list[ToolCallBlock] = []
    for index, raw_tool in enumerate(sequence(value)):
        function = read(raw_tool, "function", None)
        if function is None:
            raise _invalid_response(f"tool call {index} has no function")
        name = non_empty_string(read(function, "name", None))
        if name is None:
            raise _invalid_response(f"tool call {index} has no function name")
        tool_call_id = non_empty_string(read(raw_tool, "id", None)) or f"litellm-tool-{index}"
        calls.append(
            ToolCallBlock(
                tool_call_id,
                name,
                arguments=_arguments_object(read(function, "arguments", "{}")),
            )
        )
    return tuple(calls)


def _arguments_object(value: object) -> Mapping[str, JsonValue]:
    if isinstance(value, Mapping):
        projected = json_object(cast(object, value))
        return projected
    if value is None or value == "":
        return {}
    if not isinstance(value, str):
        raise _invalid_response("tool arguments must be a JSON object or string")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise _invalid_response("tool arguments are not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise _invalid_response("tool arguments JSON must decode to an object")
    return json_object(cast(object, parsed))


def _arguments_fragment(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(
            json_object(cast(object, value)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise _invalid_response("streamed tool arguments must be a string or object")


def _tool_index(value: object, *, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return fallback
    return value


def _finish_reason(value: object) -> FinishReason:
    if value == "stop":
        return FinishReason.STOP
    if value == "length":
        return FinishReason.LENGTH
    if value in {"tool_calls", "function_call", "tool_call"}:
        return FinishReason.TOOL_CALL
    if value == "content_filter":
        return FinishReason.CONTENT_FILTER
    return FinishReason.OTHER


def _usage(value: object) -> TokenUsage | None:
    if value is None:
        return None
    prompt_details = read(value, "prompt_tokens_details", None)
    completion_details = read(value, "completion_tokens_details", None)
    cached = integer(read(value, "cache_read_input_tokens", MISSING))
    if cached == 0 and prompt_details is not None:
        cached = integer(read(prompt_details, "cached_tokens", 0))
    cache_write = integer(read(value, "cache_creation_input_tokens", 0))
    reasoning = integer(read(value, "reasoning_tokens", MISSING))
    if reasoning == 0 and completion_details is not None:
        reasoning = integer(read(completion_details, "reasoning_tokens", 0))
    return TokenUsage(
        integer(read(value, "prompt_tokens", 0)),
        integer(read(value, "completion_tokens", 0)),
        cached_input_tokens=cached,
        cache_write_input_tokens=cache_write,
        reasoning_tokens=reasoning,
    )


def _provider_metadata(
    raw: object,
) -> tuple[dict[str, JsonValue], ModelContinuation | None]:
    raw_fields = read(raw, "provider_specific_fields", cast(object, {}))
    continuation = _continuation(read(raw_fields, _CONTINUATION_FIELD, None))
    fields = json_object(raw_fields)
    fields.pop(_CONTINUATION_FIELD, None)
    fingerprint = non_empty_string(read(raw, "system_fingerprint", None))
    if fingerprint is not None:
        fields["system_fingerprint"] = fingerprint
    hidden: object = read(raw, "_hidden_params", cast(object, {}))
    provider = non_empty_string(read(hidden, "custom_llm_provider", None))
    if provider is not None:
        fields["litellm_provider"] = provider
    return fields, continuation


def _continuation(value: object) -> ModelContinuation | None:
    if not isinstance(value, Mapping):
        return None
    mapping_value = cast(object, value)
    provider = non_empty_string(read(mapping_value, "provider", None))
    data = read(mapping_value, "data", None)
    if provider is None or not isinstance(data, Mapping):
        return None
    if not all(isinstance(key, str) for key in cast(Mapping[object, object], data)):
        return None
    projected = json_value(cast(object, data))
    assert isinstance(projected, Mapping)
    return ModelContinuation(provider, cast(Mapping[str, JsonValue], projected))


def _invalid_response(detail: str) -> ModelRequestError:
    return ModelRequestError(
        f"LiteLLM returned an invalid response: {detail}.",
        provider="litellm",
    )


__all__: list[str] = []
