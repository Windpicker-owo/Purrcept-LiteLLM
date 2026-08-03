"""Compile immutable Core requests into LiteLLM Chat Completions parameters.

This module owns the provider-neutral-to-LiteLLM payload mapping: ordered
System instructions, conversation messages, reminders, tools, model settings,
cache hints, media, and adapter-scoped provider options.  It performs no I/O
and never mutates the source request.  Reserved transport, message, streaming,
and callback fields cannot be overridden through open-ended provider options;
their ownership remains with :class:`purrcept_litellm.LiteLLMBackend`.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import cast

from purrcept_core.models import (
    CacheMode,
    ContentBlock,
    ImageBlock,
    ImageUrl,
    JsonValue,
    Message,
    MessageRole,
    ModelRequest,
    ModelRequestError,
    PromptStability,
    ReasoningBlock,
    ReminderPlacement,
    SystemInstruction,
    SystemReminder,
    TextBlock,
    ToolCallBlock,
    ToolChoice,
    ToolResultBlock,
    ToolSpec,
)

from ._values import thaw_json
from .config import PROVIDER_OPTIONS_KEY, LiteLLMConfig

_RESERVED_OPTIONS = frozenset(
    {
        "api_base",
        "api_key",
        "api_version",
        "callback",
        "callbacks",
        "failure_callback",
        "messages",
        "model",
        "stream",
        "stream_options",
        "success_callback",
        "tools",
    }
)


def compile_request(
    request: ModelRequest,
    *,
    model: str,
    config: LiteLLMConfig,
    stream: bool,
) -> dict[str, object]:
    """Build one isolated LiteLLM invocation without mutating the request."""

    messages: list[dict[str, object]] = []
    for instruction in request.instructions:
        messages.append(_system_message(instruction))

    # Placement is a semantic hand-off from Core. Instruction reminders extend
    # the stable control prefix and therefore must precede every world-authored
    # conversation message. AUTO retains the adapter's historical tail policy;
    # providers that need a different automatic strategy can introduce it
    # explicitly without changing the meaning of INSTRUCTIONS or TAIL.
    for reminder in request.reminders:
        if reminder.placement is ReminderPlacement.INSTRUCTIONS:
            messages.append(_reminder_message(reminder))
    conversation_start = len(messages)
    for message in request.messages:
        messages.extend(_message_payloads(message))
    conversation_end = len(messages)
    for reminder in request.reminders:
        if reminder.placement is not ReminderPlacement.INSTRUCTIONS:
            messages.append(_reminder_message(reminder))

    tools = [_tool_payload(tool) for tool in request.tools]
    _apply_prompt_cache(
        request,
        messages=messages,
        tools=tools,
        conversation_range=(conversation_start, conversation_end),
        config=config,
    )

    options = _base_options(config)
    options.update(_request_options(request))
    options.update(
        {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
    )
    if stream:
        options["stream_options"] = {"include_usage": True}
    if tools:
        options["tools"] = tools
        options["tool_choice"] = request.settings.tool_choice.value
    elif request.settings.tool_choice is ToolChoice.NONE:
        options["tool_choice"] = ToolChoice.NONE.value

    settings = request.settings
    if settings.temperature is not None:
        options["temperature"] = settings.temperature
    if settings.max_output_tokens is not None:
        options["max_tokens"] = settings.max_output_tokens
    if settings.stop_sequences:
        options["stop"] = list(settings.stop_sequences)
    if settings.parallel_tool_calls is not None and tools:
        options["parallel_tool_calls"] = settings.parallel_tool_calls
    return options


def _base_options(config: LiteLLMConfig) -> dict[str, object]:
    options = {
        key: thaw_json(value)
        for key, value in config.default_options.items()
        if key not in _RESERVED_OPTIONS
    }
    if config.api_key is not None:
        options["api_key"] = config.api_key
    if config.api_base is not None:
        options["api_base"] = config.api_base
    if config.api_version is not None:
        options["api_version"] = config.api_version
    if config.timeout is not None:
        options["timeout"] = config.timeout
    return options


def _request_options(request: ModelRequest) -> dict[str, object]:
    raw = request.provider_options.get(PROVIDER_OPTIONS_KEY)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ModelRequestError(
            f"provider_options[{PROVIDER_OPTIONS_KEY!r}] must be a JSON object.",
            provider="litellm",
        )
    options = {key: thaw_json(value) for key, value in cast(Mapping[str, JsonValue], raw).items()}
    reserved = sorted(_RESERVED_OPTIONS.intersection(options))
    if reserved:
        joined = ", ".join(reserved)
        raise ModelRequestError(
            f"provider_options[{PROVIDER_OPTIONS_KEY!r}] cannot override: {joined}.",
            provider="litellm",
        )
    return options


def _system_message(instruction: SystemInstruction) -> dict[str, object]:
    return {
        "role": "system",
        "content": _content_value(instruction.content, field_name="instruction"),
    }


def _reminder_message(reminder: SystemReminder) -> dict[str, object]:
    return {
        "role": "system",
        "content": _content_value(reminder.content, field_name="reminder"),
    }


def _message_payloads(message: Message) -> tuple[dict[str, object], ...]:
    if message.role is MessageRole.TOOL:
        return _tool_result_messages(message)
    return (_chat_message(message),)


def _chat_message(message: Message) -> dict[str, object]:
    content: list[ContentBlock] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, object]] = []
    for block in message.content:
        if isinstance(block, ReasoningBlock):
            if message.role is not MessageRole.ASSISTANT:
                raise ModelRequestError(
                    "ReasoningBlock is only valid in assistant messages.",
                    provider="litellm",
                )
            reasoning.append(block.text)
        elif isinstance(block, ToolCallBlock):
            if message.role is not MessageRole.ASSISTANT:
                raise ModelRequestError(
                    "ToolCallBlock is only valid in assistant messages.",
                    provider="litellm",
                )
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(
                            thaw_json(cast(JsonValue, block.arguments)),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
            )
        elif isinstance(block, ToolResultBlock):
            raise ModelRequestError(
                "ToolResultBlock is only valid in tool messages.",
                provider="litellm",
            )
        else:
            content.append(block)

    payload: dict[str, object] = {"role": message.role.value}
    if content:
        payload["content"] = _content_value(tuple(content), field_name="message")
    else:
        payload["content"] = None
    if message.name is not None:
        payload["name"] = message.name
    if reasoning:
        # DeepSeek and other thinking providers require their previous hidden
        # reasoning to be returned beside the visible assistant content. The
        # structured field preserves that protocol without exposing it as text.
        payload["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _tool_result_messages(message: Message) -> tuple[dict[str, object], ...]:
    results: list[dict[str, object]] = []
    for block in message.content:
        if not isinstance(block, ToolResultBlock):
            raise ModelRequestError(
                "tool messages may contain only ToolResultBlock values.",
                provider="litellm",
            )
        content = _content_value(block.content, field_name="tool result")
        if block.is_error:
            content = _prefix_error(content)
        payload: dict[str, object] = {
            "role": "tool",
            "tool_call_id": block.tool_call_id,
            "content": content,
        }
        if message.name is not None:
            payload["name"] = message.name
        results.append(payload)
    return tuple(results)


def _prefix_error(content: object) -> object:
    prefix = "[tool_error]\n"
    if isinstance(content, str):
        return f"{prefix}{content}"
    return [{"type": "text", "text": prefix}, *cast(list[object], content)]


def _content_value(
    blocks: Sequence[ContentBlock],
    *,
    field_name: str,
) -> object:
    if not blocks:
        return ""
    if all(isinstance(block, TextBlock) for block in blocks):
        return "".join(cast(TextBlock, block).text for block in blocks)

    content: list[dict[str, object]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            if block.alt_text is not None:
                content.append({"type": "text", "text": block.alt_text})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_url(block)},
                }
            )
        else:
            raise ModelRequestError(
                f"{field_name} contains an unsupported {type(block).__name__}.",
                provider="litellm",
            )
    return content


def _image_url(block: ImageBlock) -> str:
    source = block.source
    if isinstance(source, ImageUrl):
        return source.url
    encoded = base64.b64encode(source.data).decode("ascii")
    return f"data:{source.media_type};base64,{encoded}"


def _tool_payload(tool: ToolSpec) -> dict[str, object]:
    function: dict[str, object] = {
        "name": tool.name,
        "parameters": thaw_json(cast(JsonValue, tool.parameters)),
    }
    if tool.description is not None:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _apply_prompt_cache(
    request: ModelRequest,
    *,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]],
    conversation_range: tuple[int, int],
    config: LiteLLMConfig,
) -> None:
    """Place stable, tool, and growing-history cache breakpoints.

    ``PREFER`` and ``EXPLICIT`` are best-effort provider hints. A single System
    breakpoint caches only the static contract, which is especially wasteful
    for a long-lived Entity whose tool schema and transcript are also repeated.
    The three targets below stay within the common four-breakpoint provider
    limit and never mark the volatile tail reminder.
    """

    cache = request.cache
    if cache.mode in {CacheMode.AUTO, CacheMode.DISABLED}:
        return
    if cache.strict and not config.allow_strict_prompt_cache:
        raise ModelRequestError(
            "strict prompt caching requires LiteLLMConfig("
            "allow_strict_prompt_cache=True) and a compatible provider.",
            provider="litellm",
        )

    breakpoints = 0
    instruction_messages = messages[: len(request.instructions)]
    instruction_pairs = zip(request.instructions, instruction_messages, strict=True)
    for instruction, message in reversed(tuple(instruction_pairs)):
        if instruction.stability is PromptStability.VOLATILE:
            continue
        if _mark_message_cache_control(message):
            breakpoints += 1
            break
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        breakpoints += 1

    start, end = conversation_range
    for message in reversed(messages[start:end]):
        if _mark_message_cache_control(message):
            breakpoints += 1
            break

    if cache.strict and breakpoints == 0:
        raise ModelRequestError(
            "strict prompt caching needs an instruction, tool, or conversation message.",
            provider="litellm",
        )


def _mark_message_cache_control(message: dict[str, object]) -> bool:
    """Attach one ephemeral breakpoint to the final cacheable content block."""

    raw_content = message.get("content")
    if isinstance(raw_content, str):
        # Compatibility: LiteLLM understands message-level cache_control and
        # moves it into a provider content block when required. Keeping the
        # original string shape prevents unsupported automatic-cache providers
        # from seeing the same historical message alternate between string and
        # block-array encodings as the growing breakpoint moves forward.
        message["cache_control"] = {"type": "ephemeral"}
        return True
    if not isinstance(raw_content, list) or not raw_content:
        return False
    content = cast(list[dict[str, object]], raw_content)
    content[-1]["cache_control"] = {"type": "ephemeral"}
    return True


__all__: list[str] = []
