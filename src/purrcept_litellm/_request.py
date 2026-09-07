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
    stream: bool = True,
) -> dict[str, object]:
    """Build one isolated LiteLLM invocation without mutating the request."""

    messages: list[dict[str, object]] = []
    for instruction in request.instructions:
        messages.append(_system_message(instruction))

    # Placement is position relative to history, not wire role. Only
    # SystemInstruction uses ``system``; every reminder is a user message.
    # INSTRUCTIONS reminders still precede conversation so they sit next to
    # the trusted prefix, but they remain user-level and are not hoisted into
    # the provider system blob. TAIL and AUTO follow the transcript.
    reminder_indexes: set[int] = set()
    for reminder in request.reminders:
        if reminder.placement is ReminderPlacement.INSTRUCTIONS:
            messages.append(_reminder_message(reminder))
            reminder_indexes.add(len(messages) - 1)
    conversation_start = len(messages)
    messages.extend(_conversation_messages(request.messages))
    conversation_end = len(messages)
    for reminder in request.reminders:
        if reminder.placement is not ReminderPlacement.INSTRUCTIONS:
            messages.append(_reminder_message(reminder))
            reminder_indexes.add(len(messages) - 1)

    tools = [_tool_payload(tool) for tool in request.tools]
    _apply_prompt_cache(
        request,
        messages=messages,
        tools=tools,
        conversation_range=(conversation_start, conversation_end),
        reminder_indexes=reminder_indexes,
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
    # Reminders are request-local control text, not system instructions.
    # Emitting them as ``system`` lets Chat Completions gateways concatenate
    # them into the prompt prefix and truncate prefix cache on every change.
    return {
        "role": "user",
        "content": _content_value(reminder.content, field_name="reminder"),
    }


def _conversation_messages(history: Sequence[Message]) -> list[dict[str, object]]:
    """Keep tool responses adjacent, then expose their images in a user media message.

    Compatibility: Chat Completions tool content accepts text parts only. Core
    retains images inside ToolResultBlock; this transport projection moves only
    pixels into a labelled companion message without changing stored history.
    Images must wait until every adjacent tool result has been emitted, since
    a user message between responses to parallel tool calls breaks the protocol.
    """

    messages: list[dict[str, object]] = []
    images: list[ContentBlock] = []
    for message in history:
        if message.role is MessageRole.TOOL:
            results, attachments = _tool_result_messages(message)
            messages.extend(results)
            images.extend(attachments)
        else:
            if images:
                messages.append(
                    {"role": "user", "content": _content_value(images, field_name="tool images")}
                )
                images.clear()
            messages.append(_chat_message(message))
    if images:
        messages.append(
            {"role": "user", "content": _content_value(images, field_name="tool images")}
        )
    return messages


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


def _tool_result_messages(
    message: Message,
) -> tuple[tuple[dict[str, object], ...], tuple[ContentBlock, ...]]:
    """Separate tool text from visual attachments while preserving call identity and errors."""

    results: list[dict[str, object]] = []
    images: list[ContentBlock] = []
    for block in message.content:
        if not isinstance(block, ToolResultBlock):
            raise ModelRequestError(
                "tool messages may contain only ToolResultBlock values.",
                provider="litellm",
            )
        text_parts = tuple(part for part in block.content if not isinstance(part, ImageBlock))
        image_parts = tuple(part for part in block.content if isinstance(part, ImageBlock))
        content = _content_value(text_parts, field_name="tool result")
        if image_parts:
            status = " (tool error)" if block.is_error else ""
            images.append(
                TextBlock(f"Visual attachments from tool result {block.tool_call_id}{status}:")
            )
            images.extend(image_parts)
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
    return tuple(results), tuple(images)


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
    reminder_indexes: set[int],
    config: LiteLLMConfig,
) -> None:
    """Place stable, tool, and growing-history cache breakpoints.

    ``AUTO`` is this adapter's default and matches ``PREFER`` / ``EXPLICIT``:
    best-effort hints on the static contract, the last tool, and the growing
    transcript. A single System breakpoint caches only the static contract,
    which is especially wasteful for a long-lived Entity whose tool schema and
    transcript are also repeated. The three targets below stay within the
    common four-breakpoint provider limit.

    Reminders are request-local and often volatile (world view, clocks). Marking
    them would write a cache that almost never hits, so they are never given
    ``cache_control`` even when they sit next to the trusted prefix.
    """

    cache = request.cache
    if cache.mode is CacheMode.DISABLED:
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
    for index in range(end - 1, start - 1, -1):
        if index in reminder_indexes:
            continue
        if _mark_message_cache_control(messages[index]):
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
