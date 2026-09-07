from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import cast

import pytest
from purrcept_core.models import (
    CacheMode,
    ContentBlock,
    ImageBlock,
    ImageBytes,
    ImageUrl,
    Message,
    MessageRole,
    ModelContinuation,
    ModelRequest,
    ModelRequestError,
    ModelSettings,
    PromptCachePolicy,
    PromptStability,
    ReasoningBlock,
    ReminderPlacement,
    ReminderScope,
    SystemInstruction,
    SystemReminder,
    TextBlock,
    ToolCallBlock,
    ToolChoice,
    ToolResultBlock,
    ToolSpec,
)

from purrcept_litellm import LiteLLMBackend

from .helpers import CaptureCompletion, text_response


@dataclass(frozen=True, slots=True)
class UnsupportedBlock(ContentBlock):
    value: str


async def test_complete_request_is_compiled_to_litellm_chat_format() -> None:
    completion = CaptureCompletion(text_response("done"))
    backend = LiteLLMBackend(
        api_key="secret",
        api_base="https://proxy.invalid/v1",
        api_version="v1",
        timeout=12,
        default_options={
            "top_p": 0.8,
            "model": "must-not-win",
            "messages": ["must-not-win"],
        },
        completion=completion,
    )
    request = ModelRequest(
        (
            Message(
                MessageRole.USER,
                (
                    TextBlock("inspect "),
                    ImageBlock(ImageUrl("https://example.invalid/image.png"), alt_text="diagram"),
                    ImageBlock(ImageBytes(b"png", "image/png")),
                ),
                name="operator",
                metadata={"not": "sent"},
            ),
            Message(
                MessageRole.ASSISTANT,
                (
                    ReasoningBlock("hidden thought"),
                    TextBlock("calling"),
                    ToolCallBlock("call-1", "lookup", arguments={"query": "purrcept"}),
                ),
            ),
            Message(
                MessageRole.TOOL,
                (
                    ToolResultBlock(
                        "call-1",
                        content=(TextBlock("not found"),),
                        is_error=True,
                    ),
                ),
                name="lookup",
            ),
        ),
        instructions=(SystemInstruction.from_text("Be exact."),),
        reminders=(
            SystemReminder(
                (TextBlock("First reminder."),),
                key="first",
                scope=ReminderScope.TURN,
                placement=ReminderPlacement.TAIL,
                priority=10,
            ),
            SystemReminder(
                (TextBlock("Second reminder."),),
                key="second",
                scope=ReminderScope.NEXT_REQUEST,
                placement=ReminderPlacement.INSTRUCTIONS,
                priority=5,
            ),
        ),
        tools=(
            ToolSpec(
                "lookup",
                description="Look up a value.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        ),
        settings=ModelSettings(
            temperature=0.2,
            max_output_tokens=256,
            stop_sequences=("END",),
            tool_choice=ToolChoice.REQUIRED,
            parallel_tool_calls=True,
        ),
        cache=PromptCachePolicy(mode=CacheMode.DISABLED),
        continuation=ModelContinuation("ignored-provider", {"cursor": "ignored"}),
        metadata={"not": "sent"},
        provider_options={
            "unrelated": {"ignored": True},
            "purrcept_litellm": {
                "reasoning_effort": "medium",
                "top_p": 0.7,
            },
        },
    )

    response = await backend.generate(request, model="openai/test-model")

    assert response.text == "done"
    assert len(completion.calls) == 1
    call = completion.calls[0]
    assert call["model"] == "openai/test-model"
    assert call["api_key"] == "secret"
    assert call["api_base"] == "https://proxy.invalid/v1"
    assert call["api_version"] == "v1"
    assert call["timeout"] == 12
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}
    assert call["top_p"] == 0.7
    assert call["reasoning_effort"] == "medium"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 256
    assert call["stop"] == ["END"]
    assert call["tool_choice"] == "required"
    assert call["parallel_tool_calls"] is True
    assert "metadata" not in call
    assert "continuation" not in call

    messages = cast(list[dict[str, object]], call["messages"])
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert messages[0]["content"] == "Be exact."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Second reminder."
    user_content = cast(list[dict[str, object]], messages[2]["content"])
    assert user_content == [
        {"type": "text", "text": "inspect "},
        {"type": "text", "text": "diagram"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/image.png"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5n"},
        },
    ]
    assert messages[2]["name"] == "operator"
    assert messages[3]["content"] == "calling"
    assert messages[3]["reasoning_content"] == "hidden thought"
    tool_calls = cast(list[dict[str, object]], messages[3]["tool_calls"])
    assert tool_calls[0]["id"] == "call-1"
    assert tool_calls[0]["function"] == {
        "name": "lookup",
        "arguments": '{"query":"purrcept"}',
    }
    assert messages[4] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "[tool_error]\nnot found",
        "name": "lookup",
    }
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "First reminder."
    assert [message["content"] for message in messages if message["role"] == "system"] == [
        "Be exact.",
    ]
    tools = cast(list[dict[str, object]], call["tools"])
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]


async def test_tool_choice_none_is_sent_without_tools() -> None:
    completion = CaptureCompletion(text_response())
    backend = LiteLLMBackend(completion=completion)
    request = ModelRequest(
        (Message.user("hello"),),
        settings=ModelSettings(tool_choice=ToolChoice.NONE),
    )

    await backend.generate(request, model="model")

    assert completion.calls[0]["tool_choice"] == "none"
    assert "tools" not in completion.calls[0]
    assert "parallel_tool_calls" not in completion.calls[0]


@pytest.mark.parametrize("reserved", ["model", "messages", "stream", "tools"])
async def test_request_provider_options_cannot_override_reserved_fields(
    reserved: str,
) -> None:
    backend = LiteLLMBackend(completion=CaptureCompletion(text_response()))
    request = ModelRequest(
        (Message.user("hello"),),
        provider_options={"purrcept_litellm": {reserved: "bad"}},
    )

    with pytest.raises(ModelRequestError, match="cannot override"):
        await backend.generate(request, model="model")


async def test_request_provider_options_namespace_must_be_an_object() -> None:
    backend = LiteLLMBackend(completion=CaptureCompletion(text_response()))
    request = ModelRequest(
        (Message.user("hello"),),
        provider_options={"purrcept_litellm": ["bad"]},
    )

    with pytest.raises(ModelRequestError, match="must be a JSON object"):
        await backend.generate(request, model="model")


@pytest.mark.parametrize(
    "message",
    [
        Message(
            MessageRole.USER,
            (ToolCallBlock("call", "tool"),),
        ),
        Message(
            MessageRole.ASSISTANT,
            (ToolResultBlock("call"),),
        ),
        Message(
            MessageRole.USER,
            (ReasoningBlock("not an assistant turn"),),
        ),
        Message(
            MessageRole.TOOL,
            (TextBlock("bad"),),
        ),
    ],
)
async def test_invalid_role_block_combinations_are_rejected(message: Message) -> None:
    backend = LiteLLMBackend(completion=CaptureCompletion(text_response()))

    with pytest.raises(ModelRequestError):
        await backend.generate(ModelRequest((message,)), model="model")


@pytest.mark.parametrize("field", ["instruction", "reminder"])
async def test_unsupported_system_content_is_rejected(field: str) -> None:
    backend = LiteLLMBackend(completion=CaptureCompletion(text_response()))
    block = UnsupportedBlock("bad")
    kwargs: dict[str, object]
    if field == "instruction":
        kwargs = {"instructions": (SystemInstruction((block,)),)}
    else:
        kwargs = {"reminders": (SystemReminder((block,)),)}

    with pytest.raises(ModelRequestError, match="UnsupportedBlock"):
        await backend.generate(
            ModelRequest((Message.user("hello"),), **kwargs),  # type: ignore[arg-type]
            model="model",
        )


async def test_empty_tool_result_and_multimodal_error_are_supported() -> None:
    completion = CaptureCompletion(text_response())
    backend = LiteLLMBackend(completion=completion)
    request = ModelRequest(
        (
            Message(
                MessageRole.TOOL,
                (
                    ToolResultBlock("empty"),
                    ToolResultBlock(
                        "image",
                        content=(ImageBlock(ImageUrl("https://example.invalid/a.png")),),
                        is_error=True,
                    ),
                ),
            ),
        ),
        cache=PromptCachePolicy(mode=CacheMode.DISABLED),
    )

    await backend.generate(request, model="model")

    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    assert messages[0]["content"] == ""
    assert messages[1]["content"] == "[tool_error]\n"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == [
        {"type": "text", "text": "Visual attachments from tool result image (tool error):"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/a.png"},
        },
    ]


async def test_tool_images_follow_all_parallel_results_without_mutating_history() -> None:
    """A media companion cannot interrupt the tool-response group or overtake tail reminders."""

    completion = CaptureCompletion(text_response())
    backend = LiteLLMBackend(completion=completion)
    image = ImageBlock(ImageBytes(b"pixels", "image/png"), alt_text="tool screenshot")
    history = (
        Message(
            MessageRole.ASSISTANT,
            (
                ToolCallBlock("first", "capture", arguments={}),
                ToolCallBlock("second", "read", arguments={}),
            ),
        ),
        Message(
            MessageRole.TOOL, (ToolResultBlock("first", content=(TextBlock("captured"), image)),)
        ),
        Message(MessageRole.TOOL, (ToolResultBlock("second", content=(TextBlock("read done"),)),)),
        Message.user("continue"),
    )
    request = ModelRequest(
        history,
        reminders=(
            SystemReminder((TextBlock("current world"),), placement=ReminderPlacement.TAIL),
        ),
        cache=PromptCachePolicy(mode=CacheMode.DISABLED),
    )
    await backend.generate(request, model="model")
    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    assert [item["role"] for item in messages] == [
        "assistant",
        "tool",
        "tool",
        "user",
        "user",
        "user",
    ]
    assert messages[1]["tool_call_id"] == "first" and messages[1]["content"] == "captured"
    assert messages[2]["tool_call_id"] == "second" and messages[2]["content"] == "read done"
    assert messages[3]["content"] == [
        {"type": "text", "text": "Visual attachments from tool result first:"},
        {"type": "text", "text": "tool screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGl4ZWxz"}},
    ]
    assert messages[4]["content"] == "continue"
    assert messages[5]["content"] == "current world"
    assert request.messages == history
    tool_result = history[1].content[0]
    assert isinstance(tool_result, ToolResultBlock) and tool_result.content[1] is image


async def test_prompt_cache_marks_instruction_or_tool_best_effort() -> None:
    instruction_completion = CaptureCompletion(text_response())
    instruction_backend = LiteLLMBackend(completion=instruction_completion)
    instruction_request = ModelRequest(
        (Message.user("hello"),),
        instructions=(SystemInstruction.from_text("stable"),),
        cache=PromptCachePolicy(mode=CacheMode.PREFER),
    )
    await instruction_backend.generate(instruction_request, model="model")
    instruction_messages = cast(
        list[dict[str, object]],
        instruction_completion.calls[0]["messages"],
    )
    assert instruction_messages[0]["content"] == "stable"
    assert instruction_messages[0]["cache_control"] == {"type": "ephemeral"}

    tool_completion = CaptureCompletion(text_response())
    tool_backend = LiteLLMBackend(completion=tool_completion)
    tool_request = ModelRequest(
        (Message.user("hello"),),
        tools=(ToolSpec("lookup"),),
        cache=PromptCachePolicy(mode=CacheMode.EXPLICIT),
    )
    await tool_backend.generate(tool_request, model="model")
    tools = cast(list[dict[str, object]], tool_completion.calls[0]["tools"])
    assert tools[0]["cache_control"] == {"type": "ephemeral"}


async def test_multimodal_instruction_cache_marks_final_content_item() -> None:
    completion = CaptureCompletion(text_response())
    backend = LiteLLMBackend(completion=completion)
    request = ModelRequest(
        (Message.user("hello"),),
        instructions=(
            SystemInstruction(
                (
                    TextBlock("stable"),
                    ImageBlock(ImageUrl("https://example.invalid/stable.png")),
                )
            ),
        ),
        cache=PromptCachePolicy(mode=CacheMode.PREFER),
    )

    await backend.generate(request, model="model")

    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    content = cast(list[dict[str, object]], messages[0]["content"])
    assert content[-1]["cache_control"] == {"type": "ephemeral"}


async def test_strict_prompt_cache_requires_explicit_backend_capability() -> None:
    request = ModelRequest(
        (Message.user("hello"),),
        instructions=(SystemInstruction.from_text("stable"),),
        cache=PromptCachePolicy(
            mode=CacheMode.EXPLICIT,
            key="cache-key",
            ttl=timedelta(minutes=5),
            strict=True,
        ),
    )
    backend = LiteLLMBackend(completion=CaptureCompletion(text_response()))

    with pytest.raises(ModelRequestError, match="allow_strict_prompt_cache"):
        await backend.generate(request, model="model")


async def test_strict_prompt_cache_requires_a_stable_target() -> None:
    request = ModelRequest(
        (),
        instructions=(
            SystemInstruction.from_text(
                "volatile",
                stability=PromptStability.VOLATILE,
            ),
        ),
        cache=PromptCachePolicy(mode=CacheMode.EXPLICIT, strict=True),
    )
    backend = LiteLLMBackend(
        allow_strict_prompt_cache=True,
        completion=CaptureCompletion(text_response()),
    )

    with pytest.raises(ModelRequestError, match="instruction, tool, or conversation"):
        await backend.generate(request, model="model")


async def test_preferred_cache_marks_growing_conversation_boundary() -> None:
    completion = CaptureCompletion(text_response())
    request = ModelRequest(
        (Message.user("hello"),),
        cache=PromptCachePolicy(mode=CacheMode.PREFER),
    )

    await LiteLLMBackend(completion=completion).generate(request, model="model")

    assert completion.calls[0]["messages"] == [
        {
            "role": "user",
            "content": "hello",
            "cache_control": {"type": "ephemeral"},
        }
    ]


async def test_preferred_cache_keeps_volatile_tail_after_growing_breakpoint() -> None:
    """Cache repeated transcript content without marking the world-view tail."""

    completion = CaptureCompletion(text_response())
    request = ModelRequest(
        (
            Message.user("old user"),
            Message.assistant("old assistant"),
            Message.user("current observation"),
        ),
        instructions=(SystemInstruction.from_text("stable system"),),
        reminders=(
            SystemReminder(
                (TextBlock("volatile world view"),),
                placement=ReminderPlacement.TAIL,
            ),
        ),
        tools=(ToolSpec("play_action"),),
        cache=PromptCachePolicy(mode=CacheMode.PREFER),
    )

    await LiteLLMBackend(completion=completion).generate(request, model="model")

    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    assert messages[0]["role"] == "system"
    assert messages[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[1]
    assert "cache_control" not in messages[2]
    assert messages[3]["cache_control"] == {"type": "ephemeral"}
    assert messages[4]["role"] == "user"
    assert messages[4]["content"] == "volatile world view"
    assert "cache_control" not in messages[4]
    tools = cast(list[dict[str, object]], completion.calls[0]["tools"])
    assert tools[0]["cache_control"] == {"type": "ephemeral"}


async def test_preferred_cache_does_not_mark_instruction_or_tail_reminders() -> None:
    """Explicit cache breakpoints skip request-local reminders."""

    completion = CaptureCompletion(text_response())
    request = ModelRequest(
        (Message.user("hello"),),
        instructions=(SystemInstruction.from_text("stable contract"),),
        reminders=(
            SystemReminder(
                (TextBlock("prefix reminder"),),
                key="prefix",
                placement=ReminderPlacement.INSTRUCTIONS,
            ),
            SystemReminder(
                (TextBlock("tail reminder"),),
                key="tail",
                placement=ReminderPlacement.TAIL,
            ),
        ),
        cache=PromptCachePolicy(mode=CacheMode.PREFER),
    )

    await LiteLLMBackend(completion=completion).generate(request, model="model")

    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    assert [message["role"] for message in messages] == ["system", "user", "user", "user"]
    assert messages[0]["content"] == "stable contract"
    assert messages[0]["cache_control"] == {"type": "ephemeral"}
    assert messages[1]["content"] == "prefix reminder"
    assert "cache_control" not in messages[1]
    assert messages[2]["content"] == "hello"
    assert messages[2]["cache_control"] == {"type": "ephemeral"}
    assert messages[3]["content"] == "tail reminder"
    assert "cache_control" not in messages[3]


async def test_instruction_and_tail_reminders_are_user_messages() -> None:
    """Reminders stay user-level; only SystemInstruction uses system."""

    completion = CaptureCompletion(text_response())
    request = ModelRequest(
        (Message.user("hello"),),
        instructions=(SystemInstruction.from_text("stable contract"),),
        reminders=(
            SystemReminder(
                (TextBlock("prefix reminder"),),
                key="prefix",
                placement=ReminderPlacement.INSTRUCTIONS,
            ),
            SystemReminder(
                (TextBlock("tail reminder"),),
                key="tail",
                placement=ReminderPlacement.TAIL,
            ),
        ),
    )

    await LiteLLMBackend(completion=completion).generate(request, model="model")

    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    assert [message["role"] for message in messages] == ["system", "user", "user", "user"]
    assert messages[0]["content"] == "stable contract"
    assert messages[1]["content"] == "prefix reminder"
    assert messages[2]["content"] == "hello"
    assert messages[3]["content"] == "tail reminder"


async def test_auto_cache_matches_prefer_breakpoints() -> None:
    """AUTO is this adapter's default and uses the same explicit breakpoints as PREFER."""

    completion = CaptureCompletion(text_response())
    request = ModelRequest(
        (Message.user("hello"),),
        instructions=(SystemInstruction.from_text("stable"),),
        reminders=(
            SystemReminder(
                (TextBlock("volatile tail"),),
                placement=ReminderPlacement.TAIL,
            ),
        ),
        cache=PromptCachePolicy(mode=CacheMode.AUTO),
    )

    await LiteLLMBackend(completion=completion).generate(request, model="model")

    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    assert messages[0]["content"] == "stable"
    assert messages[0]["cache_control"] == {"type": "ephemeral"}
    assert messages[1]["content"] == "hello"
    assert messages[1]["cache_control"] == {"type": "ephemeral"}
    assert messages[2]["content"] == "volatile tail"
    assert "cache_control" not in messages[2]


async def test_disabled_cache_does_not_modify_messages() -> None:
    completion = CaptureCompletion(text_response())
    request = ModelRequest(
        (Message.user("hello"),),
        instructions=(SystemInstruction.from_text("stable"),),
        cache=PromptCachePolicy(mode=CacheMode.DISABLED),
    )

    await LiteLLMBackend(completion=completion).generate(request, model="model")

    messages = cast(list[dict[str, object]], completion.calls[0]["messages"])
    assert messages[0]["content"] == "stable"
    assert "cache_control" not in messages[0]
    assert "cache_control" not in messages[1]
