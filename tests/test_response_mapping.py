from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from purrcept_core.models import (
    FinishReason,
    Message,
    ModelContinuation,
    ModelRequest,
    ModelRequestError,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)

from purrcept_litellm import LiteLLMBackend
from purrcept_litellm._response import convert_response

from .helpers import CaptureCompletion, text_response


async def test_non_streaming_response_maps_text_tools_usage_and_metadata() -> None:
    raw = {
        "id": "response-42",
        "model": "provider-model",
        "system_fingerprint": "fingerprint",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "reasoning_content": "private reasoning",
                    "content": [
                        {"type": "text", "text": "hello "},
                        "world",
                        {"type": "ignored", "text": "ignored"},
                    ],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"query":"purrcept"}',
                            },
                        },
                        {
                            "function": {
                                "name": "save",
                                "arguments": {"path": "result.txt"},
                            },
                        },
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "cache_creation_input_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
        "provider_specific_fields": {
            "region": "test",
            "purrcept_continuation": {
                "provider": "fixture-provider",
                "data": {"cursor": "next"},
            },
        },
    }
    backend = LiteLLMBackend(completion=CaptureCompletion(raw))

    response = await backend.generate(
        ModelRequest((Message.user("hello"),)),
        model="requested-model",
    )

    assert response.text == "hello world"
    assert response.message.content[0] == ReasoningBlock("private reasoning")
    assert response.model == "provider-model"
    assert response.response_id == "response-42"
    assert response.finish_reason is FinishReason.TOOL_CALL
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.cached_input_tokens == 3
    assert response.usage.cache_write_input_tokens == 2
    assert response.usage.reasoning_tokens == 5
    assert response.provider_metadata == {
        "region": "test",
        "system_fingerprint": "fingerprint",
    }
    assert response.continuation == ModelContinuation(
        "fixture-provider",
        {"cursor": "next"},
    )
    assert response.tool_calls == (
        ToolCallBlock("call-1", "lookup", arguments={"query": "purrcept"}),
        ToolCallBlock("litellm-tool-1", "save", arguments={"path": "result.txt"}),
    )


@pytest.mark.parametrize(
    ("raw_reason", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("function_call", FinishReason.TOOL_CALL),
        ("tool_call", FinishReason.TOOL_CALL),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("unknown", FinishReason.OTHER),
        (None, FinishReason.OTHER),
    ],
)
def test_finish_reason_mapping(raw_reason: object, expected: FinishReason) -> None:
    raw = text_response(finish_reason=cast(str, raw_reason))
    raw_choices = cast(list[dict[str, object]], raw["choices"])
    raw_choices[0]["finish_reason"] = raw_reason

    response = convert_response(raw, requested_model="fallback")

    assert response.finish_reason is expected


def test_response_uses_requested_model_and_empty_text_fallbacks() -> None:
    raw = text_response("")
    raw["model"] = None
    raw["id"] = ""
    choices = cast(list[dict[str, object]], raw["choices"])
    message = cast(dict[str, object], choices[0]["message"])
    message["content"] = None

    response = convert_response(raw, requested_model="fallback-model")

    assert response.model == "fallback-model"
    assert response.response_id is None
    assert response.message.content == (TextBlock(""),)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"choices": []}, "no choices"),
        ({"choices": [{"finish_reason": "stop"}]}, "no message"),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "call"}],
                        }
                    }
                ]
            },
            "has no function",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"function": {"arguments": "{}"}}],
                        }
                    }
                ]
            },
            "no function name",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"function": {"name": "tool", "arguments": "{"}}],
                        }
                    }
                ]
            },
            "not valid JSON",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"function": {"name": "tool", "arguments": "[]"}}],
                        }
                    }
                ]
            },
            "decode to an object",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"function": {"name": "tool", "arguments": 1}}],
                        }
                    }
                ]
            },
            "JSON object or string",
        ),
    ],
)
def test_invalid_provider_responses_are_rejected(
    raw: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ModelRequestError, match=match):
        convert_response(raw, requested_model="model")


def test_usage_accepts_direct_cache_and_reasoning_counts_and_sanitizes_invalid_values() -> None:
    raw = text_response(
        usage={
            "prompt_tokens": -1,
            "completion_tokens": True,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": -2,
            "reasoning_tokens": 6,
        }
    )

    response = convert_response(raw, requested_model="model")

    assert response.usage is not None
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.cached_input_tokens == 4
    assert response.usage.cache_write_input_tokens == 0
    assert response.usage.reasoning_tokens == 6


def test_empty_tool_arguments_and_hidden_provider_metadata_are_supported() -> None:
    raw = SimpleNamespace(
        id="response",
        model="provider-model",
        system_fingerprint=None,
        provider_specific_fields=None,
        _hidden_params={"custom_llm_provider": "openai"},
        usage=None,
        choices=[
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call",
                            "function": {"name": "lookup", "arguments": None},
                        }
                    ],
                },
            }
        ],
    )

    response = convert_response(raw, requested_model="model")

    assert dict(response.tool_calls[0].arguments) == {}
    assert response.provider_metadata == {"litellm_provider": "openai"}


@pytest.mark.parametrize(
    "continuation",
    [
        "bad",
        {"provider": "", "data": {}},
        {"provider": "provider", "data": "bad"},
        {"provider": "provider", "data": {1: "bad"}},
    ],
)
def test_invalid_provider_continuation_is_ignored(continuation: object) -> None:
    raw = text_response(provider_specific_fields={"purrcept_continuation": continuation})

    response = convert_response(raw, requested_model="model")

    assert response.continuation is None
    assert response.provider_metadata == {}
