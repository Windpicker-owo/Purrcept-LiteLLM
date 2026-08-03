from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import litellm
import pytest
from litellm.exceptions import RateLimitError
from purrcept_core.models import (
    Message,
    ModelRateLimitError,
    ModelRequest,
    ModelRequestError,
    ModelStreamCompleted,
    ModelStreamStarted,
    ReasoningBlock,
    TextDelta,
    ToolCallDelta,
    UsageUpdate,
)

from purrcept_litellm import LiteLLMBackend
from purrcept_litellm.backend import (
    _close_stream_after_local_exit,  # pyright: ignore[reportPrivateUsage]
)

from .helpers import CaptureCompletion, ChunkStream


class _ClosableChunkStream(AsyncIterator[object]):
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.closed = False
        self._close_error = close_error
        self._emitted = False

    def __aiter__(self) -> _ClosableChunkStream:
        return self

    async def __anext__(self) -> object:
        if self._emitted:
            raise StopAsyncIteration
        self._emitted = True
        return {"choices": [{"delta": {"content": "x"}}]}

    async def aclose(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _BlockingClosableStream(AsyncIterator[object]):
    def __init__(self) -> None:
        self.closed = False
        self.started = asyncio.Event()
        self._blocker = asyncio.Event()

    def __aiter__(self) -> _BlockingClosableStream:
        return self

    async def __anext__(self) -> object:
        self.started.set()
        await self._blocker.wait()
        raise AssertionError("the cancellation test unexpectedly released its blocker")

    async def aclose(self) -> None:
        self.closed = True


class _BrokenClosableStream(AsyncIterator[object]):
    def __init__(self, failure: BaseException) -> None:
        self.closed = False
        self._failure = failure

    def __aiter__(self) -> _BrokenClosableStream:
        raise self._failure

    async def __anext__(self) -> object:
        raise AssertionError("a broken iterator must never be consumed")

    async def aclose(self) -> None:
        self.closed = True


def _unused_callback(*_args: object, **_kwargs: object) -> None:
    """Stand in for a terminal hook that a deliberately incomplete fake omits."""


async def test_async_stream_maps_text_tool_fragments_usage_and_completion() -> None:
    chunks = (
        {
            "id": "stream-1",
            "model": "provider-model",
            "system_fingerprint": "fp",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "private ",
                        "content": "Hello ",
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "stream-1",
            "model": "provider-model",
            "provider_specific_fields": {"region": "test"},
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "reasoning",
                        "content": "world",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":',
                                },
                            },
                            {
                                "index": cast(int, "bad"),
                                "id": "call-2",
                                "function": {
                                    "name": "save",
                                    "arguments": {"path": "result.txt"},
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "stream-1",
            "model": "provider-model",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"Boston"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        },
    )
    completion = CaptureCompletion(ChunkStream(chunks))
    backend = LiteLLMBackend(completion=completion)
    events: list[object] = []

    response = await backend.generate(
        ModelRequest((Message.user("hello"),)),
        model="requested-model",
        emit=events.append,
    )

    assert completion.calls[0]["stream"] is True
    assert completion.calls[0]["stream_options"] == {"include_usage": True}
    assert response.text == "Hello world"
    assert response.message.content[0] == ReasoningBlock("private reasoning")
    assert response.model == "provider-model"
    assert response.response_id == "stream-1"
    assert response.provider_metadata == {
        "system_fingerprint": "fp",
        "region": "test",
    }
    assert [(tool.id, tool.name, dict(tool.arguments)) for tool in response.tool_calls] == [
        ("call-1", "weather", {"city": "Boston"}),
        ("call-2", "save", {"path": "result.txt"}),
    ]
    assert response.usage is not None
    assert response.usage.cached_input_tokens == 2
    assert response.usage.reasoning_tokens == 1
    assert isinstance(events[0], ModelStreamStarted)
    assert [event.delta for event in events if isinstance(event, TextDelta)] == [
        "Hello ",
        "world",
    ]
    tool_events = [event for event in events if isinstance(event, ToolCallDelta)]
    assert [event.arguments_delta for event in tool_events] == [
        '{"city":',
        '{"path":"result.txt"}',
        '"Boston"}',
    ]
    assert len([event for event in events if isinstance(event, UsageUpdate)]) == 1
    assert isinstance(events[-1], ModelStreamCompleted)
    assert events[-1].response is response


async def test_usage_only_stream_chunk_still_starts_and_completes() -> None:
    completion = CaptureCompletion(
        ChunkStream(
            (
                {
                    "id": "usage-only",
                    "model": "model",
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0},
                },
            )
        )
    )
    events: list[object] = []

    response = await LiteLLMBackend(completion=completion).generate(
        ModelRequest((Message.user("hello"),)),
        model="model",
        emit=events.append,
    )

    assert response.text == ""
    assert [type(event) for event in events] == [
        ModelStreamStarted,
        UsageUpdate,
        ModelStreamCompleted,
    ]


async def test_stream_choice_without_delta_and_none_arguments_are_supported() -> None:
    completion = CaptureCompletion(
        ChunkStream(
            (
                {"choices": [{"delta": None, "finish_reason": None}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "lookup",
                                            "arguments": None,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        )
    )
    events: list[object] = []

    response = await LiteLLMBackend(completion=completion).generate(
        ModelRequest((Message.user("hello"),)),
        model="model",
        emit=events.append,
    )

    assert dict(response.tool_calls[0].arguments) == {}
    tool_event = next(event for event in events if isinstance(event, ToolCallDelta))
    assert tool_event.arguments_delta == ""


async def test_final_response_is_promoted_to_a_synthetic_stream() -> None:
    completion = CaptureCompletion(
        {
            "id": "final",
            "model": "model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "thinking",
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    )
    events: list[object] = []

    response = await LiteLLMBackend(completion=completion).generate(
        ModelRequest((Message.user("hello"),)),
        model="model",
        emit=events.append,
    )

    assert [type(event) for event in events] == [
        ModelStreamStarted,
        TextDelta,
        ToolCallDelta,
        UsageUpdate,
        ModelStreamCompleted,
    ]
    assert isinstance(events[-1], ModelStreamCompleted)
    assert events[-1].response is response


async def test_synthetic_tool_only_response_without_usage_has_minimal_events() -> None:
    completion = CaptureCompletion(
        {
            "id": "final",
            "model": "model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {"name": "lookup", "arguments": ""},
                            }
                        ],
                    },
                }
            ],
            "usage": None,
        }
    )
    events: list[object] = []

    await LiteLLMBackend(completion=completion).generate(
        ModelRequest((Message.user("hello"),)),
        model="model",
        emit=events.append,
    )

    assert [type(event) for event in events] == [
        ModelStreamStarted,
        ToolCallDelta,
        ModelStreamCompleted,
    ]


async def test_stream_for_non_streaming_request_is_rejected() -> None:
    backend = LiteLLMBackend(completion=CaptureCompletion(ChunkStream(())))

    with pytest.raises(ModelRequestError, match="stream for a non-streaming"):
        await backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
        )


async def test_empty_stream_and_missing_tool_name_are_rejected() -> None:
    empty_backend = LiteLLMBackend(completion=CaptureCompletion(ChunkStream(())))
    with pytest.raises(ModelRequestError, match="produced no chunks"):
        await empty_backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
            emit=lambda event: None,
        )

    missing_name = ChunkStream(
        (
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call", "function": {"arguments": "{}"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )
    )
    missing_backend = LiteLLMBackend(completion=CaptureCompletion(missing_name))
    with pytest.raises(ModelRequestError, match="has no function name"):
        await missing_backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
            emit=lambda event: None,
        )


async def test_invalid_streamed_tool_argument_type_is_rejected() -> None:
    stream = ChunkStream(
        (
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "tool",
                                        "arguments": ["bad"],
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        )
    )

    with pytest.raises(ModelRequestError, match="must be a string or object"):
        await LiteLLMBackend(completion=CaptureCompletion(stream)).generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
            emit=lambda event: None,
        )


async def test_stream_iteration_errors_are_mapped_but_sink_errors_propagate() -> None:
    rate_limit = RateLimitError("slow down", "fixture", "model")
    backend = LiteLLMBackend(completion=CaptureCompletion(ChunkStream((), error=rate_limit)))
    with pytest.raises(ModelRateLimitError):
        await backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
            emit=lambda event: None,
        )

    sink_backend = LiteLLMBackend(
        completion=CaptureCompletion(ChunkStream(({"choices": [{"delta": {"content": "x"}}]},)))
    )

    def broken_sink(event: object) -> None:
        del event
        raise RuntimeError("sink failed")

    with pytest.raises(RuntimeError, match="sink failed"):
        await sink_backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
            emit=broken_sink,
        )


@pytest.mark.parametrize("close_error", [None, RuntimeError("close failed")])
async def test_local_stream_failure_closes_stream_and_releases_callback_barrier(
    monkeypatch: pytest.MonkeyPatch,
    close_error: BaseException | None,
) -> None:
    stream = _ClosableChunkStream(close_error=close_error)

    async def incomplete_litellm(**_kwargs: object) -> object:
        return stream

    monkeypatch.setattr(litellm, "acompletion", incomplete_litellm)
    backend = LiteLLMBackend(callbacks=(_unused_callback,))
    sink_error = RuntimeError("sink failed")

    def broken_sink(_event: object) -> None:
        raise sink_error

    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(
            backend.generate(
                ModelRequest((Message.user("hello"),)),
                model="model",
                emit=broken_sink,
            ),
            timeout=1,
        )

    assert caught.value is sink_error
    assert stream.closed is True


async def test_stream_cancellation_closes_stream_and_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _BlockingClosableStream()

    async def incomplete_litellm(**_kwargs: object) -> object:
        return stream

    monkeypatch.setattr(litellm, "acompletion", incomplete_litellm)
    backend = LiteLLMBackend(callbacks=(_unused_callback,))
    task = asyncio.create_task(
        backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
            emit=lambda _event: None,
        )
    )

    await stream.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert stream.closed is True


async def test_broken_async_iterator_is_closed_without_masking_its_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator_error = RuntimeError("iterator setup failed")
    stream = _BrokenClosableStream(iterator_error)

    async def incomplete_litellm(**_kwargs: object) -> object:
        return stream

    monkeypatch.setattr(litellm, "acompletion", incomplete_litellm)
    backend = LiteLLMBackend(callbacks=(_unused_callback,))

    with pytest.raises(RuntimeError) as caught:
        await backend.generate(
            ModelRequest((Message.user("hello"),)),
            model="model",
            emit=lambda _event: None,
        )

    assert caught.value is iterator_error
    assert stream.closed is True


async def test_stream_cleanup_accepts_a_synchronous_aclose() -> None:
    class SyncCloser:
        def __init__(self) -> None:
            self.closed = False

        def aclose(self) -> None:
            self.closed = True

    stream = SyncCloser()

    await _close_stream_after_local_exit(stream)

    assert stream.closed is True


async def test_default_litellm_acompletion_works_offline_with_mock_response() -> None:
    backend = LiteLLMBackend(default_options={"mock_response": "mocked"})

    response = await backend.generate(
        ModelRequest((Message.user("hello"),)),
        model="openai/test",
    )

    assert response.text == "mocked"
