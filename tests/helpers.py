from __future__ import annotations

from collections.abc import AsyncIterator


class CaptureCompletion:
    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if not self.results:
            raise AssertionError("unexpected completion call")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class ChunkStream(AsyncIterator[object]):
    def __init__(
        self,
        chunks: tuple[object, ...],
        *,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self._index = 0

    def __aiter__(self) -> ChunkStream:
        return self

    async def __anext__(self) -> object:
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk
        if self._error is not None:
            error = self._error
            self._error = None
            raise error
        raise StopAsyncIteration


def text_response(
    text: str = "ok",
    *,
    response_id: str = "response-1",
    model: str = "fixture-model",
    finish_reason: str = "stop",
    usage: object = None,
    provider_specific_fields: object = None,
    reasoning_content: str | None = None,
) -> dict[str, object]:
    return {
        "id": response_id,
        "model": model,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": reasoning_content,
                    "tool_calls": None,
                },
            }
        ],
        "usage": usage,
        "provider_specific_fields": provider_specific_fields,
    }
