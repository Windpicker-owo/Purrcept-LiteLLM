from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from purrcept_core.models import (
    BackendConformanceCase,
    BackendConformanceScenario,
    JsonValue,
    Model,
    ModelResponse,
    run_backend_conformance,
)

from purrcept_litellm import LiteLLMBackend

from .helpers import ChunkStream


class ScenarioCompletion:
    def __init__(self, scenario: BackendConformanceScenario) -> None:
        self._scenario = scenario
        self._calls = 0

    async def __call__(self, **kwargs: object) -> object:
        assert kwargs["model"] == self._scenario.model_name
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        _assert_prompt_control(self._scenario.case, self._calls, kwargs)

        if self._calls == 0:
            error = self._scenario.expected_error
            response = self._scenario.expected_response
        else:
            step = self._scenario.follow_up_steps[self._calls - 1]
            error = None
            response = step.expected_response
        self._calls += 1
        if error is not None:
            raise error
        assert response is not None
        return ChunkStream((_response_chunk(response),))


def _assert_prompt_control(
    case: BackendConformanceCase,
    call_index: int,
    kwargs: Mapping[str, object],
) -> None:
    messages = cast(list[dict[str, object]], kwargs["messages"])
    serialized = json.dumps(messages, ensure_ascii=False)
    allowed = {
        "role",
        "content",
        "name",
        "tool_calls",
        "tool_call_id",
        "cache_control",
    }
    assert all(set(message) <= allowed for message in messages)

    if case is BackendConformanceCase.REQUEST_SEMANTICS:
        # Reminders are user messages. INSTRUCTIONS still precede history;
        # TAIL and AUTO follow it. Only SystemInstruction uses system.
        assert (
            serialized.index("Follow the conformance contract.")
            < serialized.index("Use the requested response format.")
            < serialized.index("Use the lookup tool if needed.")
            < serialized.index("Return concise output.")
            < serialized.index("Preserve the adapter fallback semantics.")
        )
        assert kwargs["tool_choice"] == "required"
        assert kwargs["parallel_tool_calls"] is True
    elif case is BackendConformanceCase.REMINDER_REMOVAL:
        if call_index == 0:
            assert "temporary-control-old" in serialized
        else:
            assert "temporary-control-old" not in serialized
    elif case is BackendConformanceCase.REMINDER_REPLACEMENT:
        if call_index == 0:
            assert "persistent-control-old" in serialized
            assert "persistent-control-new" not in serialized
        else:
            assert "persistent-control-old" not in serialized
            assert "persistent-control-new" in serialized


def _response_chunk(response: ModelResponse) -> dict[str, object]:
    tool_calls = [
        {
            "index": index,
            "id": tool.id,
            "function": {
                "name": tool.name,
                "arguments": json.dumps(dict(tool.arguments), separators=(",", ":")),
            },
        }
        for index, tool in enumerate(response.tool_calls)
    ]
    delta: dict[str, object] = {}
    if response.text:
        delta["content"] = response.text
    if tool_calls:
        delta["tool_calls"] = tool_calls

    provider_fields: dict[str, object] = dict(response.provider_metadata)
    if response.continuation is not None:
        provider_fields["purrcept_continuation"] = {
            "provider": response.continuation.provider,
            "data": _ordinary_json(response.continuation.data),
        }
    usage: object = None
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": response.usage.cached_input_tokens,
            "cache_creation_input_tokens": response.usage.cache_write_input_tokens,
            "reasoning_tokens": response.usage.reasoning_tokens,
        }
    return {
        "id": response.response_id,
        "model": response.model,
        "choices": [
            {
                "finish_reason": response.finish_reason.value,
                "delta": delta,
            }
        ],
        "usage": usage,
        "provider_specific_fields": provider_fields,
    }


def _ordinary_json(value: Mapping[str, JsonValue]) -> dict[str, object]:
    return {
        key: (
            _ordinary_json(cast(Mapping[str, JsonValue], item))
            if isinstance(item, Mapping)
            else list(item)
            if isinstance(item, tuple)
            else item
        )
        for key, item in value.items()
    }


async def test_litellm_backend_passes_core_provider_conformance() -> None:
    def factory(scenario: BackendConformanceScenario) -> Model:
        return Model(
            LiteLLMBackend(
                allow_strict_prompt_cache=True,
                completion=ScenarioCompletion(scenario),
            ),
            scenario.model_name,
        )

    report = await run_backend_conformance(factory)

    report.raise_for_failures()
    assert report.passed
