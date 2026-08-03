from __future__ import annotations

import os

import pytest
from purrcept_core.models import (
    Message,
    ModelRequest,
    ModelSettings,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelStreamStarted,
)

from purrcept_litellm import LiteLLMBackend

_ENABLED = os.environ.get("PURRCEPT_LITELLM_RUN_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not _ENABLED,
    reason="set PURRCEPT_LITELLM_RUN_INTEGRATION=1 to call a real provider",
)
async def test_real_provider_streaming_round_trip() -> None:
    model = os.environ.get("PURRCEPT_LITELLM_MODEL")
    if not model:
        pytest.fail("PURRCEPT_LITELLM_MODEL is required for the integration test.")

    backend = LiteLLMBackend(
        api_key=os.environ.get("PURRCEPT_LITELLM_API_KEY"),
        api_base=os.environ.get("PURRCEPT_LITELLM_API_BASE"),
        timeout=60,
        default_options={"drop_params": True},
    )
    request = ModelRequest(
        (Message.user("Reply with a short confirmation that the Purrcept integration works."),),
        settings=ModelSettings(temperature=0, max_output_tokens=64),
    )
    events: list[ModelStreamEvent] = []

    response = await backend.generate(request, model=model, emit=events.append)

    assert response.text.strip()
    assert isinstance(events[0], ModelStreamStarted)
    assert isinstance(events[-1], ModelStreamCompleted)
    assert events[-1].response == response
