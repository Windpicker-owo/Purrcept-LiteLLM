"""Keep readable reasoning observable without mixing it into assistant text."""

from purrcept_core.models import ReasoningBlock, ReasoningDelta, TextDelta

from purrcept_litellm._response import StreamAccumulator


def test_reasoning_fragments_emit_separately_and_preserve_final_state() -> None:
    """A live observer and the next model request receive the same reasoning text."""

    accumulator = StreamAccumulator("example")
    first = accumulator.ingest({"choices": [{"delta": {"reasoning_content": "Plan "}}]})
    second = accumulator.ingest(
        {"choices": [{"delta": {"reasoning_content": "ready", "content": "Hello"}}]}
    )
    events = (*first, *second)
    assert [event.delta for event in events if isinstance(event, ReasoningDelta)] == [
        "Plan ",
        "ready",
    ]
    assert [event.delta for event in events if isinstance(event, TextDelta)] == ["Hello"]
    response = accumulator.complete()
    assert response.message.content[0] == ReasoningBlock("Plan ready")
    assert response.text == "Hello"
