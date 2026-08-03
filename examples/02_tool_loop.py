"""Let a real model call a typed Python tool through Purrcept's tool loop."""

from __future__ import annotations

import asyncio
import os

from purrcept_core import AgentDriver, AgentFlow, InlineExecutor
from purrcept_core.models import Conversation, Model, ModelSettings, tool

from purrcept_litellm import LiteLLMBackend


@tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers.

    Args:
        left: The first integer.
        right: The second integer.
    """

    return left * right


def calculate(conversation: Conversation) -> AgentFlow[str]:
    result = yield from conversation.ask(
        "Call the multiply tool for 123 and 456, then report the exact result."
    )
    return result.text


async def main() -> None:
    model_name = os.environ.get("PURRCEPT_LITELLM_MODEL")
    if not model_name:
        raise RuntimeError("Set PURRCEPT_LITELLM_MODEL to a model that supports function calling.")

    backend = LiteLLMBackend(
        api_key=os.environ.get("PURRCEPT_LITELLM_API_KEY"),
        api_base=os.environ.get("PURRCEPT_LITELLM_API_BASE"),
        timeout=60,
    )
    conversation = Model(backend, model_name).conversation(
        instructions="Use the provided tool for arithmetic. Never calculate mentally.",
        tools=(multiply,),
        settings=ModelSettings(max_output_tokens=256),
    )
    answer = await AgentDriver(InlineExecutor()).run(
        calculate(conversation),
        host=None,
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
