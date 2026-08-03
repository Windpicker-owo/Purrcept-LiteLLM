"""Run one real, streamed Purrcept conversation through LiteLLM."""

from __future__ import annotations

import asyncio
import os

from purrcept_core import AgentDriver, AgentFlow, InlineExecutor
from purrcept_core.models import Conversation, Model, ModelSettings

from purrcept_litellm import LiteLLMBackend


def chat(conversation: Conversation, question: str) -> AgentFlow[str]:
    result = yield from conversation.ask(question)
    return result.text


async def main() -> None:
    model_name = os.environ.get("PURRCEPT_LITELLM_MODEL")
    if not model_name:
        raise RuntimeError("Set PURRCEPT_LITELLM_MODEL, for example openai/gpt-4o-mini.")

    backend = LiteLLMBackend(
        api_key=os.environ.get("PURRCEPT_LITELLM_API_KEY"),
        api_base=os.environ.get("PURRCEPT_LITELLM_API_BASE"),
        timeout=60,
    )
    conversation = Model(backend, model_name).conversation(
        instructions="You are a concise and helpful assistant.",
        settings=ModelSettings(max_output_tokens=512),
    )
    answer = await AgentDriver(InlineExecutor()).run(
        chat(conversation, "用一句话解释 Purrcept 的 Effect 驱动执行方式。"),
        host=None,
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
