"""Verify image delivery after the real LiteLLM SDK's provider transformations."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from typing import cast

from purrcept_core.models import (
    ImageBlock,
    ImageBytes,
    Message,
    MessageRole,
    ModelRequest,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)

from purrcept_litellm import LiteLLMBackend


async def test_deepseek_tool_image_survives_actual_http_transport() -> None:
    """Capture HTTP bytes rather than mocking acompletion before it drops image parts."""

    received: list[dict[str, object]] = []
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = next(
                int(line.split(":", 1)[1])
                for line in headers.splitlines()
                if line.lower().startswith("content-length:")
            )
            received.append(json.loads(await reader.readexactly(length)))
            chunk = {
                "id": "local-image",
                "model": "deepseek-v4-flash-vision-exp",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            }
            body = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        backend = LiteLLMBackend(api_key="local-test-key", api_base=f"http://127.0.0.1:{port}/v1")
        request = ModelRequest(
            messages=(
                Message.user("Read the tool image."),
                Message(
                    MessageRole.ASSISTANT,
                    (
                        ReasoningBlock("I need the image."),
                        ToolCallBlock("image-call", "play_action", arguments={}),
                    ),
                ),
                Message(
                    MessageRole.TOOL,
                    (
                        ToolResultBlock(
                            "image-call",
                            content=(
                                TextBlock("Loaded image bytes."),
                                ImageBlock(ImageBytes(png, "image/png")),
                            ),
                        ),
                    ),
                ),
            )
        )
        response = await backend.generate(request, model="deepseek/deepseek-v4-flash-vision-exp")
        assert response.text == "ok"
    assert len(received) == 1
    wire = received[0]
    assert wire["model"] == "deepseek-v4-flash-vision-exp"
    messages = wire["messages"]
    assert isinstance(messages, list)
    assert messages[1]["reasoning_content"] == "I need the image."
    images: list[object] = []
    for message in cast(list[Mapping[str, object]], messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in cast(list[Mapping[str, object]], content):
            if part.get("type") == "image_url":
                source = cast(Mapping[str, object], part["image_url"])
                images.append(source["url"])
    assert images == ["data:image/png;base64," + base64.b64encode(png).decode()]
