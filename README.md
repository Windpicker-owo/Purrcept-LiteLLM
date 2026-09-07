# Purrcept LiteLLM

`purrcept_litellm` 是 `purrcept_core` 的 LiteLLM Provider 插件。它实现
`ModelBackend` 协议，把 Core 的统一模型请求映射到 LiteLLM Chat Completions，并将
Provider 响应、工具调用、流式增量和错误重新归一化为 Core 类型。

当前版本：`0.1.3`，对应 `purrcept_core >=0.5.1,<0.6.0`、`litellm >=1.100.0,<2`。

## 安装

```bash
pip install purrcept_litellm
```

在本仓库开发：

```bash
uv sync
```

## 最小用法

LiteLLM 使用带 Provider 前缀的模型名，例如 `openai/gpt-4o-mini`、
`anthropic/claude-sonnet-4-5`。推荐让 LiteLLM 从对应 Provider 的标准环境变量读取
凭据：

```powershell
$env:OPENAI_API_KEY = "..."
$env:PURRCEPT_LITELLM_MODEL = "openai/gpt-4o-mini"
uv run python examples/01_chat.py
```

也可以显式传入凭据或接入 LiteLLM Proxy / OpenAI-compatible 服务：

```python
import os

from purrcept_core import AgentDriver, AgentFlow, InlineExecutor
from purrcept_core.models import Conversation, Model
from purrcept_litellm import LiteLLMBackend


def chat(conversation: Conversation) -> AgentFlow[str]:
    result = yield from conversation.ask("你好，请简短介绍自己。")
    return result.text


async def main() -> None:
    backend = LiteLLMBackend(
        api_key=os.environ["LLM_API_KEY"],
        api_base="https://llm.example.com/v1",
        timeout=60,
    )
    conversation = Model(backend, "openai/my-model").conversation(
        instructions="你是一名简洁、可靠的助手。",
    )
    answer = await AgentDriver(InlineExecutor()).run(
        chat(conversation),
        host=None,
    )
    print(answer)
```

请求默认走流式 Chat Completions，避免长生成在网关空闲超时。`Generate` 会依次产生
`ModelStreamStarted`、`TextDelta` / `ToolCallDelta`、`UsageUpdate` 和
`ModelStreamCompleted`。即使调用方不提供 `emit`，Adapter 仍会消费流并返回组装后的
`ModelResponse`。

完整的自动工具循环见 [`examples/02_tool_loop.py`](examples/02_tool_loop.py)。

## 配置边界

运行时拥有后端和凭据：

```python
backend = LiteLLMBackend(
    api_key="...",
    api_base="https://...",
    api_version="...",
    timeout=60,
    default_options={
        "drop_params": True,
        "num_retries": 2,
    },
)
```

一次请求的 LiteLLM 特有选项必须放在唯一命名空间中：

```python
conversation = model.conversation(
    provider_options={
        "purrcept_litellm": {
            "reasoning_effort": "medium",
            "drop_params": True,
        }
    }
)
```

`model`、`messages`、`stream`、`stream_options`、`tools`、`api_key`、
`api_base`、`api_version`、`callback(s)`、`success_callback` 和
`failure_callback` 是保留字段，不能由请求级选项覆盖。凭据只应进入后端配置或 Provider
环境变量，不应写进对话状态。

## OpenTelemetry 可观测性

OpenTelemetry 是可选依赖：

```bash
pip install "purrcept_litellm[otel]"
```

下面的 callback 使用 LiteLLM 自带的 OpenTelemetry 集成，但拥有私有
`TracerProvider`，并且只通过单次请求的 success/failure callback 参数执行。它不会写入
LiteLLM 或 OpenTelemetry 的进程全局 callback/provider：

```python
from purrcept_litellm import (
    LiteLLMBackend,
    LiteLLMOpenTelemetryConfig,
    create_opentelemetry_callback,
)

otel = create_opentelemetry_callback(
    LiteLLMOpenTelemetryConfig(
        exporter="otlp_http",
        endpoint="http://127.0.0.1:4318/v1/traces",
        service_name="purrcept-engine",
        environment="development",
        capture_message_content="SPAN_ONLY",
    )
)
backend = LiteLLMBackend(callbacks=(otel,))

# Runtime 停止接收新请求并等待在途请求结束后：
otel.force_flush()
otel.shutdown()
```

`NO_CONTENT` 是默认值，只记录模型、耗时、token、状态和错误等运行信息。
`SPAN_ONLY` 会额外把真实请求提示词和模型返回写入 span 属性；它可能包含个人信息和密钥
片段，只应发送到可信的本地或受控 OTLP Collector。`headers` 使用 LiteLLM/OTLP 的字符串
格式，并会从配置对象的 `repr` 中隐藏。

每个主请求 span 同时带有 OpenInference 的 `LLM` 分类。支持 OpenInference 的界面（例如
Phoenix）可以按 System、User、Assistant 和 Tool 展开完整消息，并用最后一条输入和首个
模型结果填充紧凑的 Input/Output 摘要。工具声明以标准 `llm.tools.*.tool.json_schema`
记录，因此也可以进入 Phoenix Span Replay。完整消息仍只由 LiteLLM 的 `gen_ai.*` 属性
承载；Adapter 不会再镜像整段消息序列，但 Input 摘要会包含最后一条消息的完整 content。
`NO_CONTENT` 只保留 `LLM` 分类，不会通过这些补充字段泄露提示词、返回或工具 schema。

使用默认 completion 时，Backend 会在返回或抛错前等待本次 request-local callback
完成，因此 Terminal 这类短生命周期事件循环不会在关闭时取消尚未记录的 span。HTTP
exporter 仍由 `BatchSpanProcessor` 异步发送；Runtime 关闭阶段负责最终的
`force_flush()` 和 `shutdown()`。

## 映射范围

| Core | LiteLLM |
|---|---|
| `SystemInstruction` | 前置 `system` message |
| `Message` / `TextBlock` | chat message / text content |
| `ImageUrl` / `ImageBytes` | `image_url` / data URI |
| `ToolSpec` | OpenAI-compatible function tool |
| `ToolCallBlock` / `ToolResultBlock` | assistant tool call / tool message |
| `ModelSettings` | temperature、max tokens、stop、tool choice、parallel calls |
| `TokenUsage` | prompt、completion、cache、reasoning tokens |
| `ModelStreamEvent` | LiteLLM chunk lifecycle |
| LiteLLM exceptions | Core 的认证、限流、上下文、临时、不可用、请求错误 |

未知但可 JSON 化的 Provider 字段会保留在 `ModelResponse.provider_metadata`，不会污染
Core 的稳定协议。

## Prompt control、缓存与 continuation

- Core 已编译好的 reminder 一律作为 `user` message 发送，不进入 `system`。插件不重新
  解释 scope、priority 或 replacement；生命周期仍由 `Conversation` / `PromptCompiler`
  管理。只有 `SystemInstruction` 使用 `system`。
- `INSTRUCTIONS` placement 仍插在对话历史之前，但角色是 `user`。`TAIL` / `AUTO` 跟在
  历史之后。相对顺序保持 Core 编译结果。
- `cache="auto"`（Adapter 默认，等同 `prefer`）与 `cache="explicit"` 会在稳定
  instruction、最后一个 tool、以及不断增长的对话历史上添加 LiteLLM `cache_control`
  提示。Provider 是否支持和如何计费由 Provider 决定。
- reminder（含 `INSTRUCTIONS` 前置与 `TAIL` / `AUTO` 尾部）从不打显式缓存标记。它们是
  请求局部、常常易变的控制文本；标记它们只会制造几乎不会命中的 cache write。
- `cache="disabled"` 不添加任何 `cache_control`。
- strict cache 默认拒绝。只有确认目标 Provider 兼容后，才设置
  `allow_strict_prompt_cache=True`。
- LiteLLM Chat Completions 没有统一、可靠的服务器 continuation 句柄，因此
  `ModelContinuation` 不会默认发给 Provider。推荐使用 Core 默认的
  `continuation_policy="client_managed"`，由本地规范历史重建下一次请求。这也确保已过期
  reminder 不会因 Provider 保存了旧状态而泄漏到后续轮次。
- 自定义 LiteLLM Provider 若在 `provider_specific_fields.purrcept_continuation` 返回
  `{provider, data}`，插件会将其恢复为 Core 的不透明 continuation，供显式策略使用。

## 真实 Provider 验证

默认测试完全离线。启用真实流式调用：

```powershell
$env:PURRCEPT_LITELLM_RUN_INTEGRATION = "1"
$env:PURRCEPT_LITELLM_MODEL = "openai/gpt-4o-mini"
$env:OPENAI_API_KEY = "..."
uv run pytest tests/test_real_integration.py -m integration -q
```

若使用统一代理，可改用 `PURRCEPT_LITELLM_API_KEY` 和
`PURRCEPT_LITELLM_API_BASE`。

完整质量门：

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=purrcept_litellm --cov-branch
uv run python -m build
```

## 设计原则

`purrcept_core` 只定义 Provider 无关协议和 Agent 语义；本包只承担 LiteLLM
传输适配。具体模型凭据、网络客户端生命周期和部署策略归 Runtime 所有。这样 Core
不会依赖任何厂商 SDK，其他 Provider 也可以继续以独立 Python 插件实现。

### 工具结果中的图片

从 0.1.2 起，Core `ToolResultBlock` 中的图片在 Chat Completions 传输时映射到带
工具调用 ID 说明的 `user` 视觉消息。原始工具消息只保留文本；同一批工具回执全部发出后
才插入图片，随后再发送后续历史与尾部 Reminder。这样兼容只允许文本 tool 内容的接口，
同时保留 Core 历史中的真实图像块、错误标记和调用身份。

0.1.3 将 LiteLLM 最低版本提高到 1.100.0，采用其 DeepSeek 视觉修复
（[上游 #38397](https://github.com/BerriAI/litellm/pull/38397)）。旧版 DeepSeek 适配器会将
多模态 user 消息折叠为字符串，静默丢弃图片。现在保留原生 `deepseek/` 路由；回归测试
通过真实 LiteLLM SDK 向本地 HTTP 接收端发送请求，并核对最终 base64 图像字节。
