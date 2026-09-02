# Changelog

All notable changes to `purrcept_litellm` are documented here.

## Unreleased

- Serialize Core `SystemReminder` values as Chat Completions `user` messages.
  Only `SystemInstruction` remains `system`. `INSTRUCTIONS` placement still
  precedes history; `TAIL` / `AUTO` follow it. Emitting reminders as `system`
  let gateways hoist them into the prompt prefix and truncate prefix cache.

## 0.1.0 - 2026-07-28

- Add a LiteLLM-backed implementation of the `purrcept_core.models.ModelBackend`
  protocol.
- Map text, images, tool calls, tool results, settings, usage, reasoning data,
  provider metadata, and streaming events.
- Preserve Core prompt-control ordering and reminder lifecycle semantics.
- Add namespaced LiteLLM options, immutable backend configuration, prompt-cache
  hints, and normalized provider errors.
- Add Core provider-conformance coverage, offline unit tests, runnable examples,
  and an opt-in real-provider integration test.
- Add OpenInference LLM classification, compact request/response summaries, and
  structured tool schemas so Phoenix can render role-grouped prompts and Span Replay.
