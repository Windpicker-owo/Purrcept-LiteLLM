# Changelog

All notable changes to `purrcept_litellm` are documented here.

## Unreleased

## 0.1.3 - 2026-09-08

- Publish validated wheel and source distributions automatically when a GitHub Release
  is published, using PyPI Trusted Publishing and an isolated publishing job.

- Remove obsolete non-streaming and pre-image-split cache/error branches; cover
  cache fallback past assistant tool calls without marking dynamic reminders.
- Require LiteLLM 1.100.0 for upstream DeepSeek vision forwarding (#38397).
  Verify tool-result image bytes at a loopback HTTP receiver after the real SDK
  transformations; retain native DeepSeek routing without local provider patches.
- Map images in Core tool results to a labelled user media message after the complete
  adjacent tool-response group. Chat Completions tool messages remain text-only;
  original Core history, tool-call identities, errors and reminder ordering are preserved.

- Emit Core `ReasoningDelta` events for readable provider reasoning while preserving
  the final `ReasoningBlock` and keeping assistant text on its own channel. Requires Core 0.5.1.

- Serialize Core `SystemReminder` values as Chat Completions `user` messages.
  Only `SystemInstruction` remains `system`. `INSTRUCTIONS` placement still
  precedes history; `TAIL` / `AUTO` follow it. Emitting reminders as `system`
  let gateways hoist them into the prompt prefix and truncate prefix cache.
- Treat `cache="auto"` as `prefer`: mark the stable instruction, last tool, and
  growing transcript. Never attach `cache_control` to reminder messages.
- Always send `stream=true`. Callers without an event sink still receive the
  assembled `ModelResponse`; non-streaming completions were timing out on long
  generations.

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
