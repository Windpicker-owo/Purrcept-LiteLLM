"""Execute Core model requests through LiteLLM Chat Completions.

This module owns the :class:`LiteLLMBackend` boundary and the lifecycle of one
invocation, not provider selection, callback construction, or callback exporter
shutdown.  A request is compiled by :mod:`purrcept_litellm._request`, executed
through the selected asynchronous completion callable, converted by
:mod:`purrcept_litellm._response`, and mapped onto Core errors when necessary.

The default LiteLLM SDK path additionally joins only the request-local callback
hooks it supplied.  This ordering is essential for short-lived event loops:
LiteLLM detaches async success logging, so returning earlier could cancel span
creation before an owning runtime can flush the exporter.  Injected completion
callables remain borrowed test or embedding seams and are not assumed to use
LiteLLM's background callback scheduler.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from importlib import import_module
from typing import NoReturn, Protocol, cast

from purrcept_core.models import (
    JsonValue,
    ModelEventSink,
    ModelRequest,
    ModelRequestError,
    ModelResponse,
    ModelStreamCompleted,
)

from ._request import compile_request
from ._response import StreamAccumulator, convert_response, synthetic_stream_events
from .config import LiteLLMConfig
from .errors import map_litellm_error


class CompletionCallable(Protocol):
    """An injected LiteLLM-compatible asynchronous completion function."""

    def __call__(self, **kwargs: object) -> Awaitable[object]:
        """Return a final response or async stream wrapper."""

        ...


class LiteLLMBackend:
    """Use LiteLLM Chat Completions as a unified Purrcept model backend.

    Runtime-owned callbacks are snapshotted when the backend is created. Every
    invocation receives fresh ``success_callback`` and ``failure_callback``
    lists because LiteLLM consumes and mutates those request-local lists while
    classifying synchronous and asynchronous callback hooks. The backend never
    registers callbacks in LiteLLM's process-global callback collections.
    """

    __slots__ = ("_await_callback_completion", "_callbacks", "_completion", "_config")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        allow_strict_prompt_cache: bool = False,
        default_options: Mapping[str, JsonValue] | None = None,
        callbacks: Sequence[object] = (),
        completion: CompletionCallable | None = None,
    ) -> None:
        self._config = LiteLLMConfig(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            timeout=timeout,
            allow_strict_prompt_cache=allow_strict_prompt_cache,
            default_options={} if default_options is None else default_options,
        )
        self._callbacks = _snapshot_callbacks(callbacks)
        selected = _default_completion if completion is None else completion
        if not callable(cast(object, selected)):
            raise TypeError("completion must be an asynchronous callable or None.")
        self._completion = cast(CompletionCallable, selected)
        # The injected completion seam is borrowed test/embedding code and may
        # not implement LiteLLM's callback scheduler. The built-in completion is
        # the boundary whose detached worker lifecycle this backend must join.
        self._await_callback_completion = completion is None

    @property
    def config(self) -> LiteLLMConfig:
        """Return the immutable backend configuration."""

        return self._config

    @property
    def callbacks(self) -> tuple[object, ...]:
        """Return the immutable callbacks attached to every model invocation."""

        return self._callbacks

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        """Compile, execute, and convert one model request."""

        payload = compile_request(
            request,
            model=model,
            config=self._config,
            stream=emit is not None,
        )
        callback_completion = (
            _RequestCallbackCompletion(self._callbacks)
            if self._callbacks and self._await_callback_completion
            else None
        )
        abandon_callbacks = (
            callback_completion.abandon
            if callback_completion is not None
            else _ignore_callback_completion
        )
        if callback_completion is not None:
            payload["success_callback"] = callback_completion.success_callbacks
            payload["failure_callback"] = callback_completion.failure_callbacks
        elif self._callbacks:
            # Compatibility: LiteLLM currently partitions callbacks by mutating
            # these lists. Fresh copies preserve backend ownership and prevent a
            # concurrent or previous request from changing later instrumentation.
            # Its async completion path also classifies a CustomLogger instance
            # inconsistently between success and failure. Selecting its concrete
            # event hooks here keeps both paths request-scoped and executable.
            payload["success_callback"] = [
                _event_callback(callback, method_name="async_log_success_event")
                for callback in self._callbacks
            ]
            payload["failure_callback"] = [
                _event_callback(callback, method_name="log_failure_event")
                for callback in self._callbacks
            ]
        try:
            try:
                raw = await self._completion(**payload)
            except Exception as error:
                _raise_mapped(error, model=model)

            if isinstance(raw, AsyncIterable):
                if emit is None:
                    abandon_callbacks()
                    await _close_stream_after_local_exit(cast(object, raw))
                    raise ModelRequestError(
                        "LiteLLM returned a stream for a non-streaming request.",
                        provider="litellm",
                        model=model,
                    )
                return await _consume_stream(
                    cast(AsyncIterable[object], raw),
                    emit=emit,
                    model=model,
                    abandon_callbacks=abandon_callbacks,
                )

            response = convert_response(raw, requested_model=model)
            if emit is not None:
                for event in synthetic_stream_events(response):
                    emit(event)
            return response
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Lifecycle: process or task termination may stop LiteLLM before it
            # can schedule either terminal callback. Releasing only our barrier
            # preserves that control-flow signal across every invocation phase.
            abandon_callbacks()
            raise
        finally:
            if callback_completion is not None:
                # Lifecycle: LiteLLM 1.94 dispatches successful SDK callbacks
                # through a detached global logging worker. Awaiting our
                # request-owned barriers keeps the callback on this event loop
                # without inspecting or mutating that global worker.
                await callback_completion.wait()


def _snapshot_callbacks(callbacks: Sequence[object]) -> tuple[object, ...]:
    """Validate and freeze callback ownership at the backend boundary."""

    raw_callbacks = cast(object, callbacks)
    if isinstance(raw_callbacks, (str, bytes)) or not isinstance(raw_callbacks, Sequence):
        raise TypeError("callbacks must be a sequence of LiteLLM callback values.")
    snapshot = tuple(callbacks)
    if any(callback is None for callback in snapshot):
        raise TypeError("callbacks must not contain None.")
    return snapshot


def _event_callback(callback: object, *, method_name: str) -> object:
    """Select a CustomLogger event hook while preserving plain callbacks."""

    candidate = getattr(callback, method_name, None)
    return candidate if callable(candidate) else callback


class _RequestCallbackCompletion:
    """Join callback work that LiteLLM otherwise detaches from ``acompletion``.

    One event belongs to each backend callback and is shared by its success and
    failure wrappers because exactly one terminal path is expected per request.
    Non-callable LiteLLM callback identifiers pass through unchanged; they have
    process-owned lifecycles and cannot provide a request-local completion
    signal. Runtime-created OpenTelemetry callbacks expose concrete event hooks
    and therefore always participate in the barrier.
    """

    __slots__ = ("_events", "failure_callbacks", "success_callbacks")

    def __init__(self, callbacks: tuple[object, ...]) -> None:
        events: list[asyncio.Event] = []
        success_callbacks: list[object] = []
        failure_callbacks: list[object] = []

        for callback in callbacks:
            success = _event_callback(callback, method_name="async_log_success_event")
            failure = _event_callback(callback, method_name="log_failure_event")
            if not callable(success) or not callable(failure):
                success_callbacks.append(success)
                failure_callbacks.append(failure)
                continue

            completed = asyncio.Event()
            events.append(completed)
            success_callbacks.append(_async_callback_with_completion(success, completed))
            failure_callbacks.append(_sync_callback_with_completion(failure, completed))

        self._events = tuple(events)
        self.success_callbacks = success_callbacks
        self.failure_callbacks = failure_callbacks

    async def wait(self) -> None:
        """Wait until every request-owned callable reaches a terminal hook."""

        if self._events:
            await asyncio.gather(*(event.wait() for event in self._events))

    def abandon(self) -> None:
        """Release waiters when LiteLLM cannot reach a terminal callback.

        Abandonment does not cancel a callback that is already running. It only
        makes the request barrier terminal so a local stream failure or caller
        cancellation cannot be replaced by an indefinite lifecycle wait.
        """

        for event in self._events:
            event.set()


def _async_callback_with_completion(
    callback: Callable[..., object],
    completed: asyncio.Event,
) -> Callable[..., Awaitable[object]]:
    """Wrap one success callback and signal after its returned awaitable ends."""

    async def invoke(*args: object, **kwargs: object) -> object:
        try:
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                return await cast(Awaitable[object], result)
            return result
        finally:
            completed.set()

    return invoke


def _sync_callback_with_completion(
    callback: Callable[..., object],
    completed: asyncio.Event,
) -> Callable[..., object]:
    """Wrap one failure callback, including an unexpectedly async result."""

    def invoke(*args: object, **kwargs: object) -> object:
        try:
            result = callback(*args, **kwargs)
        except BaseException:
            completed.set()
            raise
        if inspect.isawaitable(result):
            pending = asyncio.create_task(
                _await_callback_result(cast(Awaitable[object], result), completed)
            )
            pending.add_done_callback(_consume_callback_task_result)
        else:
            completed.set()
        return result

    return invoke


async def _await_callback_result(result: Awaitable[object], completed: asyncio.Event) -> None:
    """Complete an async failure hook scheduled from LiteLLM's sync dispatcher."""

    try:
        await result
    finally:
        completed.set()


def _consume_callback_task_result(task: asyncio.Task[None]) -> None:
    """Retrieve a failure-hook exception after LiteLLM's sync dispatcher returns."""

    try:
        task.result()
    except BaseException:
        # Compatibility: LiteLLM treats callback failures as non-blocking. The
        # completion event is the ownership signal; the provider result remains
        # authoritative even when an observability callback fails.
        pass


def _ignore_callback_completion() -> None:
    """Provide a uniform abandonment hook when no callback barrier exists."""


async def _default_completion(**kwargs: object) -> object:
    module = import_module("litellm")
    raw_function: object = module.__dict__["acompletion"]
    function = cast(Callable[..., Awaitable[object]], raw_function)
    return await function(**kwargs)


async def _consume_stream(
    stream: AsyncIterable[object],
    *,
    emit: ModelEventSink,
    model: str,
    abandon_callbacks: Callable[[], None] = _ignore_callback_completion,
) -> ModelResponse:
    """Consume one stream while separating provider and local termination.

    Provider iteration errors remain terminal LiteLLM outcomes, so the backend
    still joins their failure callbacks. A local parser, event sink, or caller
    interruption happens before LiteLLM observes normal exhaustion; those paths
    close the iterator and release the callback barrier before propagating the
    original exception.
    """

    accumulator = StreamAccumulator(model)
    try:
        iterator = aiter(stream)
    except BaseException:
        abandon_callbacks()
        await _close_stream_after_local_exit(stream)
        raise

    while True:
        try:
            chunk = await anext(iterator)
        except StopAsyncIteration:
            break
        except Exception as error:
            _raise_mapped(error, model=model)
        except BaseException:
            # Cancellation is local to the caller, so LiteLLM may never emit a
            # provider failure callback for this interrupted iterator.
            abandon_callbacks()
            await _close_stream_after_local_exit(iterator)
            raise

        try:
            for event in accumulator.ingest(chunk):
                emit(event)
        except BaseException:
            # The provider stream has not reached its terminal iteration yet.
            # Release the callback barrier before closing it, because closing a
            # partially consumed LiteLLM stream does not promise a terminal hook.
            abandon_callbacks()
            await _close_stream_after_local_exit(iterator)
            raise

    response = accumulator.complete()
    emit(ModelStreamCompleted(response))
    return response


async def _close_stream_after_local_exit(stream: object) -> None:
    """Close a partially consumed stream without replacing its causal error.

    LiteLLM and custom completion implementations may expose ``aclose`` without
    sharing a concrete stream type. Closing is best-effort because a cleanup
    failure must not mask the sink exception or cancellation that caused the
    early exit.
    """

    try:
        close = getattr(stream, "aclose", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await cast(Awaitable[object], result)
    except BaseException:
        pass


def _raise_mapped(error: Exception, *, model: str) -> NoReturn:
    mapped = map_litellm_error(error, model=model)
    if mapped is error:
        raise error
    raise mapped from error


__all__ = ["CompletionCallable", "LiteLLMBackend"]
