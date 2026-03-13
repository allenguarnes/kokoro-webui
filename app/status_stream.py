from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

_SUBSCRIBER_DISCONNECT_CHECK_SECONDS = 15.0


@dataclass(frozen=True)
class StatusStreamEvent:
    event: str
    data: str
    event_id: str


class StatusStreamHub(Protocol):
    """Single-node status broadcast contract.

    The current app uses an in-memory implementation, but this protocol is the
    seam for future multi-node backends such as Redis Pub/Sub or Redis Streams.
    """

    def subscriber_count(self) -> int: ...

    def request_refresh(self) -> None: ...

    async def wait_for_refresh(self, timeout_seconds: float) -> bool: ...

    async def publish_snapshot(self, snapshot: dict[str, object]) -> bool: ...

    def prime_snapshot(self, snapshot: dict[str, object]) -> bool: ...

    def subscribe(self) -> asyncio.Queue[StatusStreamEvent]: ...

    def unsubscribe(self, queue: asyncio.Queue[StatusStreamEvent]) -> None: ...


class InMemoryStatusStreamHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[StatusStreamEvent]] = set()
        self._last_serialized_snapshot: str | None = None
        self._next_event_id = 0
        self._refresh_event = asyncio.Event()

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def request_refresh(self) -> None:
        self._refresh_event.set()

    async def wait_for_refresh(self, timeout_seconds: float) -> bool:
        try:
            await asyncio.wait_for(self._refresh_event.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return False
        finally:
            self._refresh_event.clear()
        return True

    async def publish_snapshot(self, snapshot: dict[str, object]) -> bool:
        serialized = json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
        if serialized == self._last_serialized_snapshot:
            return False
        self._last_serialized_snapshot = serialized
        self._next_event_id += 1
        event = StatusStreamEvent(
            event="health_snapshot",
            data=serialized,
            event_id=str(self._next_event_id),
        )
        stale_queues: list[asyncio.Queue[StatusStreamEvent]] = []
        for queue in self._subscribers:
            try:
                if queue.full():
                    _ = queue.get_nowait()
                queue.put_nowait(event)
            except asyncio.QueueFull:
                stale_queues.append(queue)
        for queue in stale_queues:
            self._subscribers.discard(queue)
        return True

    def prime_snapshot(self, snapshot: dict[str, object]) -> bool:
        serialized = json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
        if serialized == self._last_serialized_snapshot:
            return False
        self._last_serialized_snapshot = serialized
        self._next_event_id += 1
        return True

    def subscribe(self) -> asyncio.Queue[StatusStreamEvent]:
        queue: asyncio.Queue[StatusStreamEvent] = asyncio.Queue(maxsize=1)
        if self._last_serialized_snapshot is not None:
            queue.put_nowait(
                StatusStreamEvent(
                    event="health_snapshot",
                    data=self._last_serialized_snapshot,
                    event_id=str(self._next_event_id),
                )
            )
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[StatusStreamEvent]) -> None:
        self._subscribers.discard(queue)


async def iter_status_events(
    hub: StatusStreamHub,
    *,
    disconnected: Callable[[], Awaitable[bool]],
) -> AsyncGenerator[dict[str, str], None]:
    queue = hub.subscribe()
    try:
        while True:
            if await disconnected():
                return
            try:
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=_SUBSCRIBER_DISCONNECT_CHECK_SECONDS,
                )
            except TimeoutError:
                continue
            yield {
                "event": event.event,
                "data": event.data,
                "id": event.event_id,
            }
    finally:
        hub.unsubscribe(queue)


def create_status_stream_hub(*, backend: str = "memory") -> StatusStreamHub:
    if backend != "memory":
        raise ValueError(f"Unsupported status stream backend: {backend}")
    return InMemoryStatusStreamHub()
