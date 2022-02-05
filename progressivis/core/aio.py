from __future__ import annotations

import sys

from asyncio import (
    sleep,
    Lock,
    Event,
    Condition,
    gather,
    wait,
    run,
    Future,
    Task,
    iscoroutinefunction,
    FIRST_COMPLETED,
    ALL_COMPLETED,
)
from asyncio import create_task as _create_task
from asyncio import (
    set_event_loop,
    # get_event_loop as get_event_loop,
    new_event_loop,
    get_running_loop
)

from typing import Coroutine, Any, Optional


async def _gather(*coros: Coroutine[Any, Any, Any]) -> None:
    await gather(*coros)


def run_gather(*coros: Coroutine[Any, Any, Any]) -> None:
    return run(_gather(*coros))


if sys.version_info < (3, 8):

    def create_task(
        coroutine: Coroutine[Any, Any, Any], name: Optional[str] = None
    ) -> Task[Any]:
        return _create_task(coroutine)


else:

    def create_task(
        coroutine: Coroutine[Any, Any, Any], name: Optional[str] = None
    ) -> Task[Any]:
        return _create_task(coroutine, name=name)


__all__ = [
    "sleep",
    "Lock",
    "Event",
    "Condition",
    "gather",
    "wait",
    "Future",
    "Task",
    "iscoroutinefunction",
    "set_event_loop",
    "new_event_loop",
    "get_running_loop",
    "FIRST_COMPLETED",
    "ALL_COMPLETED",
    "create_task",
    "run",
]
