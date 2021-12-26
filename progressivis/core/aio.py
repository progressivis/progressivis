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
    iscoroutinefunction,
    FIRST_COMPLETED,
    ALL_COMPLETED,
)
from asyncio import create_task as _create_task
from asyncio import set_event_loop, new_event_loop, get_running_loop

from typing import Coroutine, Any


async def _gather(*coros: Coroutine[Any, Any, Any]) -> None:
    await gather(*coros)


def run_gather(*coros: Coroutine[Any, Any, Any]) -> None:
    return run(_gather(*coros))


if sys.version.startswith("3.7."):
    def create_task(coroutine: Coroutine, name: str = None):
        return _create_task(coroutine)
elif sys.version.startswith("3.8."):
    def create_task(coroutine: Coroutine, name: str = None):
        return _create_task(coroutine, name=name)


__all__ = [
    "sleep",
    "Lock",
    "Event",
    "Condition",
    "gather",
    "wait",
    "Future",
    "iscoroutinefunction",
    "set_event_loop",
    "new_event_loop",
    "get_running_loop",
    "FIRST_COMPLETED",
    "ALL_COMPLETED",
    "create_task",
]
