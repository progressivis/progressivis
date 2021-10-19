import sys

from asyncio import (sleep, Lock, Event, Condition, gather, wait, run, Future,
                     iscoroutinefunction, FIRST_COMPLETED, ALL_COMPLETED)
from asyncio import create_task as _create_task
from asyncio import set_event_loop, get_event_loop, new_event_loop, get_running_loop


async def _gather(*coros):
    await gather(*coros)


def run_gather(*coros):
    return run(_gather(*coros))

major, minor, *_ = sys.version_info
assert major == 3
if minor <= 7:
    def create_task(coroutine, name=None):
        return _create_task(coroutine)
else:
    def create_task(coroutine, name=None):
        return _create_task(coroutine, name=name)

__all__ = ["sleep", "Lock", "Event", "Condition", "gather", "wait",
           "Future", "iscoroutinefunction",
           "set_event_loop", "new_event_loop", "get_running_loop",
           "FIRST_COMPLETED", "ALL_COMPLETED",
           "create_task"]
