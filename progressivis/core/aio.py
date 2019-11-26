import sys

from asyncio import (sleep, Lock, Event, gather, wait, FIRST_COMPLETED, ALL_COMPLETED)
from asyncio import create_task as _create_task


if sys.version.startswith('3.7.'):
    def create_task(coroutine, name=None):
        return _create_task(coroutine)
elif sys.version.startswith('3.8.'):
    def create_task(coroutine, name=None):
        return _create_task(coroutine, name=name)

__all__ = ["sleep", "Lock", "Event", "gather", "wait",
           "FIRST_COMPLETED", "ALL_COMPLETED",
           create_task]
