from __future__ import annotations

from functools import partial
from progressivis.core.module import (
    Module,
    ReturnRunStep,
    def_input,
)
from typing import Any, Callable, Sized


def _print_len(x: Sized) -> None:
    if x is not None:
        print(len(x))


@def_input("df")
class Every(Module):
    "Module running a function at each iteration"

    def __init__(
        self,
        proc: Callable[[Any], None] = _print_len,
        constant_time: bool = True,
        disp: str | None = None,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self._proc = proc
        self._constant_time = constant_time

    def predict_step_size(self, duration: float) -> int:
        if self._constant_time:
            return 1
        return super().predict_step_size(duration)

    def run_step(
        self, run_number: int, step_size: float, quantum: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("df")
        df = slot.data()
        self._proc(df)
        slot.clear_buffers()
        return self._return_run_step(Module.state_blocked, steps_run=1)


def _prt(x: Any) -> None:
    print(x)


class Print(Every):
    "Module to print its input slot"

    def __init__(self, **kwds: Any) -> None:
        if "proc" not in kwds:
            kwds["proc"] = _prt
        super().__init__(quantum=0.1, constant_time=True, **kwds)


def _tick(x: Any, tick: str) -> None:
    print(tick, end="", flush=True)

class Tick(Every):
    "Module to print a tick string when data arrives in the input slot"

    def __init__(self, tick: str='.', **kwds: Any) -> None:
        if "proc" not in kwds:
            kwds["proc"] = partial(_tick, tick=tick)
        super().__init__(quantum=0.1, constant_time=True, **kwds)
