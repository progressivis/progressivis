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
    """This module runs a function at each run_step with the content of
    its input slot, which can be of any type.  By default, it prints
    the length of its input, which could be useful for PTable, PDict, and
    PIntSet.
    """

    def __init__(
        self,
        proc: Callable[[Any], None] = _print_len,
        constant_time: bool = True,
        **kwds: Any,
    ) -> None:
        r"""
        Args:
            proc: callable with one argument, which will be the input slot data.
            constant_time: True by default, set it to False if the printing function takes a time roughly proportional to the data size
            kwds: extra keyword args to be passed to the :class:`Module <progressivis.core.Module>` superclass
        """
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
    """
    This module prints the contents of its input slot.
    """

    def __init__(self, **kwds: Any) -> None:
        r"""
        Args:
            proc: callable with one argument, which will be the input slot data.
            kwds: extra keyword args to be passed to the :class:`Module <progressivis.core.Module>` superclass
        """
        if "proc" not in kwds:
            kwds["proc"] = _prt
        super().__init__(quantum=0.1, constant_time=True, **kwds)


def _tick(x: Any, tick: str) -> None:
    print(tick, end="", flush=True)

class Tick(Every):
    """
    This module prints a tick string when data arrives in the input slot, by default, a dot.
    It is useful to monitor the liveliness of a pipeline; different characters or strings can
    differentiate different parts of a pipeline.
    """

    def __init__(self, tick: str='.', **kwds: Any) -> None:
        r"""
        Args:
            tick: the string to print at every run_step
            kwds: extra keyword args to be passed to the :class:`Module <progressivis.core.Module>` superclass
        """
        if "proc" not in kwds:
            kwds["proc"] = partial(_tick, tick=tick)
        super().__init__(quantum=0.1, constant_time=True, **kwds)
