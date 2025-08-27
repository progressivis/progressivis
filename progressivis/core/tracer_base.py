"""
Interface for tracing the scheduler activity.
Mainly used by the TimePredictor to compute the module speed.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Callable, Any, Optional, List, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from progressivis.table.api import PTable


class Tracer(metaclass=ABCMeta):
    default: Callable[[str, Any], "Tracer"]

    @abstractmethod
    def start_run(self, ts: float, run_number: int) -> None:
        pass

    @abstractmethod
    def end_run(
        self,
        ts: float,
        run_number: int,
        progress_current: float,
        progress_max: float,
        # quality: float,
        next_state: int,
        step_run: int,
        debug: bool,
    ) -> None:
        pass

    @abstractmethod
    def run_stopped(self, run_number: int) -> None:
        pass

    @abstractmethod
    def exception(self, run_number: int) -> None:
        pass

    @abstractmethod
    def terminated(self, run_number: int) -> None:
        pass

    @abstractmethod
    def trace_stats(self, max_runs: Optional[int] = None) -> PTable:
        _ = max_runs  # keeps pylint mute
        raise NotImplementedError("trace_stats")

    @abstractmethod
    def get_speed(self, depth: int = 15) -> List[Optional[float]]:
        return []

    @abstractmethod
    def last_steps_durations(self, length: int = 7) -> Tuple[List[float], List[int]]:
        pass
