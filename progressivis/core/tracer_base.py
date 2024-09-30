from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Callable, Any, Optional, List, TYPE_CHECKING


if TYPE_CHECKING:
    from progressivis.table.api import PTable


class Tracer(metaclass=ABCMeta):
    default: Callable[[str, Any], "Tracer"]

    @abstractmethod
    def start_run(self, ts: float, run_number: int, **kwds: Any) -> None:
        pass

    @abstractmethod
    def end_run(self, ts: float, run_number: int, **kwds: Any) -> None:
        pass

    @abstractmethod
    def run_stopped(self, ts: float, run_number: int, **kwds: Any) -> None:
        pass

    @abstractmethod
    def before_run_step(self, ts: float, run_number: int, **kwds: Any) -> None:
        pass

    @abstractmethod
    def after_run_step(self, ts: float, run_number: int, **kwds: Any) -> None:
        pass

    @abstractmethod
    def exception(self, ts: float, run_number: int, **kwds: Any) -> None:
        pass

    @abstractmethod
    def terminated(self, ts: float, run_number: int, **kwds: Any) -> None:
        pass

    @abstractmethod
    def trace_stats(self, max_runs: Optional[int] = None) -> PTable:
        _ = max_runs  # keeps pylint mute
        raise NotImplementedError("trace_stats")

    @abstractmethod
    def get_speed(self, depth: int = 15) -> List[Optional[float]]:
        return []
