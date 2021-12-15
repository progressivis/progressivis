from typing import Callable, Any, Optional

from abc import ABCMeta, abstractmethod


class Tracer(metaclass=ABCMeta):
    default: Callable[[str, Any], 'Tracer']

    @abstractmethod
    def start_run(self, ts: float, run_number: int, **kwds):
        pass

    @abstractmethod
    def end_run(self, ts: float, run_number: int, **kwds):
        pass

    @abstractmethod
    def run_stopped(self, ts: float, run_number: int, **kwds):
        pass

    @abstractmethod
    def before_run_step(self, ts: float, run_number: int, **kwds):
        pass

    @abstractmethod
    def after_run_step(self, ts: float, run_number: int, **kwds):
        pass

    @abstractmethod
    def exception(self, ts: float, run_number: int, **kwds):
        pass

    @abstractmethod
    def terminated(self, ts: float, run_number: int, **kwds):
        pass

    @abstractmethod
    def trace_stats(self, max_runs: Optional[int] = None):
        _ = max_runs  # keeps pylint mute
        return []
