from __future__ import annotations

from typing import Callable

from abc import abstractmethod
import logging
import numpy as np

from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from .tracer_base import Tracer

logger = logging.getLogger(__name__)


class TimePredictor:
    default: Callable[[], "TimePredictor"]

    def __init__(self) -> None:
        self.name: Optional[str] = None

    @abstractmethod
    def fit(self, tracer: Tracer) -> None:
        pass

    @abstractmethod
    def predict(self, duration: float, default_step: int) -> int:
        pass


class ConstantTimePredictor(TimePredictor):
    def __init__(self, t: int) -> None:
        super().__init__()
        self.t = t

    def predict(self, duration: float, default_step: int) -> int:
        return self.t


class LinearTimePredictor(TimePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.a: float = 0

    def fit(self, tracer: Tracer) -> None:
        if tracer is None:
            return

        # expr_ = (trace_df["type"] == "step") & (trace_df["duration"] != 0)
        # if expr_ is False:
        #     step_traces = np.array([], dtype="int64")
        # else:
        #     (step_traces,) = np.where(expr_)
        # n = len(step_traces)
        # if n < 1:
        #     return
        # if n > 7:
        #     step_traces = step_traces[-7:]
        # # (operations, durations) = tracer.last_steps_durations(7)
        # durations = trace_df["duration"][step_traces]
        # operations = trace_df["steps_run"][step_traces]
        durations, operations = tracer.last_steps_durations(7)

        num = np.sum(operations)
        time = np.sum(durations)
        if num == 0:
            return
        a = num / time
        if a > 0:
            self.a = a
        else:
            logger.debug(
                "LinearPredictor %s: predictor fit found a negative slope, ignoring",
                self.name,
            )

    def predict(self, duration: float, default_step: int) -> int:
        if self.a == 0:
            return default_step
        # TODO account for the confidence interval and take min of the 95% CI
        steps = int(np.max([0, np.ceil(duration * self.a)]))
        logger.debug(
            "LinearPredictor %s: Predicts %d steps for duration %f",
            self.name,
            steps,
            duration,
        )
        return steps


TimePredictor.default = LinearTimePredictor
