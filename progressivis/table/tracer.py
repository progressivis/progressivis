from __future__ import annotations

from ..core.tracer_base import Tracer
from .table import PTable
import numpy as np

from typing import Optional, List, Dict, Union, cast, Any, Tuple


class PTableTracer(Tracer):
    TRACER_DSHAPE = (
        "{"
        "type: string,"
        "run: int64,"
        "start: real,"
        "end: real,"
        "duration: real,"
        "detail: string,"
        "steps_run: int32,"
        "next_state: int32,"
        "progress_current: real,"
        "progress_max: real,"
        # "quality: real"
        "}"
    )
    TRACER_INIT: Dict[str, Union[int, float, np.floating[Any], str]] = dict(
        [
            ("type", ""),
            ("run", 0),
            ("start", np.nan),
            ("end", np.nan),
            ("duration", np.nan),
            ("detail", ""),
            ("steps_run", 0),
            ("next_state", 0),
            ("progress_current", 0.0),
            ("progress_max", 0.0),
            # ("quality", 0.0),
        ]
    )

    def __init__(self, name: str, storagegroup: Any) -> None:
        self.table = PTable(
            "trace_" + name,
            dshape=PTableTracer.TRACER_DSHAPE,
            storagegroup=storagegroup,
            chunks=256,
        )
        self.table.add(PTableTracer.TRACER_INIT)
        self.last_run_start: Dict[
            str, Union[int, float, np.floating[Any], str]
        ] | None = None
        self.last_run_details = ""
        self.has_debug = False

    def trace_stats(self, max_runs: Optional[int] = None) -> PTable:
        return self.table

    def start_run(self, ts: float, run_number: int) -> None:
        self.last_run_start = dict(PTableTracer.TRACER_INIT)
        self.last_run_start["run"] = run_number
        self.last_run_start["start"] = ts

    def end_run(
            self,
            ts: float,
            run_number: int,
            progress_current: float,
            progress_max: float,
            # quality: float,
            next_state: int,
            steps_run: int,
            debug: bool
    ) -> None:
        assert self.last_run_start
        row = self.last_run_start
        row["next_state"] = next_state
        row["steps_run"] = steps_run
        row["end"] = ts
        row["duration"] = ts - cast(float, row["start"])
        row["detail"] = self.last_run_details
        if debug:
            row["type"] = "debug_step"
            self.has_debug = True
        else:
            row["type"] = "step"
        row["progress_current"] = progress_current
        row["progress_max"] = progress_max
        # row["quality"] = quality
        self.table.add(row)
        self.last_run_step_start = None
        self.last_run_details = ""

    def run_stopped(self, run_number: int) -> None:
        self.last_run_details += "stopped"

    def exception(self, run_number: int) -> None:
        self.last_run_details += "exception"

    def terminated(self, run_number: int) -> None:
        self.last_run_details += "terminated"

    def get_speed(self, depth: int = 15) -> List[Optional[float]]:
        res: List[Optional[float]] = []
        elt: Optional[float]
        non_zero = self.table.eval("steps_run!=0", as_slice=False)
        sz = min(depth, len(non_zero))
        idx = non_zero[-sz:]
        for d, s in zip(
            self.table["duration"].loc[idx], self.table["steps_run"].loc[idx]
        ):
            if np.isnan(d) or d == 0:
                if not s:
                    continue
                elt = None
            else:
                elt = float(s) / d
            res.append(elt)
        return res

    def last_steps_durations(
            self, length: int = 7
    ) -> Tuple[List[float], List[int]]:
        # TODO optimize to search backward to avoid scanning the whole table
        if len(self.table) < 2:
            return ([], [])
        if self.has_debug:
            expr_ = (
                (self.table["type"].values == "step") &
                (self.table["duration"].values != 0)
            )
        else:
            expr_ = (self.table["duration"].values != 0)

        if len(expr_) == 0:
            step_traces = np.array([], dtype="int64")
        else:
            (step_traces,) = np.where(expr_)

        n = len(step_traces)
        if n == 0:
            return ([], [])
        if n > length:
            step_traces = step_traces[-length:]
        durations = self.table["duration"][step_traces]
        operations = self.table["steps_run"][step_traces]
        return (list(durations), list(operations))


Tracer.default = PTableTracer
