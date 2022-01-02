from __future__ import annotations

from ..core.tracer_base import Tracer
from .table import Table
import numpy as np

from typing import Optional, List, Dict, Union, cast, Any


class TableTracer(Tracer):
    TRACER_DSHAPE = (
        "{"
        "type: string,"
        "start: real,"
        "end: real,"
        "duration: real,"
        "detail: string,"
        "run: int64,"
        "steps: int32,"
        "steps_run: int32,"
        "next_state: int32,"
        "progress_current: real,"
        "progress_max: real,"
        "quality: real"
        "}"
    )
    TRACER_INIT = dict(
        [
            ("type", ""),
            ("start", np.nan),
            ("end", np.nan),
            ("duration", np.nan),
            ("detail", ""),
            ("run", 0),
            ("steps", 0),
            ("steps_run", 0),
            ("next_state", 0),
            ("progress_current", 0.0),
            ("progress_max", 0.0),
            ("quality", 0.0),
        ]
    )

    def __init__(self, name: str, storagegroup: Any) -> None:
        self.table = Table(
            "trace_" + name,
            dshape=TableTracer.TRACER_DSHAPE,
            storagegroup=storagegroup,
            chunks=256,
        )
        self.table.add(TableTracer.TRACER_INIT)
        self.step_count = 0
        self.last_run_step_start: Optional[Dict[str, Union[int, float, str]]] = None
        self.last_run_step_details = ""
        self.last_run_start: Optional[
            Dict[str, Union[int, float, np.floating, str]]
        ] = None
        self.last_run_details = ""

    def trace_stats(self, max_runs: Optional[int] = None) -> Table:
        return self.table

    def start_run(self, ts: float, run_number: int, **kwds: Any) -> None:
        self.last_run_start = dict(TableTracer.TRACER_INIT)  # type: ignore
        self.last_run_start["start"] = ts
        self.last_run_start["run"] = run_number
        self.step_count = 0

    def end_run(self, ts: float, run_number: int, **kwds: Any) -> None:
        if self.last_run_start is None:
            return
        row = self.last_run_start
        row["end"] = ts
        row["duration"] = ts - cast(float, row["start"])
        row["detail"] = self.last_run_details if self.last_run_details else ""
        row["steps"] = self.step_count
        #        row['loadavg'] = os.getloadavg()[0]
        row["type"] = "run"
        row["progress_current"] = kwds.get("progress_current", 0.0)
        row["progress_max"] = kwds.get("progress_max", 0.0)
        row["quality"] = kwds.get("quality", 0.0)
        self.table.add(row)
        self.last_run_details = ""
        self.last_run_start = None

    def run_stopped(self, ts: float, run_number: int, **kwds: Any) -> None:
        self.last_run_details += "stopped"

    def before_run_step(self, ts: float, run_number: int, **kwds: Any) -> None:
        self.last_run_step_start = {
            "start": ts,
            "run": run_number,
            "steps": self.step_count,
        }

    def after_run_step(self, ts: float, run_number: int, **kwds: Any) -> None:
        assert self.last_run_step_start
        row = self.last_run_step_start
        for (name, dflt) in TableTracer.TRACER_INIT.items():
            if name not in row:
                row[name] = kwds.get(name, dflt)
        row["end"] = ts
        row["duration"] = ts - cast(float, row["start"])
        row["detail"] = self.last_run_step_details
        assert self.last_run_start
        last_run_start = self.last_run_start
        last_run_start["steps_run"] = cast(int, last_run_start["steps_run"]) + cast(
            int, row["steps_run"]
        )
        if "debug" in kwds:
            row["type"] = "debug_step"
        else:
            row["type"] = "step"
        self.table.add(row)
        self.step_count += 1
        self.last_run_details = ""
        self.last_run_step_start = None

    def exception(self, ts: float, run_number: int, **kwds: Any) -> None:
        self.last_run_details += "exception"

    def terminated(self, ts: float, run_number: int, **kwds):
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


Tracer.default = TableTracer
