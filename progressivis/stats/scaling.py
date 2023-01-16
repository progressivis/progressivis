from __future__ import annotations
import numpy as np
import numexpr as ne
import logging
from ..core import Sink
from ..core.pintset import PIntSet
from ..core.utils import indices_len
from ..core.slot import SlotDescriptor, Slot
from ..table import PTable, BasePTable, BasePColumn
from ..table.module import PTableModule
from ..utils.psdict import PDict
from . import Min, Max, Histogram1D
from ..core.decorators import process_slot, run_if_any
from ..table.dshape import dshape_all_dtype

from typing import cast, Optional, Union, Dict, Tuple, Any, List, TYPE_CHECKING


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.module import ReturnRunStep

    ColsTo = Dict[str, Tuple[float, float]]
    NumCol = Union[np.ndarray[Any, Any], BasePColumn]


class MinMaxScaler(PTableModule):
    """
    Scaler
    """

    parameters = [
        ("delta", np.dtype(float), -5),
        ("ignore_max", np.dtype(int), 0),
    ]  # 5%, 0

    inputs = [
        SlotDescriptor("table", type=PTable, required=True),
        SlotDescriptor("min", type=PDict, required=True),
        SlotDescriptor("max", type=PDict, required=True),
        SlotDescriptor("control", type=PDict, required=False),
    ]
    outputs = [SlotDescriptor("info", type=PDict, required=False)]

    def __init__(
        self,
        usecols: Optional[List[str]] = None,
        reset_threshold: int = 1000,
        **kwds: Any,
    ):
        """
        usecols: scaled cols. all the other cols are transfered as is
        reset_threshold: when n_rows < reset_threshold min/max changes trigger
                         a reset without any warning
        """
        super().__init__(**kwds)
        self._usecols = usecols
        self._reset_threshold = reset_threshold
        self._cmin: Dict[str, float] = {}  # current min
        self._cmax: Dict[str, float] = {}  # current max
        self._has_cmin_cmax = False
        # self._delta = None
        self._control_data: Optional[Dict[str, int]] = None
        self._clipped = 0
        self._ignored = 0
        self._info: Dict[str, int] = {}

    def reset(self) -> None:
        if self.result is not None:
            self.table.truncate()
        self._cmin.clear()
        self._cmax.clear()
        self._has_cmin_cmax = False
        self._control_data = None
        self._clipped = 0
        self._ignored = 0
        if self._info:
            self._info.update(
                {"clipped": 0, "ignored": 0, "has_buffered": 0, "last_reset": 0}
            )

    def scale(
        self, chunk: BasePTable, cols: List[str], usecols: List[str], clip_cols: ColsTo
    ) -> Dict[str, NumCol]:
        res: Dict[str, NumCol] = {}
        for c in cols:
            arr: Union[np.ndarray[Any, Any], BasePColumn] = chunk[c]  # .to_array()
            if c not in usecols:
                res[c] = arr
                continue
            min_ = self._cmin[c]
            max_ = self._cmax[c]
            width = max_ - min_
            _ = width  # for flake8
            val: np.ndarray[Any, Any] = ne.evaluate("(arr-min_)/width")
            res[c] = val
            if c in clip_cols:
                if self._info:
                    arr = res[c]  # helping the evaluator
                    self._info["clipped"] += np.sum(
                        ne.evaluate("(arr<0.0) | (arr>1.0)")
                    )
                np.clip(val, 0.0, 1.0, out=val)
        return res

    def get_ignore(self, chunk: BasePTable, oversized_cols: ColsTo) -> PIntSet:
        ignore = PIntSet()
        for c, (min_, max_) in oversized_cols.items():
            arr = chunk[c]  # .to_array()
            _ = arr  # for flake8
            ignore.update(np.where(ne.evaluate(f"(arr<{min_}) | (arr>{max_})"))[0])
        return ignore

    def get_ignore_credit(self) -> int:
        return (
            int(self.params["ignore_max"])
            if self._control_data is None
            else int(self._control_data["ignore_max"])
        ) - self._ignored

    def get_delta(
        self, usecols: List[str], min_: Dict[str, float], max_: Dict[str, float]
    ) -> Dict[str, float]:
        delta: float = self.params["delta"]
        if self._control_data is not None:
            delta = self._control_data["delta"]
        res: Dict[str, float] = {}
        if delta < 0:
            for c in usecols:
                extent = max_[c] - min_[c]
                res[c] = extent * delta / -100.0
        else:
            for c in usecols:
                res[c] = delta
        return res

    def starting(self) -> None:
        super().starting()
        opt_slot = self.get_output_slot("info")
        if opt_slot:
            logger.debug("Maintaining info")
            self.maintain_info(True)
        else:
            logger.debug("Not maintaining info")
            self.maintain_info(False)

    def maintain_info(self, yes: bool = True) -> None:
        if yes and not self._info:
            self._info = PDict({"clipped": 0, "ignored": 0, "needs_changes": False})
        elif not yes:
            self._info = {}

    def info(self) -> Dict[str, int]:
        return self._info

    def get_data(self, name: str) -> Any:
        if name == "info":
            return self.info()
        return super().get_data(name)

    def check_bounds(
        self,
        min_data: Dict[str, float],
        max_data: Dict[str, float],
        usecols: List[str],
        to_clip: Dict[str, Tuple[float, float]],
        to_ignore: Dict[str, Tuple[float, float]],
    ) -> bool:
        delta = self.get_delta(usecols, min_data, max_data)
        for c in usecols:
            if min_data[c] >= self._cmin[c] and max_data[c] <= self._cmax[c]:
                continue
            lax_min = self._cmin[c] - delta[c]
            lax_max = self._cmax[c] + delta[c]
            if min_data[c] < lax_min or max_data[c] > lax_max:
                to_ignore[c] = (lax_min, lax_max)
                # continue
            to_clip[c] = (lax_min, lax_max)  # actually a simple set should suffice
        return not to_clip and not to_ignore

    def update_bounds(
        self, min_data: Dict[str, float], max_data: Dict[str, float]
    ) -> None:
        self._cmin.update(min_data)
        self._cmax.update(max_data)
        self._has_cmin_cmax = True

    def reset_min_max(
        self, dfslot: Slot, min_slot: Slot, max_slot: Slot, run_number: int
    ) -> None:
        dfslot.reset()
        dfslot.update(run_number)
        min_slot.reset()
        min_slot.update(run_number)
        max_slot.reset()
        max_slot.update(run_number)
        self.reset()  # reset all but keep the recent bounds
        self.update_bounds(min_slot.data(), max_slot.data())

    @process_slot("table", reset_cb="reset")
    @process_slot("min")
    @process_slot("max")
    # @process_slot("control")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            input_df = dfslot.data()
            min_slot = ctx.min
            min_data = min_slot.data()
            max_slot = ctx.max
            max_data = max_slot.data()
            cols_to_clip: ColsTo = {}
            cols_to_ignore: ColsTo = {}
            if not (input_df and min_data and max_data):
                return self._return_run_step(self.state_blocked, steps_run=0)
            cols = self._columns or input_df.columns
            usecols: List[str] = self._usecols or cols
            # self._delta = self.get_delta(usecols, min_data, max_data)
            control_slot = self.get_input_slot("control")
            if control_slot is not None:
                self._control_data = control_slot.data()
                if self._control_data is None:
                    return self._return_run_step(self.state_blocked, steps_run=0)
                if control_slot.has_buffered():
                    control_slot.clear_buffers()
                    if self._info and self._info.get("needs_changes"):
                        self._info["needs_changes"] = False
                    if self._control_data.get("reset"):
                        self.reset_min_max(dfslot, min_slot, max_slot, run_number)
                        self._info["last_reset"] = run_number
                    # else:
                    #    self._info["last_reset"] = self._control_data.get("reset")
                    self._info["has_buffered"] = run_number
                else:
                    if self._info and self._info.get("needs_changes"):
                        return self._return_run_step(self.state_blocked, steps_run=0)
            if min_slot.has_buffered() or max_slot.has_buffered():
                min_slot.clear_buffers()
                max_slot.clear_buffers()
                if self._has_cmin_cmax:
                    if not self.check_bounds(
                        min_data, max_data, usecols, cols_to_clip, cols_to_ignore
                    ):
                        if (
                            self.result is None
                            or len(self.result) <= self._reset_threshold
                        ):
                            # there is not much processed data => resetting
                            # all without checking tol, etc.
                            self.reset_min_max(dfslot, min_slot, max_slot, run_number)
                            # we will process data from the beginning
                            # but with the most recent knowledge about min/max
                            return self._return_run_step(
                                self.state_blocked, steps_run=0
                            )
                else:  # at start or after data reset
                    self.update_bounds(min_data, max_data)

            indices = dfslot.created.next(step_size, as_slice=False)  # returns a PIntSet
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            tbl: BasePTable = self.filter_columns(input_df, indices)
            ignore_ilocs = self.get_ignore(tbl, cols_to_ignore)
            if ignore_ilocs:
                len_ii = len(ignore_ilocs)
                if len_ii > self.get_ignore_credit():
                    if self._info:
                        self._info["needs_changes"] = True
                        return self._return_run_step(
                            self.next_state(dfslot), steps // 2
                        )
                    else:
                        self.reset_min_max(dfslot, min_slot, max_slot, run_number)
                        return self._return_run_step(self.state_blocked, steps_run=0)
                self._ignored += len_ii
                if self._info:
                    self._info["ignored"] += len_ii
                rm_ids = PIntSet(np.array(indices)[cast(List[int], ignore_ilocs)])
                indices = indices - rm_ids
                if not indices:
                    return self._return_run_step(self.state_blocked, steps_run=0)
                tbl_ii: Optional[BasePTable] = tbl.loc[indices, :]
                assert tbl_ii
                tbl = tbl_ii
            sc_data = self.scale(tbl, cols, usecols, cols_to_clip)
            if self.result is None:
                ds = dshape_all_dtype(usecols, np.dtype("float64"))
                self.result = PTable(
                    self.generate_table_name("scaled"),
                    dshape=ds,  # input_df.dshape,
                    create=True,
                )
            self.table.append(sc_data, indices=indices)
            return self._return_run_step(self.next_state(dfslot), steps)

    def create_dependent_modules(
        self, input_module: PTableModule, input_slot: str = "result", hist: bool = False
    ) -> None:
        s = self.scheduler()
        self.input.table = input_module.output[input_slot]
        self.min = Min(scheduler=s)
        self.min.input.table = input_module.output[input_slot]
        self.max = Max(scheduler=s)
        self.max.input.table = input_module.output[input_slot]
        self.input.min = self.min.output.result
        self.input.max = self.max.output.result
        self.hist: Dict[str, Histogram1D] = {}
        if hist:
            assert self._usecols  # TODO: avoid this requirement
            for col in self._usecols:
                hist1d = Histogram1D(scheduler=s, column=col)
                hist1d.input.table = input_module.output[input_slot]
                hist1d.input.min = self.min.output.result
                hist1d.input.max = self.max.output.result
                sink = Sink(scheduler=s)
                sink.input.inp = hist1d.output.result
                self.hist[col] = hist1d
