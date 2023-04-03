import logging
from collections.abc import Iterable

from ..core import Print, Sink
from ..core.pintset import PIntSet
from ..core.utils import indices_len
from ..table.table import PTable
from ..core.module import Module, def_input, def_output, ReturnRunStep
from ..core.scheduler import Scheduler
from ..table.range_query import RangeQuery
from ..table.hist_index import HistogramIndex
from ..io import DynVar
from ..utils.psdict import PDict

# from .var import OnlineVariance
from ..stats import Min, Max, Var, Distinct, Corr
from ..stats.kll import KLLSketch

# from ..table.hub import Hub
from ..stats import Histogram1D
from ..stats.histogram1d_categorical import Histogram1DCategorical
from ..core.decorators import process_slot, run_if_any
from ..table import PTableSelectedView
from ..table.dshape import dshape_fields, DataShape
from typing import Optional, List, Tuple, Any, Callable, Union, Dict

logger = logging.getLogger(__name__)


def _is_string_col(table_: PTable, col: str) -> bool:
    col_type = dict(dshape_fields(table_.dshape))[col]
    return str(col_type) == "string"


IfModule = Union[
    "KLLSketchIf",
    "Histogram1DCategoricalIf",
    "Histogram1DIf",
    "HistogramIndexIf",
    "RangeQueryIf",
]


def _run_step_common(
    self_: IfModule,
    super_call: Callable[..., Dict[str, int]],
    run_number: int,
    step_size: int,
    howlong: float,
    is_string: bool,
) -> Dict[str, int]:
    if self_._enabled:
        return super_call(run_number, step_size, howlong)
    slot = self_.get_input_slot("table")
    input_df = slot.data()
    if self_._enabled is None:
        if input_df is None:
            return self_._return_run_step(self_.state_blocked, steps_run=0)
        assert self_._columns is not None
        assert isinstance(self_._columns[0], str)
        self_._enabled = _is_string_col(input_df, self_._columns[0]) is is_string
    if self_._enabled:
        return super_call(run_number, step_size, howlong)
    slot.clear_buffers()
    return self_._return_terminate()


class KLLSketchIf(KLLSketch):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._enabled: Optional[bool] = None

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return _run_step_common(
            self, super().run_step, run_number, step_size, howlong, is_string=False
        )


class Histogram1DCategoricalIf(Histogram1DCategorical):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._enabled = None

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return _run_step_common(
            self, super().run_step, run_number, step_size, howlong, is_string=True
        )


class Histogram1DIf(Histogram1D):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._enabled = None

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return _run_step_common(
            self, super().run_step, run_number, step_size, howlong, is_string=False
        )


class HistogramIndexIf(HistogramIndex):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._enabled = None

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return _run_step_common(
            self, super().run_step, run_number, step_size, howlong, is_string=False
        )


class RangeQueryIf(RangeQuery):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._enabled = None
        self._flag = False

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return _run_step_common(
            self, super().run_step, run_number, step_size, howlong, is_string=False
        )


def make_sketch_barplot(
    input_module: Module, col: int, scheduler: Scheduler, input_slot: str = "result"
) -> Tuple[KLLSketch, Histogram1DCategorical]:
    s = scheduler
    sketch = KLLSketchIf(scheduler=s, column=col)
    sketch.params.binning = 128
    sketch.input.table = input_module.output[input_slot]
    sink = Sink(scheduler=s)
    sink.input.inp = sketch.output.result
    barplot = Histogram1DCategoricalIf(scheduler=s, column=col)
    barplot.input.table = input_module.output[input_slot]
    sink = Sink(scheduler=s)
    sink.input.inp = barplot.output.result
    # hub = Hub(scheduler=s)
    # hub.input.table = hist1d_num.output.result
    # hub.input.table = hist1d_str.output.result
    # hub.column = col
    # return hub
    return sketch, barplot


@def_input("table", PTable)
@def_input("min", PDict, required=False)
@def_input("max", PDict, required=False)
@def_input("var", PDict, required=False)
@def_input("distinct", PDict, required=False)
@def_input("corr", PDict, required=False)
@def_output("result", PTableSelectedView)
@def_output("dshape", PDict, required=False)
class StatsExtender(Module):
    """
    Adds statistics on input data
    """

    def __init__(self, usecols: Optional[List[str]] = None, **kwds: Any) -> None:
        """
        usecols: these columns are transmitted to stats modules by default
        """
        super().__init__(**kwds)
        self._usecols = usecols
        self.visible_cols: List[str] = []
        self.decorations: List[str] = []
        self._raw_dshape: Optional[DataShape] = None
        self._dshape_flag = False

    def reset(self) -> None:
        if self.result is not None:
            self.result.selection = PIntSet()

    def starting(self) -> None:
        super().starting()
        ds_slot = self.get_output_slot("dshape")
        if ds_slot:
            logger.debug("Maintaining dshape")
            self._dshape_flag = True
        else:
            self._dshape_flag = False

    def maintain_dshape(self, data: PTable) -> None:
        if not self._dshape_flag or self._raw_dshape == data.dshape:
            return
        self._raw_dshape = data.dshape
        if self.dshape is None:
            self.dshape = PDict()
        self.dshape.update(dshape_fields(data.dshape))

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            input_df = dfslot.data()
            if not input_df:
                return self._return_run_step(self.state_blocked, steps_run=0)
            cols = self._columns or input_df.columns
            usecols = self._usecols or cols
            self.visible_cols = usecols
            for name_ in self.input_slot_names():
                if name_ in ("table", "_params"):
                    continue
                dec_slot = self.get_input_slot(name_)
                if dec_slot and dec_slot.has_buffered():
                    dec_slot.clear_buffers()
            indices = dfslot.created.next(step_size, as_slice=False)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            self.maintain_dshape(input_df)
            if self.result is None:
                self.result = PTableSelectedView(input_df, PIntSet([]))
            self.result.selection |= indices
            return self._return_run_step(self.next_state(dfslot), steps)

    def _get_usecols(self, x: Any) -> Optional[List[Any]]:
        return list(x) if isinstance(x, Iterable) else self._usecols

    def _get_usecols_hist(self, x: Any, hist: Any) -> Optional[List[Any]]:
        if not hist:
            return self._get_usecols(x)
        if not x:
            return self._get_usecols(hist)
        if isinstance(x, Iterable) and isinstance(hist, Iterable):
            return list(set(list(x) + list(hist)))
        return self._usecols

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str = "result",
        min_: bool = False,
        max_: bool = False,
        hist: bool = False,
        var: bool = False,
        distinct: bool = False,
        corr: bool = False,
        dshape: bool = True,
    ) -> None:
        s = self.scheduler()
        self.input.table = input_module.output[input_slot]
        if min_ or hist:
            self.dep.min = Min(scheduler=s, columns=self._get_usecols_hist(min_, hist))
            self.dep.min.input.table = input_module.output[input_slot]
            self.input.min = self.dep.min.output.result
            self.decorations.append("min")
        if max_ or hist:
            self.dep.max = Max(scheduler=s, columns=self._get_usecols_hist(max_, hist))
            self.dep.max.input.table = input_module.output[input_slot]
            self.input.max = self.dep.max.output.result
            self.decorations.append("max")
        self.dep.hist = {}
        if hist:
            usecols = self._get_usecols(hist) or []
            for col in usecols:
                self.dep.hist[col] = {}
                h_col = self.dep.hist[col]  # shortcut
                # dyn variables
                h_col["lower"] = lower = DynVar({col: None}, scheduler=s)
                h_col["upper"] = upper = DynVar({col: None}, scheduler=s)
                # lower.column = col
                # upper.column = col
                h_col["range_query"] = range_query = RangeQueryIf(
                    scheduler=s, column=col, columns=[col]
                )
                range_query.params.column = col
                # print(range_query.scheduler(), self.scheduler(), lower.scheduler())
                # assert range_query.scheduler() == lower.scheduler()
                hist_index = HistogramIndexIf(scheduler=s, columns=[col])
                range_query.create_dependent_modules(
                    input_module,
                    input_slot,
                    min_=self.dep.min,
                    max_=self.dep.max,
                    min_value=lower,
                    max_value=upper,
                    hist_index=hist_index,
                )
                assert range_query.scheduler() == range_query.dep.min_value.scheduler()
                # histogram 1D
                h_col["hist1d"] = hist1d = Histogram1DIf(scheduler=s, column=col)
                hist1d.input.table = range_query.output.result
                hist1d.input.min = range_query.output.min
                hist1d.input.max = range_query.output.max
                # sketching + barplot
                sink = Sink(scheduler=s)
                sink.input.inp = hist1d.output.result
                h_col["sketching"], h_col["barplot"] = make_sketch_barplot(
                    input_module, col, s, input_slot="result"
                )
                # lower.prioritize = set([h_col["sketching"].name])
                # upper.prioritize = set([h_col["sketching"].name])
        if var:
            self.var = Var(
                scheduler=s, columns=self._get_usecols(var), ignore_string_cols=True
            )
            self.var.input.table = input_module.output[input_slot]
            self.input.var = self.var.output.result
            self.decorations.append("var")
        if distinct:
            self.distinct = Distinct(scheduler=s, columns=self._get_usecols(distinct))
            self.distinct.input.table = input_module.output[input_slot]
            self.input.distinct = self.distinct.output.result
            self.decorations.append("distinct")
        if corr:
            self.corr = Corr(
                scheduler=s, columns=self._get_usecols(corr), ignore_string_cols=True
            )
            self.corr.input.table = input_module.output[input_slot]
            self.input.corr = self.corr.output.result

        if dshape:
            pr = Print(proc=lambda x: None, scheduler=s)
            pr.input[0] = self.output.dshape
