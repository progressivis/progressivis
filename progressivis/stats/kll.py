from __future__ import annotations

from ..core.utils import indices_len, fix_loc
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any
from datasketches import kll_floats_sketch  # type: ignore
from ..core.utils import integer_types
from ..core.types import Floats
from collections.abc import Sequence
import numpy as np
import logging
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.module import Parameters, ReturnRunStep


class KLLSketch(TableModule):
    parameters: Parameters = [
        ("binning", np.dtype(object), []),
        ("quantiles", np.dtype(object), []),
    ]
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, column: str, k: int = 200, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.column: str = column
        self._k: int = k
        self._kll: kll_floats_sketch = kll_floats_sketch(k)
        self.default_step_size: int = 10000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.psdict.clear()
        self._kll = kll_floats_sketch(self._k)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            indices = ctx.table.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if not steps:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = ctx.table.data()
            column = input_df[self.column]
            column = column.loc[fix_loc(indices)]
            # if self._kll is None:
            #    self._kll_func = kll_floats_sketch
            #    self._kll = self._kll_func(self._k)
            kll = self._kll
            sk = kll_floats_sketch(self._k)  # self._kll_func(self._k)
            sk.update(column)
            assert kll
            kll.merge(sk)
            max_ = kll.get_max_value()
            min_ = kll.get_min_value()
            quantiles: Floats = []
            splits: Floats = []
            pmf: Floats = []
            if self.params.quantiles:
                quantiles = kll.get_quantiles(self.params.quantiles)
            if self.params.binning:
                par_bin = self.params.binning
                if isinstance(par_bin, integer_types):
                    num_splits = par_bin
                    splits = np.linspace(min_, max_, num_splits)
                    pmf = kll.get_pmf(splits[:-1])
                elif isinstance(par_bin, Sequence):
                    splits = par_bin
                    pmf = kll.get_pmf(splits)
                elif isinstance(par_bin, dict):
                    lower_ = par_bin["lower"]
                    upper_ = par_bin["upper"]
                    num_splits = par_bin["n_splits"]
                    splits = np.linspace(lower_, upper_, num_splits)
                    pmf = kll.get_pmf(splits[:-1])
            res = dict(max=max_, min=min_, quantiles=quantiles, splits=splits, pmf=pmf)
            if self.result is None:
                self.result = PsDict(res)
            else:
                self.psdict.update(res)
            return self._return_run_step(self.next_state(ctx.table), steps)
