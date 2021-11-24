from ..core.utils import indices_len, fix_loc
from ..core.bitmap import bitmap
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any
from datasketches import kll_floats_sketch, kll_ints_sketch
import numpy as np

import logging
logger = logging.getLogger(__name__)


class KLLSketch(TableModule):
    parameters = [('bins', np.dtype(int), 128),
                  ('quantile', np.dtype(float), 0.5)]
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, column, k=200, **kwds):
        super().__init__(**kwds)
        self.column = column
        self._k = k
        self._kll = None
        self.default_step_size = 10000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super().is_ready()

    def reset(self):
        if self.result is not None:
            self.result.clear()
        self._kll = None

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            indices = ctx.table.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if not steps:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = ctx.table.data()
            column = input_df[self.column]
            dtype_ = column.dtype
            column = column.loc[fix_loc(indices)]
            if self._kll is None:
                if np.issubdtype(dtype_, np.integer):
                    self._kll_func = kll_ints_sketch
                elif np.issubdtype(dtype_, np.floating):
                    self._kll_func = kll_floats_sketch
                
                else:
                    raise ProgressiveError(f"Type {dtype_} of {self.column} "
                                           "is invalid for sketching")
                self._kll = self._kll_func(self._k)
            sk = self._kll_func(self._k)
            sk.update(column)
            self._kll.merge(sk)
            kll = self._kll
            max_ = kll.get_max_value()
            min_ = kll.get_min_value()
            quantile = kll.get_quantile(self.params.quantile)
            num_splits = self.params.bins
            step = (max_ - min_) / num_splits
            splits = [min_ + (i*step) for i in range(0, num_splits)]
            pmf = kll.get_pmf(splits)
            splits.append(max_)
            res = dict(max=max_, min=min_, quantile=quantile, splits=splits, pmf=pmf)
            if self.result is None:
                self.result = PsDict(res) 
            else:
                self.result.update(res)                 
            return self._return_run_step(self.next_state(ctx.table), steps)
