from ..core.utils import indices_len, fix_loc
from ..core.bitmap import bitmap
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any
from datasketches import kll_floats_sketch, kll_ints_sketch
from ..core.utils import integer_types
from collections.abc import Sequence
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KLLSketch(TableModule):
    parameters = [('binning', object, []),
                  ('quantiles', object, [])]
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
                self._kll_func = kll_floats_sketch
                """
                if np.issubdtype(dtype_, np.integer):
                    self._kll_func = kll_ints_sketch
                elif np.issubdtype(dtype_, np.floating):
                    self._kll_func = kll_floats_sketch
                
                else:
                    raise ProgressiveError(f"Type {dtype_} of {self.column} "
                                           "is invalid for sketching")
                """
                self._kll = self._kll_func(self._k)
            kll = self._kll                
            sk = self._kll_func(self._k)
            sk.update(column)
            kll.merge(sk)
            max_ = kll.get_max_value()
            min_ = kll.get_min_value()
            quantiles = splits = pmf = []
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
                    lower_ = par_bin['lower']
                    upper_ = par_bin['upper']
                    num_splits = par_bin['n_splits']
                    splits = np.linspace(lower_, upper_, num_splits)
                    pmf = kll.get_pmf(splits[:-1])
            res = dict(max=max_, min=min_, quantiles=quantiles, splits=splits, pmf=pmf)
            if self.result is None:
                self.result = PsDict(res) 
            else:
                self.result.update(res)                 
            return self._return_run_step(self.next_state(ctx.table), steps)
