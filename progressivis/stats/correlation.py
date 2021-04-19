import numpy as np
from itertools import product
import pandas as pd

from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.table import Table
from ..table.module import TableModule
from ..utils.psdict import PsDict
from . import Var
from ..core.decorators import process_slot, run_if_any


class OnlineCovariance(object):
    def __init__(self, ddof=1):
        self.reset()
        self.ddof = ddof


    def reset(self):
        self.n = 0
        self.mean_x = 0
        self.sum_x = 0
        self.mean_y = 0
        self.sum_y = 0
        self.cm = 0

    def include(self, x, y):
        self.n += 1
        dx = x - self.mean_x
        self.sum_x += x
        self.sum_y += y        
        self.mean_x = self.sum_x / self.n
        self.mean_y = self.sum_y / self.n
        self.cm += dx * (y - self.mean_y)

    def add(self, array_x, array_y):
            for x, y in zip(array_x, array_y):
                self.include(x, y)

    @property
    def cov(self):
        div_ = self.n - self.ddof
        return self.cm/div_ if div_ else np.nan
"""
def _prettify(dict_in):
    res = {}
    for k, v in dict_in:
        lk = list(k)
        if len(lk) == 1:
            res[f"{lk[0]}__{lk[0]}"] = v
            continue
        res[f"{lk[0]}__{lk[1]}"] = v
        res[f"{lk[1]}__{lk[0]}"] = v                
"""

class Cov(TableModule):
    """
    Compute the covariance matrix (a dict, actually) of the columns of an input table.
    """
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._data = {}
        self.default_step_size = 1000
        
    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super().is_ready()

    def op(self, chunk):
        cols = chunk.columns
        ret = {}
        done_ = set()
        for cx, cy in product(cols, cols):
            key = frozenset([cx, cy])
            if key in done_:
                continue
            data = self._data.get(key)
            if data is None:
                data = OnlineCovariance()
                self._data[key] = data
            data.add(chunk[cx], chunk[cy])
            done_.add(key)
            ret[key] = data.cov
        return ret

    def reset(self):
        if self.result is None:
            self.result.resize(0)
        if self._data is not None:
            for oc in self._data.values():
                oc.reset()
            

    def result_as_df(self, columns):
        """
        Convenience method
        """
        res = pd.DataFrame(index=columns, columns=columns, dtype='float64')
        for kx, ky in product(columns, columns):
            res.loc[kx, ky] = self.result[frozenset([kx, ky])]
        return res
            

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            cov_ = self.op(self.filter_columns(input_df, fix_loc(indices)))
            if self.result is None:
                self.result = PsDict(other=cov_)
            else:
                self.result.update(cov_)
            return self._return_run_step(self.next_state(dfslot), steps)


class Corr(Cov):
    """
    Compute the correlation matrix (a dict, actually) of the columns of an input table.
    """
    inputs = [SlotDescriptor('var', type=Table, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)

    @process_slot("table", reset_cb="reset")
    @process_slot("var")    
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            varslot = ctx.var
            var_data = varslot.data()
            if var_data is None:
                return self._return_run_step(self.state_blocked, steps_run=0)
            varslot.clear_buffers()
            indices = dfslot.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            cov_ = self.op(self.filter_columns(input_df, fix_loc(indices)))
            std_ = {k: np.sqrt(v) for (k, v) in var_data.items()}
            corr_ = {}
            for k, v in cov_.items():
                lk = list(k)
                if len(lk) == 1:
                    kx = ky = lk[0]
                else:
                    kx = lk[0]
                    ky = lk[1]
                corr_[k] = v/(std_[kx]*std_[ky])
            if self.result is None:
                self.result = PsDict(other=corr_)
            else:
                self.result.update(corr_)
            return self._return_run_step(self.next_state(dfslot), steps)

    def create_dependent_modules(self, input_module, input_slot='result'):
        s = self.scheduler()
        self.input.table = input_module.output[input_slot]
        self.var = Var(scheduler=s)
        self.var.input.table = input_module.output[input_slot]
        self.input.var = self.var.output.result
