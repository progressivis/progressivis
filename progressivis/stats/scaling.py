import numpy as np
from itertools import product
import pandas as pd
import numexpr as ne

from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.table import Table
from ..table.module import TableModule
from ..utils.psdict import PsDict
from .var import OnlineVariance
from . import Min, Max
from ..core.decorators import process_slot, run_if_any
from ..table.dshape import dshape_all_dtype

class MinMaxScaler(TableModule):
    """
    Scaler
    """
    inputs = [SlotDescriptor('table', type=Table, required=True),
              SlotDescriptor('min', type=Table, required=True),
              SlotDescriptor('max', type=Table, required=True),


    ]

    def __init__(self, usecols=None, **kwds):
        super().__init__(**kwds)
        self._usecols = usecols

    def reset(self):
        if self.result is None:
            self.result.truncate()

    def scale(self, chunk, cols, usecols, min_data, max_data):
        res = {}
        for c in cols:
            arr = chunk[c] #.to_array()
            if c not in usecols:
                res[c] = arr
                continue
            min_ = min_data[c]
            max_ = max_data[c]
            delta = max_ - min_
            res[c] = ne.evaluate('(arr-min_)/delta')
        return res
            
    @process_slot("table", reset_cb="reset")
    @process_slot("min", reset_cb="reset")
    @process_slot("max", reset_cb="reset")        
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            minslot = ctx.min
            min_data = minslot.data()
            maxslot = ctx.max
            max_data = maxslot.data()
            if min_data is None or max_data is None:
                return self._return_run_step(self.state_blocked, steps_run=0)
            minslot.clear_buffers()
            maxslot.clear_buffers()
            indices = dfslot.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            cols = self._columns or input_df.columns
            usecols = self._usecols or cols
            sc_data = self.scale(self.filter_columns(input_df, fix_loc(indices)), cols, usecols, min_data, max_data)
            if self.result is None:
                ds = dshape_all_dtype(input_df.columns, np.dtype("float64"))
                self.result = Table(self.generate_table_name('scaled'),
                                    dshape=ds,  # input_df.dshape,
                                    create=True)
            #import pdb;pdb.set_trace()
            self.result.append(sc_data, indices=indices)
            return self._return_run_step(self.next_state(dfslot), steps)

    def create_dependent_modules(self, input_module, input_slot='result'):
        s = self.scheduler()
        self.input.table = input_module.output[input_slot]
        self.min = Min(scheduler=s)
        self.min.input.table = input_module.output[input_slot]
        self.max = Max(scheduler=s)
        self.max.input.table = input_module.output[input_slot]
        self.input.min = self.min.output.result
        self.input.max = self.max.output.result        
