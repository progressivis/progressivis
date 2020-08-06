import numpy as np
from ..core.utils import indices_len, fix_loc
from ..table.module import TableModule
from ..table.table import Table
from ..core.decorators import *
from progressivis import ProgressiveError, SlotDescriptor


class Unary(TableModule):
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, ufunc, columns=None, **kwds):
        super().__init__(**kwds)
        self._ufunc = ufunc
        self._columns = columns
        #import pdb;pdb.set_trace()
        self._kwds = {} #self._filter_kwds(kwds, ufunc)

    def reset(self):
        self._table.resize(0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            print("run_step", step_size)
            data_in = ctx.table.data()
            if self._table is None:
                self._table = Table(self.generate_table_name('elementwise1'),
                                    dshape=data_in.dshape, create=True)
            cols = self.get_columns(data_in)
            if len(cols) == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            indices = ctx.table.created.next(step_size)
            steps = indices_len(indices)
            vec = self.filter_columns(data_in, fix_loc(indices)).raw_unary(self._ufunc, **self._kwds)
            self._table.append(vec)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

def make_unary(cname, ufunc):
    def _init_func(self_, *args, **kwds):
        Unary.__init__(self_, ufunc, *args, **kwds)
    cls = type(cname, (Unary,), {})
    cls.__init__ = _init_func
    return cls

Log = make_unary("Log", np.log)
