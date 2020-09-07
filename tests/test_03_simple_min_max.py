from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.stats import Min, RandomTable
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.slot import SlotDescriptor
from progressivis.core.decorators import *
from progressivis.core.utils import indices_len, fix_loc
from progressivis.utils.psdict import PsDict
import numpy as np


class Max(TableModule):
    """
    Simplified Max, adapted for documentation
    """
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super().is_ready()

    def reset(self):
        if self._table is not None:
            self._table.fill(-np.inf)

    def run_step(self, run_number, step_size, howlong):
        slot = self.get_input_slot('table')
        if slot.updated.any() or slot.deleted.any():
            slot.reset()
            if self._table is not None:
                self._table.resize(0)
            slot.update(run_number)
        indices = slot.created.next(step_size) # /!\ 
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        data = slot.data()
        import pdb;pdb.set_trace()
        op = data.loc[fix_loc(indices)].max(keepdims=False)
        if self._table is None:
            self._table = PsDict(op)
        else:
            for k, v in self._table.items():
                self._table[k] = np.maximum(op[k], v)
        return self._return_run_step(self.next_state(slot), steps_run=steps)


class MaxDec(TableModule):
    """
    Simplified Max, usefull for documentation
    """
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super().is_ready()

    def reset(self):
        if self._table is not None:
            self._table.fill(-np.inf)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            indices = ctx.table.created.next(step_size) # returns a slice
            steps = indices_len(indices)
            input_df = ctx.table.data()
            op = input_df.loc[fix_loc(indices)].max(keepdims=False)
            if self._table is None:
                self._table = PsDict(op)
            else:
                for k, v in self._table.items():
                    self._table[k] = np.maximum(op[k], v)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

class TestMinMax(ProgressiveTest):
    def te_st_min(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        min_=Min(name='min_'+str(hash(random)), scheduler=s)
        min_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = min_.output.table
        aio.run(s.start())
        #s.join()
        res1 = random.table().min()
        res2 = min_.table()
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1)
        #print('v2 = ', v2)
        self.assertTrue(np.allclose(v1, v2))

    def test_max(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        max_=Max(name='max_'+str(hash(random)), scheduler=s)
        max_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        #s.join()
        res1 = random.table().max()
        res2 = max_.table()
        self.compare(res1, res2)


if __name__ == '__main__':
    ProgressiveTest.main()
