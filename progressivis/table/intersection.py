"""
Range Query module.


"""
from progressivis.core.utils import indices_len
from ..io import Variable
from ..stats import Min, Max
from .hist_index import HistogramIndex
from .bisectmod import Bisect, _get_physical_table
from .module import TableModule
from ..core.slot import SlotDescriptor
from . import Table
from . import TableSelectedView
from ..core.bitmap import bitmap
from progressivis.table.nary import NAry
from progressivis.core.synchronized import synchronized
#from collections import defaultdict
from functools import reduce
import operator
class Intersection(NAry):
    "Intersection Module"
    parameters = []

    def __init__(self, scheduler=None, **kwds):        
        super(Intersection, self).__init__(scheduler=scheduler, **kwds)

    def predict_step_size(self, duration):
        return 1000
    @synchronized
    def run_step(self, run_number, step_size, howlong):
        _b = bitmap.asbitmap
        to_delete = bitmap([])
        to_create_maybe = bitmap()
        to_create_4sure = bitmap()
        steps = 0
        tables = []
        ph_table = None
        assert len(self.inputs) > 0
        for name in self.inputs:
            if not name.startswith('table'):
                continue
            slot = self.get_input_slot(name)
            t = slot.data()
            assert isinstance(t, TableSelectedView)
            if ph_table is None:
                ph_table = _get_physical_table(t)
            else:
                assert ph_table is _get_physical_table(t)
            tables.append(t)
            slot.update(run_number)
            if slot.deleted.any():
                deleted = slot.deleted.next(step_size)
                steps += 1
                to_delete |= _b(deleted)
            if slot.updated.any(): # actually don't care
                _ = slot.updated.next(step_size)
                #to_delete |= _b(updated)
                #to_create |= _b(updated)
                #steps += 1 # indices_len(updated) + 1
            if slot.created.any():
                created = slot.created.next(step_size)
                bm = _b(created)
                to_create_maybe |= bm
                to_create_4sure &= bm
                steps += indices_len(created)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not self._table:
            self._table = TableSelectedView(ph_table, bitmap([]))
        self._table.selection -= to_delete
        self._table.selection |= to_create_4sure
        to_create_maybe -= to_create_4sure
        eff_create = to_create_maybe
        for t in tables:
            eff_create &= t.selection
        self._table.selection |= eff_create
        return self._return_run_step(self.next_state(self.get_input_slot(self.inputs[0])), steps_run=steps)
