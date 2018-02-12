from .nary import NAry
from . import Table
from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import TableModule
import numpy as np
from ..core.utils import Dialog, indices_len, fix_loc
from ..core.bitmap import bitmap
from .mod_impl import ModuleImpl
from .binop import ops
from progressivis.core.utils import (slice_to_arange, slice_to_bitmap,
                                     indices_len, fix_loc)
class _Selection(object):
    def __init__(self, values=None):
        self._values = bitmap([]) if values is None else values

    def update(self, values):
        self._values.update(values)
        
    def remove(self, values):
        self._values = self._values -bitmap(values)
        
    def assign(self, values):
        self._values = values
        
class BisectImpl(ModuleImpl):
    def __init__(self, column, op, hist_index):
        super(BisectImpl,self).__init__()
        self._table = None
        self._column = column
        self._op = op
        if isinstance(op, str):
            self._op = ops[op]
        elif op not in ops.values():
            raise ValueError("Invalid operator {}".format(op))
        self.has_cache = False
        self.bins = None
        self.e_min = None
        self.e_max = None
        self.boundaries = None
        self._hist_index = hist_index
        
    def _eval_to_ids(self, limit, input_ids):
        x = self._table.loc[fix_loc(input_ids), self._column][0].values
        mask_ = self._op(x, limit)
        arr = slice_to_arange(input_ids)
        return bitmap(arr[np.nonzero(mask_)]) # maybe fancy indexing ...


        
    def resume(self, limit, limit_changed, created=None, updated=None, deleted=None):
        if limit_changed:
            #return self.reconstruct_from_hist_cache(limit)
            new_sel = self._hist_index.query(self._op, limit)
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            res = self._eval_to_ids(limit, updated)
            self.result.add(res)
        if created:
            res = self._eval_to_ids(limit, created)
            self.result.update(res)
        if deleted:
            self.result.remove(deleted)
        
        
        
    def start(self, table, limit, limit_changed, created=None, updated=None, deleted=None):
        self._table = table
        self.result = _Selection()
        self.is_started = True
        return self.resume(limit, limit_changed, created, updated, deleted)


class Bisect(TableModule):
    """
    """
    parameters = [('column', str, "unknown"),
                      ('op', str, ">"),
                      ("limit_key", str, ""),
                      #('hist_index', object, None) # to improve ...
                      ] 
    def __init__(self, hist_index=None, scheduler=None, **kwds):
        """
        """
        self._add_slots(kwds,'input_descriptors',
                            [SlotDescriptor('table', type=Table, required=True),
                                 SlotDescriptor('limit', type=Table, required=False)])
        super(Bisect, self).__init__(scheduler=scheduler, **kwds)
        self._impl = BisectImpl(self.params.column,
                                          self.params.op, hist_index) 

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        input_slot.update(run_number, self.id)
        steps = 0
        deleted = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(step_size)
            steps += indices_len(deleted)
        created = None
        if input_slot.created.any():
            created = input_slot.created.next(step_size)
            steps += indices_len(created)
        updated = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(step_size)
            steps += indices_len(updated)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        with input_slot.lock:
            input_table = input_slot.data()
        p = self.params
        limit_slot = self.get_input_slot('limit')
        limit_slot.update(run_number, self.id)
        limit_changed = False
        if limit_slot.deleted.any():
            limit_slot.deleted.next()
        if limit_slot.updated.any():
            limit_slot.updated.next()
            limit_changed = True
        if limit_slot.created.any():
            limit_slot.created.next()
            limit_changed = True
        if p.limit_key:
            limit_value = limit_slot.data().last(lkey)
        else:
            limit_value = limit_slot.data().last()[0]
        if not self._impl.is_started:
            self._table = TableSelectedView(input_table, bitmap([]))
            status = self._impl.start(input_table, limit_value, limit_changed,
                                                 created=created,
                                                 updated=updated,
                                                 deleted=deleted)
            self._table.selection = self._impl.result._values
        else:
            status = self._impl.resume(limit_value, limit_changed, 
                                                created=created,
                                                updated=updated,
                                                deleted=deleted)
            self._table.selection = self._impl.result._values            
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)
        
