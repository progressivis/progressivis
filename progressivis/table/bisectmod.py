from .nary import NAry
from . import Table
from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import TableModule
import numpy as np
from ..core.utils import Dialog, indices_len, fix_loc
from ..core.bitmap import bitmap
from .mod_impl import ModuleImpl

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
        self._prev_limit = None
        self._column = column
        self._op = op
        if isinstance(op, str):
            self._op = op_dict[op]
        elif op not in op_dict.values():
            raise ValueError("Invalid operator {}".format(op))
        self.has_cache = False
        self.bins = None
        self.e_min = None
        self.e_max = None
        self.boundaries = None
        self._hist_index = hist_index
        
    def _eval_to_ids(self, limit, input_ids):
        x = self._table.loc[_fix(input_ids), self._column][0].values
        mask_ = self._op(x, limit)
        arr = slice_to_arange(input_ids)
        return bitmap(arr[np.nonzero(mask_)]) # maybe fancy indexing ...


        
    def resume(self, limit, created=None, updated=None, deleted=None):
        def limit_changed():
            return  self._prev_limit is not None and limit != self._prev_limit
        if limit_changed():
            #return self.reconstruct_from_hist_cache(limit)
            new_sel = self._hist_index.query(self._op, limit)
            self.result.assign(new_sel)
        if updated:
            self.result.remove(updated)
            res = self._eval_to_ids(limit, updated)
            self.result.add(res)
        if created:
            res = self._eval_to_ids(limit, created)
            self.result.update(res)
        if deleted:
            self.result.remove(deleted)
        
        
        
    def start(self, table, limit, created=None, updated=None, deleted=None):
        self._table = table
        self.result = _Selection()
        self._prev_limit = None
        self.is_started = True
        return self.resume(limit, created, updated, deleted)


class Bisect(TableModule):
    """
    """
    parameters = [('column', str, "unknown"),
                      ('op', str, ">"),
                      ("limit_key", str, ""),
                      ('hist_index', object, None) # to improve ...
                      ] 
    def __init__(self, scheduler=None, **kwds):
        """
        """
        self._add_slots(kwds,'input_descriptors',
                            [SlotDescriptor('table', type=Table, required=True),
                                 SlotDescriptor('limit', type=Table, required=True)])
        super(Bisect, self).__init__(scheduler=scheduler, **kwds)
        self._impl = BisectImpl(self.params.column,
                                          self.params.op, self.params.hist_index) 

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
        with input_slot.lock:
            input_table = input_slot.data()
        p = self.params
        limit_slot = self.get_input_slot('limit')
        limit_slot.update(run_number, self.id)
        limit_value = None
        if limit_slot.updated.any():
            lkey = p.limit_key if p.limit_key else 0
            limit_value = limit_slot.data().last(lkey)
        if not self._impl.is_started:
            self._table = TableSelectedView(input_table, bitmap([]))
            status = self._impl.start(input_table, limit_value,
                                                 created=created,
                                                 updated=updated,
                                                 deleted=deleted)
            self._table.selection = self._impl.result._values
        else:
            status = self._impl.resume(limit_value,
                                                created=created,
                                                updated=updated,
                                                deleted=deleted)
            self._table.selection = self._impl.result._values            
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)
        
