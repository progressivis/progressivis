from . import Table
from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import TableModule
from ..core.utils import indices_len
from .filter_impl import FilterImpl
from ..core.bitmap import bitmap


class FilterMod(TableModule):
    parameters = [('expr', str, "unknown"),
                  ('user_dict', object, None)]

    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super(FilterMod, self).__init__(**kwds)
        self._impl = FilterImpl(self.params.expr, self.params.user_dict)

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        input_slot.update(run_number)
        steps = 0
        if input_slot.updated.any():
            input_slot.reset()
            if self._table is not None:
                self._table.selection = bitmap([])
            input_slot.update(run_number)
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
        input_table = input_slot.data()
        if not self._impl.is_started:
            self._table = TableSelectedView(input_table, bitmap([]))
            status = self._impl.start(input_table,
                                      created=created,
                                      updated=updated,
                                      deleted=deleted)
            self._table.selection = self._impl.result._values
        else:
            status = self._impl.resume(created=created,
                                       updated=updated,
                                       deleted=deleted)
            self._table.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
