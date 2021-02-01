"""
Range Query module.

"""

from progressivis.table.nary import NAry
from progressivis.core.utils import indices_len
from . import BaseTable
from ..core.bitmap import bitmap
from . import TableSelectedView


def _get_physical_table(t):
    return t.base or t

class Intersection(NAry):
    "Intersection Module"
    parameters = []

    def __init__(self, **kwds):
        super(Intersection, self).__init__(**kwds)
        self.run_step = self.run_step_seq

    def predict_step_size(self, duration):
        return 1000

    def run_step_progress(self, run_number, step_size, howlong):
        _b = bitmap.asbitmap
        to_delete = []
        to_create = []
        steps = 0
        tables = []
        ph_table = None
        assert len(self.inputs) > 0
        reset_ = False
        for name in self.get_input_slot_multiple():
            slot = self.get_input_slot(name)
            t = slot.data()
            assert isinstance(t, BaseTable)
            if ph_table is None:
                ph_table = _get_physical_table(t)
            else:
                assert ph_table is _get_physical_table(t)
            tables.append(t)
            # slot.update(run_number)
            if reset_ or slot.updated.any() or slot.deleted.any():
                slot.reset()
                reset_ = True
                steps += 1

            #if slot.deleted.any():
            #    deleted = slot.deleted.next(step_size)
            #    steps += 1
            #    to_delete.append(_b(deleted))
            #if slot.updated.any(): # actually don't care
            #    _ = slot.updated.next(step_size)
            #    #to_delete |= _b(updated)
            #    #to_create |= _b(updated)
            #    #steps += 1 # indices_len(updated) + 1
            if slot.created.any():
                created = slot.created.next(step_size)
                bm = _b(created)  # - to_delete
                to_create.append(bm)
                steps += indices_len(created)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        to_delete = bitmap.union(*to_delete)
        to_create_4sure = bitmap()
        if len(to_create) == len(tables):
            to_create_4sure = bitmap.intersection(*to_create)

        to_create_maybe = bitmap.union(*to_create)

        if not self._table:
            self._table = TableSelectedView(ph_table, bitmap([]))
        if reset_:
            self._table.mask = bitmap([])
        # self._table.selection -= to_delete
        self._table.mask = self._table.index|to_create_4sure
        to_create_maybe -= to_create_4sure
        eff_create = to_create_maybe
        for t in tables:
            eff_create &= t.index
        self._table.mask = self._table.index|eff_create
        # self.get_input_slot(self.inputs[0]))
        return self._return_run_step(self.state_blocked, steps)

    def run_step_seq(self, run_number, step_size, howlong):
        steps = 0
        tables = []
        ph_table = None
        assert len(self.inputs) > 0
        for name in self.get_input_slot_multiple():
            if not name.startswith('table'):
                continue
            slot = self.get_input_slot(name)
            t = slot.data()
            assert isinstance(t, BaseTable)
            if ph_table is None:
                ph_table = _get_physical_table(t)
            else:
                assert ph_table is _get_physical_table(t)
            tables.append(t)
            # slot.update(run_number)
            if slot.deleted.any():
                slot.deleted.next()
                steps += 1
            if slot.updated.any():
                slot.updated.next()
                steps += 1
            if slot.created.any():
                slot.created.next()
                steps += 1
        if steps == 0:
            return self._return_run_step(self.state_blocked, 0)
        if not self._table:
            self._table = TableSelectedView(ph_table, bitmap([]))
        self._table.mask = bitmap.intersection(*[t.index
                                                      for t in tables])
        # return self._return_run_step(self.next_state(self.get_input_slot(self.inputs[0])), steps)
        return self._return_run_step(self.state_blocked, steps)
