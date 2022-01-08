"""
Range Query module.

"""
from __future__ import annotations

from progressivis.core.module import ReturnRunStep
from progressivis.core.utils import indices_len
from progressivis.core.bitmap import bitmap
from progressivis.table.nary import NAry
from progressivis.table.table_base import BaseTable, TableSelectedView

from typing import Any, List


def _get_physical_table(t: BaseTable) -> BaseTable:
    return t.base or t


class Intersection(NAry):
    "Intersection Module"
    # parameters = []

    def __init__(self, **kwds: Any) -> None:
        super(Intersection, self).__init__(**kwds)
        self.run_step = self.run_step_seq  # type: ignore

    def predict_step_size(self, duration: float) -> int:
        return 1000

    def run_step_progress(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        _b = bitmap.asbitmap
        # to_delete: List[bitmap]
        to_create: List[bitmap]
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

            # if slot.deleted.any():
            #    deleted = slot.deleted.next(step_size)
            #    steps += 1
            #    to_delete.append(_b(deleted))
            # if slot.updated.any(): # actually don't care
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
        # to_delete = bitmap.union(*to_delete)
        to_create_4sure = bitmap()
        if len(to_create) == len(tables):
            to_create_4sure = bitmap.intersection(*to_create)

        to_create_maybe = bitmap.union(*to_create)

        if not self.result:
            self.result = TableSelectedView(ph_table, bitmap([]))
        if reset_:
            self.selected.selection = bitmap([])
        self.selected.selection = self.selected.index | to_create_4sure
        to_create_maybe -= to_create_4sure
        eff_create = to_create_maybe
        for t in tables:
            eff_create &= t.index
        self.selected.selection = self.selected.index | eff_create
        return self._return_run_step(self.state_blocked, steps)

    def run_step_seq(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        steps = 0
        tables = []
        ph_table = None
        assert len(self.inputs) > 0
        for name in self.get_input_slot_multiple():
            if not name.startswith("table"):
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
        if not self.result:
            self.result = TableSelectedView(ph_table, bitmap([]))
        self.selected.selection = bitmap.intersection(*[t.index for t in tables])
        return self._return_run_step(self.state_blocked, steps)
