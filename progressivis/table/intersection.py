"""
Range Query module.

"""
from __future__ import annotations

from progressivis.core.module import Module, ReturnRunStep, def_input, def_output, document
from progressivis.core.utils import indices_len
from progressivis.core.pintset import PIntSet
from progressivis.table.table_base import BasePTable, PTableSelectedView

from typing import Any, List


def _get_physical_table(t: BasePTable) -> BasePTable:
    return t.base or t


@document
@def_input("table", type=BasePTable, multiple=True, doc="Many tables or views sharing the same physical table")
@def_output("result", PTableSelectedView, doc="View on the physical table shared by inputs. It's index is the intersection on inputs indices")
class Intersection(Module):
    """
    Intersection Module
    It computes the intersection of indices for all its inputs and
    provides a view containing rows shared by all input tables or views.
    |:warning:| All inputs are based of the same physical table. The columns of the
    output table are given by the common physical table
    """
    # parameters = []

    def __init__(self, **kwds: Any) -> None:
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self.run_step = self.run_step_seq  # type: ignore

    def predict_step_size(self, duration: float) -> int:
        return 1000

    def run_step_progress(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        _b = PIntSet.aspintset
        # to_delete: List[PIntSet]
        to_create: List[PIntSet] = []
        steps = 0
        tables = []
        ph_table = None
        # assert len(self.inputs) > 0
        reset_ = False
        for name in self.get_input_slot_multiple("table"):
            slot = self.get_input_slot(name)
            t = slot.data()
            assert isinstance(t, BasePTable)
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
        # to_delete = PIntSet.union(*to_delete)
        to_create_4sure = PIntSet()
        if len(to_create) == len(tables):
            to_create_4sure = PIntSet.intersection(*to_create)

        to_create_maybe = PIntSet.union(*to_create)

        if not self.result:
            assert ph_table is not None
            self.result = PTableSelectedView(ph_table, PIntSet([]))
        if reset_:
            self.result.selection = PIntSet([])
        self.result.selection = self.result.index | to_create_4sure
        to_create_maybe -= to_create_4sure
        eff_create = to_create_maybe
        for t in tables:
            eff_create &= t.index
        self.result.selection = self.result.index | eff_create
        return self._return_run_step(self.state_blocked, steps)

    def run_step_seq(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        steps = 0
        tables = []
        ph_table = None
        # assert len(self.inputs) > 0
        for name in self.get_input_slot_multiple("table"):
            if not name.startswith("table"):
                continue
            slot = self.get_input_slot(name)
            t = slot.data()
            assert isinstance(t, BasePTable)
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
            assert ph_table is not None
            self.result = PTableSelectedView(ph_table, PIntSet([]))
        self.result.selection = PIntSet.intersection(*[t.index for t in tables])
        return self._return_run_step(self.state_blocked, steps)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:  # pragma no cover
        raise NotImplementedError("run_step not defined")
