from __future__ import annotations

from progressivis.core.utils import indices_len

import numpy as np

from . import BasePTable, PTable, PTableSelectedView
from ..core.module import Module, ReturnRunStep, def_input, def_output, def_parameter
from ..core.pintset import PIntSet

# from .mod_impl import ModuleImpl
from .binop import ops

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .hist_index import HistogramIndex


def _get_physical_table(t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return t if t.base is None else _get_physical_table(t.base)


class _Selection(object):
    def __init__(self, values: Optional[PIntSet] = None) -> None:
        self._values = PIntSet([]) if values is None else values

    def update(self, values: PIntSet) -> None:
        self._values.update(values)

    def remove(self, values: PIntSet) -> None:
        self._values = self._values - PIntSet(values)

    def assign(self, values: PIntSet) -> None:
        self._values = values

    def add(self, values: PIntSet) -> None:
        self._values |= values


class BisectImpl:  # (ModuleImpl):
    def __init__(self, column: str, op: str, hist_index: HistogramIndex):
        super(BisectImpl, self).__init__()
        self.is_started = False
        self._table: Optional[BasePTable] = None
        self._column = column
        if isinstance(op, str):
            self._op = ops[op]
        elif op not in ops.values():
            raise ValueError("Invalid operator {}".format(op))
        self.has_cache = False
        self._hist_index = hist_index
        self.result: _Selection = _Selection()

    def resume(
        self,
        limit: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        if limit_changed:
            new_sel = self._hist_index.query(self._op, limit)
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            res = self._hist_index.restricted_query(self._op, limit, updated)
            self.result.add(res)  # add not defined???
        if created:
            res = self._hist_index.restricted_query(self._op, limit, created)
            self.result.update(res)
        if deleted:
            self.result.remove(deleted)

    def start(
        self,
        table: BasePTable,
        limit: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        self._table = table
        self.result = _Selection()
        self.is_started = True
        self.resume(limit, limit_changed, created, updated, deleted)


@def_parameter("column", np.dtype(object), "unknown")
@def_parameter("op", np.dtype(object), ">")
@def_parameter("limit_key", np.dtype(object), "")
@def_input("table", PTable, required=True)
@def_input("limit", PTable, required=False)
@def_output("result", PTableSelectedView)
class Bisect(Module):
    """ """

    def __init__(self, hist_index: HistogramIndex, **kwds: Any) -> None:
        super(Bisect, self).__init__(**kwds)
        self._impl = BisectImpl(self.params.column, self.params.op, hist_index)
        self.default_step_size = 1000
        self._run_once = False

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        self._run_once = True
        input_slot = self.get_input_slot("table")
        # input_slot.update(run_number)
        steps = 0
        deleted = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            steps += 1  # indices_len(deleted)
        created = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            steps += indices_len(created)
        updated = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            steps += indices_len(updated)
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            self.result = PTableSelectedView(input_table, PIntSet([]))
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        param = self.params
        limit_slot = self.get_input_slot("limit")
        # limit_slot.update(run_number)
        limit_changed = False
        if limit_slot.deleted.any():
            limit_slot.deleted.next()
        if limit_slot.updated.any():
            limit_slot.updated.next()
            limit_changed = True
        if limit_slot.created.any():
            limit_slot.created.next()
            limit_changed = True
        if len(limit_slot.data()) == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if param.limit_key:
            limit_value = limit_slot.data().last(param.limit_key)
        else:
            limit_value = limit_slot.data().last()[0]
        if not self._impl.is_started:
            self._impl.start(
                input_table,
                limit_value,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
        else:
            self._impl.resume(
                limit_value,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
        self.result.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
