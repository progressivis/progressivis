from __future__ import annotations

from progressivis.core.utils import indices_len

import numpy as np

from .api import BasePTable, PTable, PTableSelectedView
from ..core.module import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document,
)
from ..core.pintset import PIntSet
from ..utils.psdict import PDict
from .binop import ops

from typing import Optional, Any
from .binning_index import BinningIndex


def _get_physical_table(t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return t if t.base is None else _get_physical_table(t.base)


def _first_val(d: dict[Any, Any]) -> Any:
    for val in d.values():
        return val


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


class SimpleFilterImpl:
    def __init__(self, column: str, op: str, hist_index: BinningIndex):
        super().__init__()
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
        value: float,
        value_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        if value_changed:
            new_sel = self._hist_index.query(self._op, value)
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            res = self._hist_index.restricted_query(
                self._op, value, updated
            )
            self.result.add(res)  # add not defined???
        if created:
            res = self._hist_index.restricted_query(
                self._op, value, created
            )
            self.result.update(res)
        if deleted:
            self.result.remove(deleted)

    def start(
        self,
        table: BasePTable,
        value: float,
        value_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        self._table = table
        self.result = _Selection()
        self.is_started = True
        self.resume(value, value_changed, created, updated, deleted)


@document
@def_parameter(
    "column",
    np.dtype(object),
    "unknown",
    doc="filtering column (condition's" " left operand)",
)
@def_parameter(
    "op", np.dtype(object), ">", doc="relational operator (i.e. '>', '>=', '<', '<=')"
)
@def_parameter("value_key", np.dtype(object), "", doc="see ``value`` input below")
@def_input("table", PTable, required=True, doc="input table or view")
@def_input(
    "value",
    PDict,
    required=True,
    doc=(
        "contains the condition's right operand."
        "if ``value_key`` is provided the right"
        " operand  is ``value['value_key']`` else"
        " the first value in the dict is used"
    ),
)
@def_input(
    "hist",
    PTable,
    doc=(
        "**BinningIndex** module output connected to the same input/column."
        "This mandatory parameter could be provided "
        "by the `create_dependent_modules()` method."
    ),
)
@def_output("result", PTableSelectedView)
class SimpleFilter(Module):
    """
    Filtering module based on a simple condition of the form

     ``<column> <operator> <value>`` where

     ``<operator> := '>' | '>=' | '<' | '<='``
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self._impl: Optional[SimpleFilterImpl] = None
        self.default_step_size = 1000
        self.input_module: Optional[Module] = None

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str,
        hist_index: Optional[BinningIndex] = None,
        **kwds: Any,
    ) -> SimpleFilter:
        """
        Creates a default configuration.

        Args:
            input_module: the input module (see the example)
            input_slot: the input slot name (e.g. ``result``)
            hist_index: optional histogram index. if not provided an
                ``BinningIndex`` is created
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        if self.input_module is not None:  # test if already called
            return self
        scheduler = self.scheduler()
        params = self.params
        self.input_module = input_module
        self.input_slot = input_slot
        with scheduler:
            if hist_index is None:
                hist_index = BinningIndex(
                    group=self.name, scheduler=scheduler
                )
                hist_index.input.table = input_module.output[input_slot][params.column,]
            filter_ = self
            filter_.dep.hist_index = hist_index
            filter_.input.hist = hist_index.output.result
            filter_.input.table = hist_index.output.result
        return self

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        if self._impl is None:
            self._impl = SimpleFilterImpl(
                self.params.column, self.params.op, self.dep.hist_index
            )
        input_slot = self.get_input_slot("table")
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # input_slot.update(run_number)
        steps = 0
        deleted = None
        hist_slot = self.get_input_slot("hist")
        hist_slot.clear_buffers()
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
        if self.result is None:
            self.result = PTableSelectedView(input_table, PIntSet([]))
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        param = self.params
        value_slot = self.get_input_slot("value")
        # value_slot.update(run_number)
        value_changed = value_slot.has_buffered()
        value_slot.clear_buffers()
        if len(value_slot.data()) == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if param.value_key:
            value_ = value_slot.data()[param.value_key]
        else:
            value_ = _first_val(value_slot.data())
        if not self._impl.is_started:
            self._impl.start(
                input_table,
                value_,
                value_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
        else:
            self._impl.resume(
                value_,
                value_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
        self.result.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
