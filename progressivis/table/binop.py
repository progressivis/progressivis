from __future__ import annotations

import operator
import logging

from ..core.slot import SlotDescriptor
from ..core.module import ReturnRunStep
from .module import TableModule
from ..core.bitmap import bitmap
from .table import Table

from typing import Any, Callable, Union, Optional, Dict

Binoperator = Callable[[Any, Any], Any]


logger = logging.getLogger(__name__)

ops: Dict[str, Binoperator] = {
    "<": operator.__lt__,
    "<=": operator.__le__,
    ">": operator.__gt__,
    ">=": operator.__ge__,
    "and": operator.__and__,
    "&": operator.__and__,
    "or": operator.__or__,
    "|": operator.__or__,
    "xor": operator.__xor__,
    "^": operator.__xor__,
    "==": operator.__eq__,
    "=": operator.__eq__,
    "!=": operator.__ne__,
    "is": operator.is_,
    "+": operator.__add__,
    "//": operator.__floordiv__,
    "<<": operator.__lshift__,
    "%": operator.__mod__,
    "*": operator.__mul__,
    "**": operator.__pow__,
    ">>": operator.__rshift__,
    "-": operator.__sub__,
    "/": operator.__truediv__,
}
inv_ops: Dict[Binoperator, str] = {v: k for k, v in ops.items()}


class Binop(TableModule):
    inputs = [
        SlotDescriptor("arg1", type=Table, required=True),
        SlotDescriptor("arg2", type=Table, required=True),
    ]

    def __init__(
        self,
        binop: Binoperator,
        combine: Union[None, str, Binoperator] = None,
        **kwds: Any
    ) -> None:
        super(Binop, self).__init__(**kwds)
        self.default_step_size = 1000
        self.op: Optional[Binoperator] = binop
        if callable(binop):
            self._op = binop
        else:
            self._op = ops[binop]
        self.combine = combine
        if combine is None or callable(combine):
            self._combine = combine
        else:
            self._combine = ops[combine]
        self._bitmap: Optional[bitmap] = None

    def get_data(self, name: str) -> Any:
        if name == "select":
            return self._bitmap
        if name == "table":
            slot = self.get_input_slot("table")
            if slot is not None:
                return slot.data()
            return None
        return super(Binop, self).get_data(name)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        arg1_slot = self.get_input_slot("table")
        assert arg1_slot is not None
        arg1_data = arg1_slot.data()
        arg2_slot = self.get_input_slot("cmp")
        assert arg2_slot is not None
        arg2_data = arg1_slot.data()

        if (
            arg1_data is None
            or len(arg1_data) == 0
            or arg2_data is None
            or len(arg2_data) == 0
        ):
            # nothing to do if no filter is specified
            self._bitmap = None
            return self._return_run_step(self.state_blocked, steps_run=1)
        if arg1_slot.deleted.any() or arg2_slot.deleted.any():
            arg1_slot.reset()
            arg1_slot.reset()
            self._bitmap = None
            arg1_slot.update(run_number)
            arg2_slot.update(run_number)

        length = min(len(arg1_data), len(arg2_data))
        cr1 = arg1_slot.created.next(as_slice=False)
        up1 = arg1_slot.updated.next(as_slice=False)
        cr2 = arg2_slot.created.next(as_slice=False)
        up2 = arg2_slot.updated.next(as_slice=False)
        work = cr1 | up1 | cr2 | up2
        work &= bitmap(slice(0, length))
        work.pop(step_size)
        return self._return_run_step(self.state_blocked, steps_run=1)
