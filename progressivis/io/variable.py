from __future__ import annotations

import logging
import copy

from progressivis import ProgressiveError, SlotDescriptor
from progressivis.table.table import PTable
from progressivis.table.constant import ConstDict
from progressivis.utils.psdict import PDict

from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from progressivis.core.module import ReturnRunStep, JSon

logger = logging.getLogger(__name__)


class Variable(ConstDict):
    inputs = [SlotDescriptor("like", type=(PTable, PDict), required=False)]

    def __init__(self, table: Optional[PDict] = None, **kwds: Any) -> None:
        super(Variable, self).__init__(table, **kwds)
        self.tags.add(self.TAG_INPUT)

    async def from_input(self, input_: JSon) -> str:
        if not isinstance(input_, dict):
            raise ProgressiveError("Expecting a dictionary")
        if self.result is None and self.get_input_slot("like") is None:
            error = f"Variable {self.name} with no initial value and no input slot"
            logger.error(error)
            return error
        if self.result is None:
            error = f"Variable {self.name} has to run once before receiving input"
            logger.error(error)
            return error
        last: PDict = copy.copy(self.result)
        error = ""
        for (k, v) in input_.items():
            if k in last:
                last[k] = v
            else:
                error += f"Invalid key {k} ignored. "
        await self.scheduler().for_input(self)
        self.result.update(last)
        return error

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if self.result is None:
            slot = self.get_input_slot("like")
            if slot is not None:
                like = slot.data()
                if like is not None:
                    if isinstance(like, PTable):
                        last = like.last()
                        assert last is not None
                        like = last.to_dict(ordered=True)
                    self.result = PDict(like)
                    self._ignore_inputs = True
        return self._return_run_step(self.state_blocked, steps_run=1)
