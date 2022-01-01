from __future__ import annotations

import logging
import copy

from progressivis import ProgressiveError, SlotDescriptor
from progressivis.table.table import Table
from progressivis.table.constant import Constant
from progressivis.core.utils import all_string
from progressivis.utils.psdict import PsDict

from typing import List, TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from progressivis.core.module import ReturnRunStep, JSon

logger = logging.getLogger(__name__)


class Variable(Constant):
    inputs = [SlotDescriptor("like", type=(Table, PsDict), required=False)]

    def __init__(self, table: Table = None, **kwds):
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
        last = copy.copy(self.result)
        error = ""
        for (k, v) in input_.items():
            if k in last:
                last[k] = v
            else:
                error += f"Invalid key {k} ignored. "
        await self.scheduler().for_input(self)
        self.psdict.update(last)
        return error

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        if self.result is None:
            slot = self.get_input_slot("like")
            if slot is not None:
                like = slot.data()
                if like is not None:
                    if isinstance(like, Table):
                        last = like.last()
                        assert last is not None
                        like = last.to_dict(ordered=True)
                    self.result = copy.copy(like)
                    self._ignore_inputs = True
        return self._return_run_step(self.state_blocked, steps_run=1)


class VirtualVariable(Constant):
    def __init__(self, names: List[str], **kwds):
        if not all_string(names):
            raise ProgressiveError(f"names {names} must be a set of strings")
        self._names = names
        self._key = frozenset(names)
        self._subscriptions: List[Tuple[Variable, Dict[str, str]]] = []
        table = None
        super(VirtualVariable, self).__init__(table, **kwds)
        self.tags.add(self.TAG_INPUT)

    def subscribe(self, var: Variable, vocabulary: Dict[str, str]):
        """
        Example: vocabulary = {'x': 'longitude', 'y': 'latitude'}
        """
        if not isinstance(var, Variable):
            raise ProgressiveError("Expecting a Variable module")
        if not isinstance(vocabulary, dict):
            raise ProgressiveError("Expecting a dictionary")
        if frozenset(vocabulary.keys()) != self._key or not all_string(
            vocabulary.values()
        ):
            raise ProgressiveError("Inconsistent vocabulary")
        self._subscriptions.append((var, vocabulary))

    async def from_input(self, input_: JSon) -> str:
        if not isinstance(input_, dict):
            return f"Expecting a dictionary in {repr(self)}"
        for var, vocabulary in self._subscriptions:
            translation = {vocabulary[k]: v for k, v in input_.items()}
            await var.from_input(translation)
        return ""

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        return self._return_run_step(self.state_blocked, steps_run=1)
