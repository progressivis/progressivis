import logging

from progressivis import ProgressiveError, SlotDescriptor
from progressivis.table.table import Table
from progressivis.table.constant import Constant
from ..core.utils import all_string
from ..utils.psdict import PsDict
import copy

logger = logging.getLogger(__name__)


class Variable(Constant):
    inputs = [SlotDescriptor('like', type=(Table, PsDict), required=False)]

    def __init__(self, table=None, **kwds):
        super(Variable, self).__init__(table, **kwds)
        self.tags.add(self.TAG_INPUT)

    async def from_input(self, input_):
        if not isinstance(input_, dict):
            raise ProgressiveError('Expecting a dictionary')
        if self.result is None and self.get_input_slot('like') is None:
            error = f'Variable {self.name} with no initial value and no input slot'
            logger.error(error)
            return error
        if self.result is None:
            error = f'Variable {self.name} has to run once before receiving input'
            logger.error(error)
            return error
        last = copy.copy(self.result)
        error = ''
        for (k, v) in input_.items():
            if k in last:
                last[k] = v
            else:
                error += f'Invalid key {k} ignored. '
        await self.scheduler().for_input(self)
        self.result.update(last)
        return error

    def run_step(self, run_number, step_size, howlong):
        if self.result is None:
            slot = self.get_input_slot('like')
            if slot is not None:
                like = slot.data()
                if like is not None:
                    if isinstance(like, Table):
                        like = like.last().to_dict(ordered=True)
                    self.result = copy.copy(like)
                    self._ignore_inputs = True
        return self._return_run_step(self.state_blocked, steps_run=1)


class VirtualVariable(Constant):
    def __init__(self, names, **kwds):
        if not all_string(names):
            raise ProgressiveError(f'names {names} must be a set of strings')
        self._names = names
        self._key = frozenset(names)
        self._subscriptions = []
        table = None
        super(VirtualVariable, self).__init__(table, **kwds)
        self.tags.add(self.TAG_INPUT)

    def subscribe(self, var, vocabulary):
        """
        Example: vocabulary = {'x': 'longitude', 'y': 'latitude'}
        """
        if not isinstance(var, Variable):
            raise ProgressiveError('Expecting a Variable module')
        if not isinstance(vocabulary, dict):
            raise ProgressiveError('Expecting a dictionary')
        if frozenset(vocabulary.keys()) != self._key or \
           not all_string(vocabulary.values()):
            raise ProgressiveError('Inconsistent vocabulary')
        self._subscriptions.append((var, vocabulary))

    async def from_input(self, input_):
        if not isinstance(input_, dict):
            raise ProgressiveError('Expecting a dictionary')
        for var, vocabulary in self._subscriptions:
            translation = {vocabulary[k]: v for k, v in input_.items()}
            await var.from_input(translation)
        return ''

    def run_step(self, run_number, step_size, howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
