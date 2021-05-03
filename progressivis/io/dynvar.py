from ..core import aio

from progressivis import ProgressiveError
from ..table.module import TableModule
from ..utils.psdict import PsDict

class DynVar(TableModule):
    def __init__(self, init_val=None, vocabulary=None, **kwds):
        super().__init__(**kwds)
        self._has_input = False
        if not (vocabulary is None  or isinstance(vocabulary, dict)):
            raise ProgressiveError('vocabulary must be a dictionary')
        self._vocabulary = vocabulary
        if not (init_val is None or isinstance(init_val, dict)):
            raise ProgressiveError('init_val must be a dictionary')        
        self.result = PsDict({} if init_val is None else init_val)        

    def is_input(self):
        return True

    def has_input(self):
        return self._has_input

    def run_step(self,run_number, step_size, howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
        #raise StopIteration()

    async def from_input(self, input_):
        #import pdb;pdb.set_trace()
        if not isinstance(input_, dict):
            raise ProgressiveError('Expecting a dictionary')
        last = PsDict(self.result) # shallow copy
        values = input_
        if self._vocabulary is not None:
            values = {self._vocabulary[k]: v for k, v in values.items()}
        for (k, v) in input_.items():
            last[k] = v
        await self.scheduler().for_input(self)
        #last['_update'] = run_number
        self.result.update(values)
        self._has_input = True
        await aio.sleep(0)
        return ''
