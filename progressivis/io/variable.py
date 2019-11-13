import logging
import asyncio as aio
logger = logging.getLogger(__name__)

from progressivis import ProgressiveError, SlotDescriptor
from progressivis.table.table import Table
from progressivis.table.constant import Constant
from ..core.utils import all_string

class Variable(Constant):
    def __init__(self, table=None, **kwds):
        self._has_input = False
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('like', type=Table, required=False)])
        super(Variable, self).__init__(table, **kwds)
        #self._frozen = True
        
    def is_input(self):
        return True

    def has_input(self):
        return self._has_input
    async def from_input(self, input_):
        #print("RECEIVED FROM INPUT")
        if not isinstance(input_,dict):
            raise ProgressiveError('Expecting a dictionary')
        if self._table is None and self.get_input_slot('like') is None:
            error = 'Variable %s with no initial value and no input slot'%self.name
            logger.error(error)
            return error
        if self._table is None:
            error = f'Variable {self.name} have to run once before receiving input'
            logger.error(error)
            return error            
        last = self._table.last()
        if last is None:
            last = {v: None for v in self._table.columns}
        else:
            last = last.to_json()
        error = ''
        for (k, v) in input_.items():
            if k in last:
                last[k] = v
            else:
                error += 'Invalid key %s ignored. '%k
        _ = self.scheduler().for_input(self)
        #last['_update'] = run_number
        self._table.add(last)
        self._has_input = True
        self.me_first()
        await aio.sleep(0)
        return error
    
    async def run_step(self,run_number,step_size,howlong):
        if self._table is None:
            slot = self.get_input_slot('like')
            if slot is not None:
                like = slot.data()
                if like is not None:
                    #with slot.lock:
                    self._table = Table(self.generate_table_name('like'),
                                        dshape=like.dshape,
                                        create=True)
                    self._table.append(like.last().to_dict(ordered=True), indices=[0])
                    self._ignore_inputs = True
        else:
            #import pdb;pdb.set_trace()
            self._table.touch_rows(self._table.last_id-1)
            self.suspend()
        #else:
        #    import pdb;pdb.set_trace()
        #print("VARIABLE RUN STEP: ", self.has_input())
        #if self._table:
        #    print("LAST: ", self._table.last().to_dict())
        if self._has_input:
            self._has_input = False
            self.post_interaction_proc()
        return self._return_run_step(self.state_blocked, steps_run=1)
        #raise StopIteration()

class VirtualVariable(Constant):
    def __init__(self, names, **kwds):
        if not all_string(names):
            raise ProgressiveError('names {} must be a set of strings'.format(names))
        self._names = names
        self._key = frozenset(names)
        self._subscriptions = []
        table = None
        super(VirtualVariable, self).__init__(table, **kwds)

    def is_input(self):
        return True

    def subscribe(self, var, vocabulary):
        """
        Example: vocabulary = {'x': 'longitude', 'y': 'latitude'}
        """
        if not isinstance(var, Variable):
            raise ProgressiveError('Expecting a Variable module')
        if not isinstance(vocabulary, dict):
            raise ProgressiveError('Expecting a dictionary')
        if frozenset(vocabulary.keys()) != self._key or not all_string(vocabulary.values()):
            raise ProgressiveError('Inconsistent vocabulary')
        self._subscriptions.append((var, vocabulary))

    async def from_input(self, input_):
        if not isinstance(input_, dict):
            raise ProgressiveError('Expecting a dictionary')
        for var, vocabulary in self._subscriptions:
            translation = {vocabulary[k]: v for k, v in input_.items()}
            await var.from_input(translation)
        return ''

    async def run_step(self, run_number, step_size, howlong):
        self.suspend()
        return self._return_run_step(self.state_blocked, steps_run=1)
        #raise StopIteration()
