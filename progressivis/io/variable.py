from __future__ import absolute_import, division, print_function

import six
import logging
logger = logging.getLogger(__name__)

from progressivis import ProgressiveError, SlotDescriptor
from progressivis.table.table import Table
from progressivis.table.constant import Constant


class Variable(Constant):
    def __init__(self, table=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('like', type=Table, required=False)])
        super(Variable, self).__init__(table, **kwds)

    def is_input(self):
        return True

    def from_input(self, input_):
        if not isinstance(input_,dict):
            raise ProgressiveError('Expecting a dictionary')
        if self._table is None and self.get_input_slot('like') is None:
            error = 'Variable %s with no initial value and no input slot'%self.id
            logger.error(error)
            return error
        last = self._table.last()
        if last is None:
            last = {v: None for v in self._table.columns}
        else:
            last = last.to_json()
        error = ''
        for (k, v) in six.iteritems(input_):
            if k in last:
                last[k] = v
            else:
                error += 'Invalid key %s ignored. '%k
        _ = self.scheduler().for_input(self)
        #last['_update'] = run_number
        self._table.add(last)
        return error
    
    def run_step(self,run_number,step_size,howlong):
        if self._table is None:
            slot = self.get_input_slot('like')
            if slot is not None:
                like = slot.data()
                if like is not None:
                    with slot.lock:
                        self._table = Table(self.generate_table_name('like'),
                                            dshape=like.dshape,
                                            create=True)
                        self._table.append(like.last().to_dict(ordered=True), indices=[0])
        #return self._return_run_step(self.state_blocked, steps_run=1)
        raise StopIteration()
