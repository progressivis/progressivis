from progressivis import Constant, ProgressiveError, SlotDescriptor

import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Variable(Constant):
    def __init__(self, df=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('like', type=pd.DataFrame, required=False)])
        super(Variable, self).__init__(df, **kwds)

    def add_input(self, input):
        if not isinstance(input,dict):
            raise ProgressiveError('Expecting a dictionary')
        if self._df is None and self.get_input_slot('like') is None:
            error = 'Variable %s with no initial value and no input slot'%self.id
            logger.error(error)
            return error
        last = self.last_row(self._df)
        if last is None:
            last = {v: None for v in self._df.columns}
        else:
            last = last.to_dict()
        error = ''
        for (k, v) in input.iteritems():
            if k in last:
                last[k] = v
            else:
                error += 'Invalid key %s ignored. '%k
        with self.lock:
            run_number = self.scheduler().run_number()+1
            last[self.UPDATE_COLUMN] = run_number
            self._df.loc[run_number] = last
        return error
    
    def run_step(self,run_number,step_size,howlong):
        if self._df is None:
            slot = self.get_input_slot('like')
            if slot is not None:
                df = slot.data()
                if df is not None:
                    with slot.lock:
                        self._df = df.iloc[0:0] # create an empty copy
        return self._return_run_step(self.state_blocked, steps_run=1)
