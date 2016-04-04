from progressivis import Constant, ProgressiveError, SlotDescriptor
from progressivis.core.utils import last_row

import pandas as pd

import logging
logger = logging.getLogger(__name__)

class AddToRow(Constant):
    def __init__(self, df=None, **kwds):
        super(AddToRow, self).__init__(df, **kwds)

    def is_input(self):
        return True

    def from_input(self, input):
        if not isinstance(input,dict):
            raise ProgressiveError('Expecting a dictionary')
        if self._df is None:
            error = 'AddToRow %s with no initial value and no input slot'%self.id
            logger.error(error)
            return error

        run_number = 0
        for (row, value) in input.iteritems():
            self._df.loc[row, self.get_columns(self._df)] += value
            if run_number == 0:
                run_number = self.scheduler().for_input(self)
            self._df.at[row, self.UPDATE_COLUMN] = run_number
        if run_number != 0:
            self._last_update = run_number
        return "OK"

    def run_step(self,run_number,step_size,howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
