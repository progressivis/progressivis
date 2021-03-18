import logging
logger = logging.getLogger(__name__)

import numpy as np

from progressivis import ProgressiveError
from progressivis.table.constant import Constant

class AddToRow(Constant):
    def __init__(self, df=None, **kwds):
        super(AddToRow, self).__init__(df, **kwds)

    def is_input(self):
        return True

    async def from_input(self, input):
        _ = input
        if not isinstance(input,dict):
            raise ProgressiveError('Expecting a dictionary')
        if self._table is None:
            error = 'AddToRow %s with no initial value and no input slot'%self.name
            logger.error(error)
            return error

        run_number = 0
        for (row_, value) in input.items():
            #self._df.loc[row, self.get_columns(self._df)] += value
            current_row = self._table.row(row_).to_dict(ordered=True)
            vals = np.array(list(current_row.values()))
            vals += value
            self._table.loc[row_, :] = vals # TODO: implement __iadd__() on Table
            if run_number == 0:
                run_number = self.scheduler().for_input(self)
            #self._df.at[row, UPDATE_COLUMN] = run_number
        if run_number != 0:
            self._last_update = run_number
        return "OK"

    def run_step(self,run_number,step_size,howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
