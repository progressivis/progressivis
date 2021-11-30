
from ..core.utils import indices_len, fix_loc, integer_types
from ..core.slot import SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any
from ..utils.errors import ProgressiveError
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Histogram1DCategorical(TableModule):
    """
    """
    parameters = [('bins', np.dtype(int), 128),
                  ('delta', np.dtype(float), -5)]
    inputs = [
        SlotDescriptor('table', type=Table, required=True)
    ]

    schema = "{ array: var * int32, min: float64, max: float64, time: int64 }"

    def __init__(self, column, **kwds):
        super().__init__(dataframe_slot='table', **kwds)
        self.column = column
        self.total_read = 0
        self.default_step_size = 1000
        self.result = PsDict()


    def reset(self):
        if self.result:
            self.result.clear()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            if not dfslot.created.any():
                logger.info('Input buffers empty')
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            column = input_df[self.column]
            indices = dfslot.created.next(step_size)  # returns a slice or ...
            steps = indices_len(indices)
            logger.info('Read %d rows', steps)
            self.total_read += steps
            column = column.loc[fix_loc(indices)]
            if column.dtype != 'O':
                raise ProgressiveError(f"Type of '{self.column}' is"
                                       " not string")
            valcounts = pd.Series(column).value_counts().to_dict()
            for k, v in valcounts.items():
                self.result[k] = v + self.result.get(k, 0)
            return self._return_run_step(self.next_state(dfslot),
                                         steps_run=steps)


    def is_visualization(self):
        return True

    def get_visualization(self):
        return "histogram1d_categorical"
