from progressivis.core import DataFrameModule, ProgressiveError


import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


class RandomTable(DataFrameModule):
    def __init__(self, columns, rows=-1, random=np.random.rand, **kwds):
        super(RandomTable, self).__init__(**kwds)
        self.default_step_size = 1000
        if isinstance(columns,int):
            self.columns = list(range(1,columns+1))
        elif isinstance(columns,(list,np.ndarray)):
            self.columns = columns
        else:
            raise ProgressiveError('Invalid type for columns')
        cols = len(self.columns)
        self.columns.append(self.UPDATE_COLUMN)
        self.rows = rows
        self.random = random
        self._df = self.create_dataframe(self.columns, types=cols*[np.dtype(float)]+[np.dtype(int)])
        self.columns = self._df.columns # reuse the pandas index structure

    def run_step(self,run_number,step_size, howlong):
        if step_size==0: # bug
            logger.error('Received a step_size of 0')
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        logger.info('generating %d lines', step_size)
        if self.rows >= 0 and (len(self._df)+step_size) > self.rows:
            step_size = self.rows - len(self._df)
            if step_size <= 0:
                raise StopIteration
            logger.info('truncating to %d lines', step_size)

        values = {}
        for c in self.columns[:-1]:
            s = pd.Series(self.random(step_size))
            values[c] = s
        values[self.UPDATE_COLUMN] = pd.Series(step_size*[run_number], dtype=np.dtype(int))
        df = pd.DataFrame(values, columns=self.columns)
        if self._df is not None and len(self._df) != 0:
            self._df = self._df.append(df,ignore_index=True)
        else:
            self._df = df
        return self._return_run_step(self.state_ready, steps_run=step_size, creates=step_size)
