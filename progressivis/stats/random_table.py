from progressivis.core import DataFrameModule, ProgressiveError
from progressivis.core.buffered_dataframe import BufferedDataFrame

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


class RandomTable(DataFrameModule):
    def __init__(self, columns, rows=-1, random=np.random.rand, throttle=False, force_valid_ids=False, **kwds):
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
        if throttle and isinstance(throttle,(int,float)):
            self.throttle = throttle
        else:
            self.throttle = False
        self._df = self.create_dataframe(self.columns, types=cols*[np.dtype(float)]+[np.dtype(int)])
        if force_valid_ids:
            self.force_valid_id_columns(self._df)
        self.columns = self._df.columns # reuse the pandas index structure
        self._buffer = BufferedDataFrame()

    def run_step(self,run_number,step_size, howlong):
        if step_size==0: # bug
            logger.error('Received a step_size of 0')
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        logger.info('generating %d lines', step_size)
        if self.throttle:
            step_size = np.min([self.throttle,step_size])
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
        with self.lock:
            self._buffer.append(df)
            self._df = self._buffer.df()
        next_state = self.state_blocked if self.throttle else self.state_ready
        return self._return_run_step(next_state, steps_run=step_size)
