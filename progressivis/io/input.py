# -*- coding: utf-8 -*-
from progressivis import DataFrameModule
from progressivis.core.utils import create_dataframe

import pandas as pd
import numpy as np

class Input(DataFrameModule):
    parameters = [('history', np.dtype(int), 3)]
    schema = [('input', np.dtype(object), None),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, **kwds):
        super(Input, self).__init__(**kwds)
        self._df = create_dataframe(Input.schema,empty=True)
        self._last = len(self._df)
        self.default_step_size = 1000000


    def is_ready(self):
        return len(self._df) > self._last

    def run_step(self,run_number,step_size, howlong):
        self._last = len(self._df)
        return self._return_run_step(self.state_blocked, steps_run=0)
        
    def from_input(self, input):
        if not isinstance(input,list):
            input = [input]
        run_number = self.scheduler().for_input(self)
        df = pd.DataFrame({'input': input,
                           self.UPDATE_COLUMN: run_number})
        self._df = self._df.append(df,ignore_index=True)

    def is_input(self):
        return True
