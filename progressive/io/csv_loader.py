from progressive.core.dataframe import DataFrameModule

import pandas as pd
import numpy as np


class CSVLoader(DataFrameModule):
    def __init__(self, filepath_or_buffer, **kwds):
        super(CSVLoader, self).__init__(**kwds)
        self.default_step_size = kwds.get('chunksize', 1000)  # initial guess
        kwds.setdefault('chunksize', self.default_step_size)
        # Filter out the module keywords from the csv loader keywords
        csv_kwds = self._filter_kwds(kwds, pd.read_csv)
        # When called with a specified chunksize, it returns a parser
        self.parser = pd.read_csv(filepath_or_buffer, **csv_kwds)
        self._rows_read = 0

    def rows_read():
        return self._rows_read

    def run_step(self,run_number,step_size, howlong):
        df = self.parser.read(step_size)
        creates = len(df)
        self._rows_read += creates
        df[self.UPDATE_COLUMN] = run_number
        if self._df is not None:
            self._df = self._df.append(df,ignore_index=True)
        else:
            self._df = df
        return self._return_run_step(self.state_ready, steps_run=creates, creates=creates)
