from progressive.core.dataframe import DataFrameModule

import pandas as pd
import numpy as np



class CSVLoader(DataFrameModule):
    def __init__(self, filepath_or_buffer, **kwds):
        """CSVLoader(filepath_or_buffer, sep=', ', dialect=None, compression='infer', doublequote=True, escapechar=None, quotechar='"', quoting=0, skipinitialspace=False, lineterminator=None, header='infer', index_col=None, names=None, prefix=None, skiprows=None, skipfooter=None, skip_footer=0, na_values=None, na_fvalues=None, true_values=None, false_values=None, delimiter=None, converters=None, dtype=None, usecols=None, engine=None, delim_whitespace=False, as_recarray=False, na_filter=True, compact_ints=False, use_unsigned=False, low_memory=True, buffer_lines=None, warn_bad_lines=True, error_bad_lines=True, keep_default_na=True, thousands=None, comment=None, decimal='.', parse_dates=False, keep_date_col=False, dayfirst=False, date_parser=None, memory_map=False, float_precision=None, nrows=None, chunksize=None, verbose=False, encoding=None, squeeze=False, mangle_dupe_cols=True, tupleize_cols=False, infer_datetime_format=False, skip_blank_lines=True, id=None,scheduler=None,tracer=None,predictor=None,storage=None,input_descriptors=[],output_descriptors=[])
        """
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
        if step_size==0: # bug
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        df = self.parser.read(step_size)
        creates = len(df)
        self._rows_read += creates
        df[self.UPDATE_COLUMN] = run_number
        if self._df is not None:
            self._df = self._df.append(df,ignore_index=True)
        else:
            self._df = df
        return self._return_run_step(self.state_ready, steps_run=creates, creates=creates)
