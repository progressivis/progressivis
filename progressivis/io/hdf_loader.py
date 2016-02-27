from progressivis import ProgressiveError, DataFrameModule
from progressivis.core.buffered_dataframe import BufferedDataFrame

import pandas as pd

import logging
logger = logging.getLogger(__name__)


class HDFLoader(DataFrameModule):
    def __init__(self, filepath_or_buffer=None, filter=None, force_valid_ids=False, **kwds):
        """HDFLoader(filepath_or_buffer=None, force_valid_ids=False, id=None,scheduler=None,tracer=None,predictor=None,storage=None)
        """
        super(HDFLoader, self).__init__(**kwds)
        assert False, "Not working yet"
        self.default_step_size = kwds.get('chunksize', 1000)  # initial guess
        kwds.setdefault('chunksize', self.default_step_size)
        # Filter out the module keywords from the csv loader keywords
        hdf_kwds = self._filter_kwds(kwds, pd.read_hdf)
        # When called with a specified chunksize, it returns a parser
        self.filepath_or_buffer = filepath_or_buffer
        self.force_valid_ids = force_valid_ids
        self.hdf_kwds = hdf_kwds
        self._rows_read = 0
        if filter is not None and not callable(filter):
            raise ProgressiveError('filter parameter should be callable or None')
        self._filter = filter
        self._buffer = BufferedDataFrame()

    def rows_read(self):
        return self._rows_read

    def is_ready(self):
        fn = self.get_input_slot('filenames')
        if fn and fn.has_created():
            return True
        return super(HDFLoader, self).is_ready()

    def validate_parser(self, run_number):
        if self.parser is None:
            if self.filepath_or_buffer is not None:
                try:
                    self.parser = pd.read_hdf(self.filepath_or_buffer, **self.hdf_kwds)
                except IOError as e:
                    logger.error('Cannot open file %s: %s', self.filepath_or_buffer, e)
                    self.parser = None
                    return self.state_terminated
                self.filepath_or_buffer = None
        return self.state_ready

    def run_step(self,run_number,step_size, howlong):
        if step_size==0: # bug
            logger.error('Received a step_size of 0')
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        status = self.validate_parser(run_number)
        if status==self.state_terminated:
            raise StopIteration()
        elif status != self.state_ready:
            logger.error('Invalid state returned by validate_parser: %d', status)
            raise StopIteration('Unexpected situation')
        logger.info('loading %d lines', step_size)
        df = self.parser.read(step_size) # raises StopIteration at EOF

        creates = len(df)
        if creates == 0: # should not happen
            logger.error('Received 0 elements')
            raise StopIteration
        if self._filter != None:
            df = self._filter(df)
        creates = len(df)
        if creates == 0:
            logger.info('frame has been filtered out')
        else:
            self._rows_read += creates
            logger.info('Loaded %d lines', self._rows_read)
            if self.force_valid_ids:
                self.force_valid_id_columns(df)
            df[self.UPDATE_COLUMN] = run_number
            with self.lock:
                self._buffer.append(df)
                self._df = self._buffer.df()
        return self._return_run_step(self.state_ready, steps_run=creates)
