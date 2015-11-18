from progressivis import ProgressiveError, DataFrameModule, SlotDescriptor

import pandas as pd

import logging
logger = logging.getLogger(__name__)


class CSVLoader(DataFrameModule):
    def __init__(self, filepath_or_buffer=None, filter=None, force_valid_ids=False, **kwds):
        """CSVLoader(filepath_or_buffer=None, sep=', ', dialect=None, compression='infer', doublequote=True, escapechar=None, quotechar='"', quoting=0, skipinitialspace=False, lineterminator=None, header='infer', index_col=None, names=None, prefix=None, skiprows=None, skipfooter=None, skip_footer=0, na_values=None, na_fvalues=None, true_values=None, false_values=None, delimiter=None, converters=None, dtype=None, usecols=None, engine=None, delim_whitespace=False, as_recarray=False, na_filter=True, compact_ints=False, use_unsigned=False, low_memory=True, buffer_lines=None, warn_bad_lines=True, error_bad_lines=True, keep_default_na=True, thousands=None, comment=None, decimal='.', parse_dates=False, keep_date_col=False, dayfirst=False, date_parser=None, memory_map=False, float_precision=None, nrows=None, chunksize=None, verbose=False, encoding=None, squeeze=False, mangle_dupe_cols=True, tupleize_cols=False, infer_datetime_format=False, skip_blank_lines=True, force_valid_ids=False, id=None,scheduler=None,tracer=None,predictor=None,storage=None,input_descriptors=[],output_descriptors=[])
        """
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('filenames', type=pd.DataFrame,required=False)])
        super(CSVLoader, self).__init__(**kwds)
        self.default_step_size = kwds.get('chunksize', 1000)  # initial guess
        kwds.setdefault('chunksize', self.default_step_size)
        # Filter out the module keywords from the csv loader keywords
        csv_kwds = self._filter_kwds(kwds, pd.read_csv)
        # When called with a specified chunksize, it returns a parser
        self.filepath_or_buffer = filepath_or_buffer
        self.force_valid_ids = force_valid_ids
        self.parser = None
        self.csv_kwds = csv_kwds
        self._rows_read = 0
        if filter is not None and not callable(filter):
            raise ProgressiveError('filter parameter should be callable or None')
        self._filter = filter

    def rows_read(self):
        return self._rows_read

    def is_ready(self):
        fn = self.get_input_slot('filenames')
        if fn and fn.has_created():
            return True
        return super(CSVLoader, self).is_ready()

    def validate_parser(self, run_number):
        if self.parser is None:
            if self.filepath_or_buffer is not None:
                try:
                    self.parser = pd.read_csv(self.filepath_or_buffer, **self.csv_kwds)
                except IOError as e:
                    logger.error('Cannot open file %s: %s', self.filepath_or_buffer, e)
                    self.parser = None
                    return self.state_terminated
                self.filepath_or_buffer = None
            else:
                fn_slot = self.get_input_slot('filenames')
                if fn_slot is None or fn_slot.output_module is None:
                    return self.state_terminated
                df = fn_slot.data()
                fn_slot.update(run_number)
                if fn_slot.has_deleted() or fn_slot.has_updated():
                    raise ProgressiveError('Cannot handle input file changes')
                while self.parser is None:
                    indices = fn_slot.next_created(1)
                    if indices.stop==indices.start:
                        return self.state_blocked
                    filename = df.at[indices.start, 'filename']
                    try:
                        self.parser = pd.read_csv(filename, **self.csv_kwds)
                    except IOError as e:
                        logger.error('Cannot open file %s: %s', filename, e)
                        self.parser = None
                    # fall through
        return self.state_ready

    # def next_step_iter(self, run_number, step_size, howlong):
    #     for filename in filenames(run_number):
    #         try:
    #             self.parser = pd.read_csv(filename, **self.csv_kwds)
    #         except IOError as e:
    #             logger.error('Cannot open file %s: %s', filename, e)
    #             continue
    #         for df in self.parser.read(step_size):
    #             creates = len(df)
    #             if self._filter != None:
    #                 df = self._filter(df)
    #             if len(df) == 0:
    #                 logger.info('frame has been filtered out')
    #             else:
    #                 self._rows_read += creates
    #                 logger.info('Loaded %d lines', self._rows_read)
    #                 df[self.UPDATE_COLUMN] = run_number
    #             if self._df is not None:
    #                 self._df = self._df.append(df,ignore_index=True)
    #             else:
    #                 self._df = df
    #             (run_number, step_size, howlong) = yield (self.state_ready, creates)
    #             while step_size==0:
    #                 (run_number, step_size, howlong) = yield (self.state_ready, 0)

    def run_step(self,run_number,step_size, howlong):
        if step_size==0: # bug
            logger.error('Received a step_size of 0')
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        status = self.validate_parser(run_number)
        if status==self.state_terminated:
            raise StopIteration('no more filenames')
        elif status==self.state_blocked:
            return self._return_run_step(status, steps_run=0, creates=0)
        elif status != self.state_ready:
            logger.error('Invalid state returned by validate_parser: %d', status)
            raise StopIteration('Unexpected situation')
        logger.info('loading %d lines', step_size)
        try:
            df = self.parser.read(step_size) # raises StopIteration at EOF
        except StopIteration:
            fn_slot = self.get_input_slot('filenames')
            if fn_slot is None or fn_slot.output_module is None:
                raise
            self.parser = None
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)

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
            if self._df is not None:
                self._df = self._df.append(df,ignore_index=True)
            else:
                self._df = df
        return self._return_run_step(self.state_ready, steps_run=creates)
