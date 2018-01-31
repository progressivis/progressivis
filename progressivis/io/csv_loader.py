from __future__ import absolute_import, division, print_function

import logging
logger = logging.getLogger(__name__)

import pandas as pd

from progressivis import ProgressiveError, SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..table.dshape import dshape_from_dataframe
from ..core.utils import filepath_to_buffer, _infer_compression, force_valid_id_columns


class CSVLoader(TableModule):
    def __init__(self,
                 filepath_or_buffer=None,
                 filter_=None,
                 force_valid_ids=True,
                 fillvalues=None,
                 **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('filenames', type=Table,required=False)])
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
        self._compression = csv_kwds.get('compression', "infer")
        csv_kwds['compression'] = None
        self._encoding = csv_kwds.get('encoding', None)
        csv_kwds['encoding'] = None
        self._rows_read = 0
        if filter_ is not None and not callable(filter_):
            raise ProgressiveError('filter parameter should be callable or None')
        self._filter = filter_
        self._input_stream = None # stream that returns a position through the 'tell()' method
        self._input_encoding = None
        self._input_compression = None
        self._input_size = 0 # length of the file or input stream when available

        self._table_params = dict(name=self.id, fillvalues=fillvalues)
        self._table = None

    def rows_read(self):
        return self._rows_read

    def is_ready(self):
        fn = self.get_input_slot('filenames')
        # Can be called before the first update so fn.created can be None
        if fn and (fn.created is None or fn.created.any()):
            return True
        return super(CSVLoader, self).is_ready()

    def open(self, filepath):
        if self._input_stream is not None:
            self.close()
        compression = _infer_compression(filepath, self._compression)
        istream, encoding, compression, size = filepath_to_buffer(filepath,
                                                                  encoding=self._encoding,
                                                                  compression=compression)
        self._input_stream = istream
        self._input_encoding = encoding
        self._input_compression = compression
        self._input_size = size
        self.csv_kwds['encoding'] = encoding
        self.csv_kwds['compression'] = compression
        return istream

    def close(self):
        if self._input_stream is None:
            return
        try:
            self._input_stream.close()
            # pylint: disable=bare-except
        except:
            pass
        self._input_stream = None
        self._input_encoding = None
        self._input_compression = None
        self._input_size = 0

    def get_progress(self):
        if self._input_size==0:
            return (0, 0)
        pos = self._input_stream.tell()
        return (pos, self._input_size)

    def validate_parser(self, run_number):
        if self.parser is None:
            if self.filepath_or_buffer is not None:
                try:
                    self.parser = pd.read_csv(self.open(self.filepath_or_buffer), **self.csv_kwds)
                except IOError as e:
                    logger.error('Cannot open file %s: %s', self.filepath_or_buffer, e)
                    self.parser = None
                    return self.state_terminated
                self.filepath_or_buffer = None
            else:
                fn_slot = self.get_input_slot('filenames')
                if fn_slot is None or fn_slot.output_module is None:
                    return self.state_terminated
                with fn_slot.lock:
                    fn_slot.update(run_number, self.id)
                    if fn_slot.deleted.any() or fn_slot.updated.any():
                        raise ProgressiveError('Cannot handle input file changes')
                    df = fn_slot.data()
                    while self.parser is None:
                        indices = fn_slot.created.next(1)
                        if indices.stop==indices.start:
                            return self.state_blocked
                        filename = df.at[indices.start, 'filename']
                        try:
                            self.parser = pd.read_csv(self.open(filename), **self.csv_kwds)
                        except IOError as e:
                            logger.error('Cannot open file %s: %s', filename, e)
                            self.parser = None
                        # fall through
        return self.state_ready

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
            self.close()
            raise StopIteration('Unexpected situation')
        logger.info('loading %d lines', step_size)
        try:
            with self.lock:
                df = self.parser.read(step_size) # raises StopIteration at EOF
        except StopIteration:
            self.close()
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
                force_valid_id_columns(df)
            with self.lock:
                if self._table is None:
                    self._table_params['name'] = self.generate_table_name('table')
                    self._table_params['dshape'] = dshape_from_dataframe(df)
                    self._table_params['data'] = df
                    self._table_params['create'] = True
                    self._table = Table(**self._table_params)
                else:
                    self._table.append(df)
        #print("Progress: ", self.get_progress())
        return self._return_run_step(self.state_ready, steps_run=creates)

def extract_params_docstring(fn, only_defaults=False):
    defaults = fn.__defaults__
    varnames = fn.__code__.co_varnames
    argcount = fn.__code__.co_argcount
    nodefcount = argcount - len(defaults)
    reqargs = ",".join(varnames[0:nodefcount])
    defargs = ",".join(["%s=%s"%(varval[0], repr(varval[1])) for varval in zip(varnames[nodefcount:argcount], defaults)])
    if only_defaults:
        return defargs
    if not reqargs:
        return defargs
    if not defargs:
        return reqargs
    return reqargs+","+defargs

csv_docstring = "CSVLoader(" \
  + extract_params_docstring(pd.read_csv) \
  + ","+extract_params_docstring(CSVLoader.__init__, only_defaults=True) \
  + ",force_valid_ids=False,id=None,scheduler=None,tracer=None,predictor=None,storage=None,input_descriptors=[],output_descriptors=[])"
try:
    CSVLoader.__init__.__func__.__doc__ = csv_docstring
except:
    try:
        CSVLoader.__init__.__doc__ = csv_docstring
    except:
        pass

