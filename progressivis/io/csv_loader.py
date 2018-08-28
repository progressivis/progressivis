from __future__ import absolute_import, division, print_function

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from io import BytesIO
from progressivis import ProgressiveError, SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..table.dshape import dshape_from_dataframe
from ..core.utils import filepath_to_buffer, _infer_compression, force_valid_id_columns
from requests.packages.urllib3.exceptions import HTTPError
import bz2

class CSVLoader(TableModule):
    def __init__(self,
                 filepath_or_buffer=None,
                 filter_=None,
                 force_valid_ids=True,
                 fillvalues=None,
                 timeout=None,
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
        self._processed_bytes = 0
        self._flushing_stuff = None
        self._current_url = None
        self._recovery = False
        self._recovery_cnt = 0
        self._timeout = timeout
        self._table_params = dict(name=self.name, fillvalues=fillvalues)
        self._table = None

    def assign_bytes_cnt(self):
        self._processed_bytes = self.parser.f.cnt if hasattr(self.parser.f, 'cnt') else  None

    def rows_read(self):
        return self._rows_read

    def is_ready(self):
        fn = self.get_input_slot('filenames')
        # Can be called before the first update so fn.created can be None
        if fn and (fn.created is None or fn.created.any()):
            return True
        return super(CSVLoader, self).is_ready()
    
    def is_data_input(self):
        # pylint: disable=no-self-use
        "Return True if this module brings new data"
        return True

    def open(self, filepath):
        if self._input_stream is not None:
            self.close()
        compression = _infer_compression(filepath, self._compression)
        flushing_str = self.stringify_flushing_stuff()
        istream, encoding, compression, size = filepath_to_buffer(filepath,
                                                                  encoding=self._encoding,
                                                                  compression=compression,
                                                                    timeout=self._timeout,
                                                                      start_byte=self._processed_bytes,
                                                                      flushing_str=flushing_str)
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
    def _reset_recovery(self):
        self._processed_bytes = 0
        self._flushing_stuff = None
        self._recovery = False

    def stringify_flushing_stuff(self):
        """
        the flushing stuff structure is: tuple(tuple(None, list, dict), delimitator)
        """
        if self._flushing_stuff is None:
            return None
        dict_, del_ = self._flushing_stuff
        cols = list(dict_.values())
        col_index = dict_.keys()
        size = len(cols[0])
        print("RECOVER ", size, " LINES!")
        acc = BytesIO()
        has_nan = False
        for i in range(size):
            if i>0:
                acc.write(b'\n')
            for j in col_index:
                v = cols[j][i] 
                if isinstance(v, float) and np.isnan(v):
                    has_nan = True
                    continue # TODO: consider the case when data contains "true Nans"
                if j>0:
                    acc.write(del_)
                acc.write(str(v).encode('utf-8'))
        if not has_nan:
            print("no Nan-s ...")
            acc.write(b'\n')
        #else:
        #    acc.write(del_)
        ret = acc.getvalue()
        acc.close()
        return ret

    def validate_parser(self, run_number):
        if self.parser is None:
            if self.filepath_or_buffer is not None:
                try:
                    self.parser = pd.read_csv(self.open(self.filepath_or_buffer), **self.csv_kwds)
                    self._recovery_cnt += 1
                    if self._recovery:
                        print("HTTPError recovered!")
                except IOError as e:
                    logger.error('Cannot open file %s: %s', self.filepath_or_buffer, e)
                    self.parser = None
                    return self.state_terminated
                self._current_url = self.filepath_or_buffer
                self.filepath_or_buffer = None
                self._reset_recovery()
            else:
                fn_slot = self.get_input_slot('filenames')
                if fn_slot is None or fn_slot.output_module is None:
                    return self.state_terminated
                if self._recovery:
                    try:
                        print("recovery: ", self._current_url)
                        self.parser = pd.read_csv(self.open(self._current_url), **self.csv_kwds)
                    except IOError as e:
                            logger.error('Cannot open file %s: %s', filename, e)
                            self.parser = None
                    return
                with fn_slot.lock:
                    fn_slot.update(run_number)
                    if fn_slot.deleted.any() or fn_slot.updated.any():
                        raise ProgressiveError('Cannot handle input file changes')
                    df = fn_slot.data()
                    while self.parser is None:
                        indices = fn_slot.created.next(1)
                        if indices.stop==indices.start:
                            return self.state_blocked
                        filename = df.at[indices.start, 'filename']
                        try:
                            print("read_csv: ", filename)
                            self.parser = pd.read_csv(self.open(filename), **self.csv_kwds)
                        except IOError as e:
                            logger.error('Cannot open file %s: %s', filename, e)
                            self.parser = None
                        self._current_url = filename
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
        self.assign_bytes_cnt()
        #print("Processed bytes: ", self._processed_bytes)
        try:
            with self.lock:
                df = self.parser.read(step_size) # raises StopIteration at EOF
        except StopIteration:
            self._processed_bytes = 0
            self._reset_recovery()
            self.close()
            fn_slot = self.get_input_slot('filenames')
            if fn_slot is None or fn_slot.output_module is None:
                raise
            self.parser = None
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        except HTTPError as e:
            self._recovery = True
            print("HTTPError ...")
            if self._input_compression is not None:
                raise
            stuff = self.parser._engine._reader.read()
            self.assign_bytes_cnt()
            self._flushing_stuff = (stuff, self.parser._engine._reader.delimiter.encode('utf-8'))
            self._timeout = None
            self.parser = None
            self.filepath_or_buffer = self._current_url
            return self._return_run_step(self.state_blocked, steps_run=0, creates=0)            
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

