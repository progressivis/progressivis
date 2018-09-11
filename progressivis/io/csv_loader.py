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
from .read_csv import read_csv, recovery, InputSource

class CSVLoader(TableModule):
    def __init__(self,
                 filepath_or_buffer=None,
                 filter_=None,
                 force_valid_ids=True,
                 fillvalues=None,
                 timeout=None,
                 save_context=True,
                 recovery=0,
                 recovery_table_size=5,
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
        self._timeout = timeout
        self._table_params = dict(name=self.name, fillvalues=fillvalues)
        self._save_context = save_context
        self._recovery = recovery
        self._recovery_table_size = recovery_table_size
        self._recovery_table = None
        self._recovered_csv_table_name = None
        self._table = None


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

    def create_input_source(self, filepath):
        return InputSource(filepath, encoding=self._encoding,
                               compression=self._compression,
                               timeout=self._timeout, start_byte=0)

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
                if not self._recovery:
                    try:
                        self.parser = read_csv(self.create_input_source(self.filepath_or_buffer), **self.csv_kwds)
                    except IOError as e:
                        logger.error('Cannot open file %s: %s', self.filepath_or_buffer, e)
                        self.parser = None
                        return self.state_terminated
                    self.filepath_or_buffer = None
                else:
                    try:
                        if self._recovery_table is None:
                            self._recovery_table = Table(name='csv_loader_recovery', create=False)
                    except Exception as e: # TODO: specify the exception?
                        logger.error('Cannot acces recovery table', e)
                        return self.state_terminated
                    try:
                        last_ = self._recovery_table.argmax()['offset'] 
                        snapshot = self._recovery_table.loc[last_].to_dict()
                        self._recovered_csv_table_name = snapshot['table_name']
                    except Exception as e:
                        logger.error('Cannot read the snapshot', e)
                        return self.state_terminated
                    try:
                        self.parser = recovery(snapshot, self.filepath_or_buffer, **self.csv_kwds)
                    except Exception as e:
                        logger.error('Cannot recover from snapshot {}', snapshot, e)
                        self.parser = None
                        return self.state_terminated
                    self.filepath_or_buffer = None

            else:
                fn_slot = self.get_input_slot('filenames')
                if fn_slot is None or fn_slot.output_module is None:
                    return self.state_terminated
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
                            self.parser = read_csv(self.create_input_source(filename), **self.csv_kwds)
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
        #print("Processed bytes: ", self.parser._prev_pos)
        try:
            with self.lock:
                df_list = self.parser.read(step_size) # raises StopIteration at EOF
                if not df_list:
                    raise StopIteration
        except StopIteration:
            self.close()
            fn_slot = self.get_input_slot('filenames')
            if fn_slot is None or fn_slot.output_module is None:
                raise
            self.parser = None
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        df_len = sum([len(df) for df in df_list])
        creates = df_len
        if creates == 0: # should not happen
            logger.error('Received 0 elements')
            raise StopIteration
        if self._filter != None:
            df_list =[self._filter(df) for df in df_list]
        creates = sum([len(df) for df in df_list])
        if creates == 0:
            logger.info('frame has been filtered out')
        else:
            self._rows_read += creates
            logger.info('Loaded %d lines', self._rows_read)
            if self.force_valid_ids:
                for df in df_list:
                    force_valid_id_columns(df)
            with self.lock:
                if self._table is None:
                    if not self._recovery:
                        self._table_params['name'] = self.generate_table_name('table')
                        self._table_params['dshape'] = dshape_from_dataframe(df_list[0])
                        self._table_params['create'] = True
                        self._table_params['data'] = pd.concat(df_list)
                        self._table = Table(**self._table_params)
                    else:
                        self._table_params['name'] = self._recovered_csv_table_name
                        self._table_params['create'] = False
                        self._table = Table(**self._table_params)
                        self._table.append(pd.concat(df_list))
                else:
                    for df in df_list:
                        self._table.append(df)
                if self._recovery_table is None and self._save_context:
                    snapshot = self.parser.get_snapshot(run_number=run_number, table_name=self._table._name,
                                                last_id=self._table.last_id)
                    #print(snapshot)
                    import pdb;pdb.set_trace()
                    self._recovery_table = Table(name='csv_loader_recovery', data=pd.DataFrame(snapshot, index=[0]), create=True)

                elif self._save_context:
                        snapshot = self.parser.get_snapshot(
                            run_number=run_number,
                            last_id=self._table.last_id, table_name=self._table._name)
                        self._recovery_table.add(snapshot)
                        if len(self._recovery_table) > self._recovery_table_size:
                            oldest = self._recovery_table.argmin()['offset']
                            self._recovery_table.drop(np.argmin(oldest))
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

