import pandas as pd
import numpy as np
from ..core.utils import filepath_to_buffer, _infer_compression
from requests.packages.urllib3.exceptions import HTTPError
from io import BytesIO
import time

SAMPLE_SIZE = 5
MARGIN = 0.05
MAX_RETRY = 3
NL = b'\n'
ROW_MAX_LENGTH_GUESS = 10000

class Parser(object):
    def __init__(self, input_source, first_line, pd_kwds):
        self._input = input_source
        self._pd_kwds = pd_kwds
        self._oddment = first_line
        self._estimated_row_size = len(first_line)
        self._overflow_df = None
        self._prev_pos = self._input._stream.tell()
        self._last_row = 0
        self._recovery_cnt = 0
    def read(self, n):
        #import pdb;pdb.set_trace()
        ret = []
        n_ = n 
        if self._overflow_df is not None:
            len_df = len(self._overflow_df)
            #assert len_df < n
            if len_df > n:
                d = len_df - n
                tail = self._overflow_df.tail(d)
                self._overflow_df.drop(tail.index,inplace=True)
                ret.append(self._overflow_df)
                self._overflow_df = tail
                self._last_row += n
                print("consumed overflow: ", d)
                return ret
            #else
            print("consumed overflow: ", len_df)            
            self._last_row += len_df
            n_ = n - len_df
            ret.append(self._overflow_df)
            self._overflow_df = None
            if n - len_df < n*MARGIN: # almost equals
                return ret
        # it remains n_ rows to read
        row_cnt = 0
        at_least_n = int(n_*(1-MARGIN))
        retries = 0
        while row_cnt < at_least_n:
            row_size = self._estimated_row_size
            recovery_n = n_
            n_ = n_ - row_cnt 
            size = n_ * row_size
            try:
                bytes_ = self._input._stream.read(size) # do not raise StopIteration, only returns b''
            except HTTPError:
                print("HTTPError ...")
                if retries >= MAX_RETRY:
                    raise
                retries += 1
                self._recovery_cnt += 1
                time.sleep(1)
                #import pdb;pdb.set_trace()
                self._input.reopen(self._prev_pos)
                n_ = recovery_n
                print("... recovery")
                continue
            self._prev_pos = self._input._stream.tell()
            if not bytes_ and not self._oddment:
                break
            last_nl = bytes_.rfind(NL) # stop after the last NL
            csv_bytes = self._oddment+bytes_[:last_nl+1]
            self._oddment = bytes_[last_nl+1:]
            read_df = pd.read_csv(BytesIO(csv_bytes), **self._pd_kwds)
            len_df = len(read_df)
            self._estimated_row_size = len(csv_bytes)//len_df
            if len_df <= n_:
                ret.append(read_df)
                row_cnt += len_df
                self._last_row += len_df
                self._last_byte = self._input._stream.tell()                    
            else: # overflow (we read to much lines)
                d = len_df - n_
                tail = read_df.tail(d)
                read_df.drop(tail.index,inplace=True)
                ret.append(read_df)
                self._overflow_df = tail
                print("create overflow: ", len(tail))
                self._last_row += d
                self._last_byte = self._input._stream.tell()
                break
                #import pdb;pdb.set_trace()
        
        return ret

class FakeParser(object):
    def __init__(self, obj):
        self._obj = obj
        self._prev_pos = 0
    def read(self, n=0):
        return [self._obj.read(n)]

class InputSource(object):
    def __init__(self, filepath, encoding, compression=None, timeout=None, start_byte=0):
        #if self._input_stream is not None:
        #    self.close()
        compression = _infer_compression(filepath, compression)
        istream, encoding, compression, size = filepath_to_buffer(filepath,
                                                                  encoding=encoding,
                                                                  compression=compression,
                                                                    timeout=timeout,
                                                                      start_byte=start_byte)
        self._filepath = filepath
        self._stream = istream
        self._encoding = encoding
        self._compression = compression
        self._input_size = size
        self._timeout = None # for tests
        self._start_byte = start_byte
        
    def reopen(self, start_byte=0):
        if self._stream is not None:
            self.close()
        istream, encoding, compression, size = filepath_to_buffer(filepath=self._filepath,
                                                                  encoding=self._encoding,
                                                                  compression=self._compression,
                                                                    timeout=self._timeout,
                                                                      start_byte=start_byte)
        self._start_byte = start_byte
        self._stream = istream
        return istream
    def close(self):
        if self._stream is None:
            return
        try:
            self._stream.close()
            # pylint: disable=bare-except
        except:
            pass
        self._stream = None
        self._input_encoding = None
        self._input_compression = None
        self._input_size = 0
    
def get_first_row(fd):
    ret = BytesIO()
    guard = ROW_MAX_LENGTH_GUESS
    for _ in range(guard):
        c = fd.read(1)
        ret.write(c)
        if c == b'\n':
            break
    else:
        print("Warning: row longer than {}".format(guard))
    return ret.getvalue()

def read_csv(input_source, silent_before=0, **csv_kwds):
    #import pdb;pdb.set_trace()    
    #if not isinstance(filepath_or_buffer, HttpDesc):
    #    return FakeParser(pd.read_csv(filepath_or_buffer, **csv_kwds))
    istream = input_source._stream
    pd_kwds = dict(csv_kwds)
    chunksize = pd_kwds['chunksize']
    del pd_kwds['chunksize']
    pd_kwds['encoding'] = input_source._encoding
    pd_kwds['compression'] = input_source._compression
    #import pdb;pdb.set_trace() 
    first_line = get_first_row(istream)
    return Parser(input_source, first_line, pd_kwds)

        

