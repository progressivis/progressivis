from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from ..core.utils import (filepath_to_buffer,
                              _infer_compression, is_str)
from requests.packages.urllib3.exceptions import HTTPError
from io import BytesIO
import time
import bz2
import zlib
import lzma
from collections import OrderedDict
from pandas.core.dtypes.inference import is_file_like, is_sequence

SAMPLE_SIZE = 5
MARGIN = 0.05
MAX_RETRY = 3
NL = b'\n'
ROW_MAX_LENGTH_GUESS = 10000
#DEBUG_CNT = 0
from functools import partial

decompressors = dict(bz2=bz2.BZ2Decompressor,
                       zlib=zlib.decompressobj,
                       gzip=partial(zlib.decompressobj, wbits=zlib.MAX_WBITS|16),
                       xz=lzma.LZMADecompressor
                       )

def is_recoverable(inp):
    if is_str(inp):
        return True
    if not is_sequence(inp):
        return False
    types = set((type(e) for e in inp))
    if len(types) != 1:
        raise ValueError("All inputs must have the same type")
    return is_str(inp[0])


class Parser(object):
    def __init__(self, input_source, remaining, estimated_row_size,
                     offset=None, overflow_df=None, last_row=0, pd_kwds={}):
        self._input = input_source
        self._pd_kwds = pd_kwds
        self._remaining = remaining
        self._estimated_row_size = estimated_row_size
        self._overflow_df = overflow_df
        self._offset = self._input.tell() if offset is None else offset
        self._last_row = last_row
        self._recovery_cnt = 0

    def get_snapshot(self, run_number, last_id, table_name):
        if not is_recoverable(self._input._seq):
            raise ValueError("Not recoverable")
        ret = OrderedDict(
                file_seq='`'.join(self._input._seq),
                file_cnt = self._input._file_cnt,
                encoding="" if self._input._encoding is None else self._input._encoding,
                compression="" if self._input._compression is None else self._input._compression,
                remaining=self._remaining.decode('utf-8'),
                overflow_df= "" if self._overflow_df is None else self._overflow_df.to_csv(),
                offset=self._offset - len(self._input._dec_remaining),
                last_row=self._last_row,
                estimated_row_size=self._estimated_row_size,
                run_number=run_number,
                last_id=last_id,
                table_name=table_name
            )
        ret.update(check=hash(tuple(ret.values())))
        return ret

    def read(self, n):
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
                print("previous overflow partly consumed : ", d, " rows")
                return ret
            #else
            print("previous overflow entirely consumed: ", len_df, " rows")
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
                bytes_ = self._input.read(size) # do not raise StopIteration, only returns b''
            except HTTPError:
                print("HTTPError ...", self._offset)
                if retries >= MAX_RETRY:
                    raise
                retries += 1
                self._recovery_cnt += 1
                time.sleep(1)
                self._input.reopen(self._offset)
                n_ = recovery_n
                print("... recovery")
                continue
            self._offset = self._input.tell()
            if not bytes_ and not self._remaining:
                break
            last_nl = bytes_.rfind(NL) # stop after the last NL
            csv_bytes = self._remaining+bytes_[:last_nl+1]
            self._remaining = bytes_[last_nl+1:]
            read_df = pd.read_csv(BytesIO(csv_bytes), **self._pd_kwds)
            len_df = len(read_df)
            self._estimated_row_size = len(csv_bytes)//len_df
            if len_df <= n_:
                ret.append(read_df)
                row_cnt += len_df
                self._last_row += len_df
            else: # overflow (we read to much lines)
                d = len_df - n_
                tail = read_df.tail(d)
                read_df.drop(tail.index,inplace=True)
                ret.append(read_df)
                self._overflow_df = tail
                print("produced overflow: ", len(tail), "rows")
                self._last_row += d
                break
        return ret

class InputSource(object):
    def __init__(self, inp, encoding, file_cnt=0, compression=None, dec_remaining=b'', timeout=None, start_byte=0):
        """
        NB: all inputs are supposed to have the same type, encoding, compression
        TODO: check that for encoding and compression
        """
        #if self._input_stream is not None:
        #    self.close()
        if is_str(inp) or is_file_like(inp):
            inp = [inp]
        if not is_sequence(inp):
            raise ValueError("Bad input")
        types = set((type(e) for e in inp))
        if len(types) != 1:
            raise ValueError("All inputs must have the same type")
        f = inp[0]
        if not (is_str(f) or is_file_like(f)):
            raise ValueError("input type not supported")
        #if compression is not None and start_byte:
        #    raise ValueError("Cannot open a compressed file with a positive offset")
        if file_cnt >= len(inp):
            raise ValueError("File counter out of range")
        self._seq = inp
        self._file_cnt = file_cnt
        compression = _infer_compression(self.filepath, compression)
        offs = 0 if compression else start_byte
        istream, encoding, compression, size = filepath_to_buffer(self.filepath,
                                                                  encoding=encoding,
                                                                  compression=compression,
                                                                    timeout=timeout,
                                                                      start_byte=offs)
        self._encoding = encoding
        self._compression = compression
        self._input_size = size
        self._timeout = None # for tests
        self._decompressor_class = None
        self._decompressor = None
        self._dec_remaining = dec_remaining
        self._dec_offset = 0
        #self._compressed_offset = 0
        self._stream = istream
        if self._compression is not None:
            self._decompressor_class = decompressors[self._compression]
            self._decompressor = self._decompressor_class()
            self._read_compressed(start_byte) #seek


    @property
    def filepath(self):
        return self._seq[self._file_cnt]

    def switch_to_next(self):
        """
        """
        if self._file_cnt >= len(self._seq)-1:
            return False
        self._file_cnt += 1        
        istream, encoding, compression, size = filepath_to_buffer(self.filepath,
                                                                  encoding=self._encoding,
                                                                  compression=self._compression)
        if self._encoding != encoding:
            raise ValueError("all files must have the same encoding")
        if self._compression != compression:
            raise ValueError("all files must have the same compression")
        self._input_size = size
        self._timeout = None # for tests
        self._decompressor_class = None
        self._decompressor = None
        self._dec_remaining = b''
        self._dec_offset = 0
        #self._compressed_offset = 0
        if self._compression is not None:
            self._decompressor_class = decompressors[self._compression]
            self._decompressor = self._decompressor_class()
        self._stream = istream
        return True

    def reopen(self, start_byte=0):
        if self._stream is not None:
            self.close()
        if self._compression is None:
            istream, encoding, compression, size = filepath_to_buffer(filepath=self.filepath,
                                                                  encoding=self._encoding,
                                                                  compression=self._compression,
                                                                    timeout=self._timeout,
                                                                      start_byte=start_byte)
            self._stream = istream
            return istream
        istream, encoding, compression, size = filepath_to_buffer(filepath=self.filepath,
                                                                encoding=self._encoding,
                                                                compression=self._compression,
                                                                timeout=self._timeout,
                                                                      start_byte=0)
        self._stream = istream
        self._decompressor = self._decompressor_class()
        if self._dec_offset != start_byte:
            raise ValueError("PB: {}!={}".format(self._dec_offset, start_byte))
        self._read_compressed(start_byte) #seek
        return istream

    def tell(self):
        if self._compression is None:
            return self._stream.tell()
        else:
            return self._dec_offset

    def read(self, n):
        if self._compression is None:
            ret = self._stream.read(n)
        else:
            ret = self._read_compressed(n)
        if ret or not n:
            return ret
        if self.switch_to_next():
            return self.read(n)
        else:
            return b''

    def _read_compressed(self, n):
        len_remaining = len(self._dec_remaining)
        if n <= len_remaining:
            ret = self._dec_remaining[:n]
            self._dec_remaining = self._dec_remaining[n:]
            return ret
        # here n > len_remaining
        n_ = n - len_remaining
        buff = BytesIO()
        buff.write(self._dec_remaining)
        cnt = 0
        break_ = False
        while cnt < n_:
            max_length = 1024 * 100
            chunk =self._stream.read(int(max_length))
            if not len(chunk):
                break_ = True
            bytes_ = b''
            try:
                bytes_ = self._decompressor.decompress(chunk)
            except EOFError:
                break_ = True
            buff.write(bytes_)
            len_bytes = len(bytes_)
            cnt += len_bytes
            if break_: break
        self._dec_offset += cnt
        ret = buff.getvalue()
        self._dec_remaining = b''
        if len(ret) > n:
            self._dec_remaining = ret[n:]
            ret = ret[:n]
        return ret

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
    pd_kwds = dict(csv_kwds)
    chunksize = pd_kwds['chunksize']
    del pd_kwds['chunksize']
    #pd_kwds['encoding'] = input_source._encoding
    first_row = get_first_row(input_source)
    return Parser(input_source, remaining=first_row, estimated_row_size=len(first_row), pd_kwds=pd_kwds)

def recovery(snapshot, previous_file_seq, **csv_kwds):
    file_seq = snapshot['file_seq'].split('`')
    if is_str(previous_file_seq):
        previous_file_seq = [previous_file_seq]
    if previous_file_seq!= file_seq[:len(previous_file_seq)]: # we tolerate a new file_seq longer than the previous
        raise ValueError("File sequence changed, recovery aborted!")
    file_cnt = snapshot['file_cnt']
    encoding = snapshot['encoding']
    if not encoding:
        encoding = None
    compression = snapshot['compression']
    if not compression:
        compression = None
    remaining = snapshot['remaining'].encode('utf-8')
    overflow_df = snapshot['overflow_df'].encode('utf-8')
    offset = snapshot['offset']
    last_row = snapshot['last_row']
    estimated_row_size = snapshot['estimated_row_size']
    last_id = snapshot['last_id']
    #dec_remaining = snapshot['dec_remaining'].encode('utf-8')
    if overflow_df:
        overflow_df = pd.read_csv(overflow_df)
    input_source = InputSource(file_seq, encoding=encoding, compression=compression, file_cnt=file_cnt, start_byte=offset, timeout=None)
    pd_kwds = dict(csv_kwds)
    chunksize = pd_kwds['chunksize']
    del pd_kwds['chunksize']
    return Parser(input_source, remaining=remaining,
                        estimated_row_size=estimated_row_size,
                        offset=offset, overflow_df=None,
                        last_row=last_row, pd_kwds=pd_kwds)
