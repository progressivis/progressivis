from __future__ import annotations

from io import BytesIO, IOBase
import time
import bz2
import zlib
import json
from collections import OrderedDict
import functools as ft

from requests.packages.urllib3.exceptions import HTTPError
import numpy as np
import pandas as pd
from pandas.core.dtypes.inference import is_file_like, is_sequence
import lzma

from progressivis.core.utils import filepath_to_buffer, _infer_compression, is_str

from typing import (
    Optional,
    Union,
    Any,
    Dict,
    List,
    Tuple,
    cast,
)


SAMPLE_SIZE = 5
MARGIN = 0.05
MAX_RETRY = 3
NL = b"\n"
SEP = ","
ROW_MAX_LENGTH_GUESS = 10000
HEADER_CHUNK = 50


class Decompressor:
    def decompress(self, chunk: bytes) -> bytes:
        ...


class NoDecompressor(Decompressor):
    def decompress(self, chunk: bytes) -> bytes:
        return chunk


decompressors: Dict[str, Any] = dict(
    bz2=bz2.BZ2Decompressor,
    zlib=zlib.decompressobj,
    gzip=ft.partial(zlib.decompressobj, wbits=zlib.MAX_WBITS | 16),
    xz=lzma.LZMADecompressor,
)


def is_recoverable(inp: Any) -> bool:
    if is_str(inp):
        return True
    if not is_sequence(inp):
        return False
    types = set((type(e) for e in inp))
    if len(types) != 1:
        raise ValueError("All inputs must have the same type")
    return is_str(inp[0])


class Parser:
    """
    Always use Parser.create() instead of Parser() because __init__() is not awaitable
    """

    def __init__(
        self,
        input_source: InputSource,
        remaining: bytes,
        estimated_row_size: int,
        offset: int = 0,
        overflow_df: Optional[pd.DataFrame] = None,
        pd_kwds: Dict[str, Any] = {},
        chunksize: int = 0,
        usecols: Optional[List[str]] = None,
        names: Optional[Union[np.ndarray[Any, Any], List[str]]] = None,
        header: Union[None, str, int] = "infer",
    ):
        self._input = input_source
        self._pd_kwds = pd_kwds
        self._remaining = remaining
        self._estimated_row_size = estimated_row_size
        self._overflow_df: Optional[pd.DataFrame] = overflow_df
        self._offset: int = offset
        self._recovery_cnt: int = 0
        self._nb_cols: Optional[int] = None if overflow_df is None else len(
            overflow_df.columns
        )
        self._chunksize: int = chunksize
        self._usecols: Optional[List[str]] = usecols
        self._names: Optional[Union[np.ndarray[Any, Any], List[str]]] = names
        self._header: Union[None, str, int] = header

    @staticmethod
    def create(
        input_source: InputSource,
        remaining: bytes,
        estimated_row_size: int,
        offset: Optional[int] = None,
        overflow_df: Optional[pd.DataFrame] = None,
        pd_kwds: Dict[str, Any] = {},
        chunksize: int = 0,
        usecols: Optional[List[str]] = None,
        names: Optional[Union[np.ndarray[Any, Any], List[str]]] = None,
        header: Union[None, str, int] = "infer",
    ) -> Parser:
        par = Parser(
            input_source,
            remaining,
            estimated_row_size,
            input_source.tell() if offset is None else offset,
            overflow_df,
            pd_kwds,
            chunksize,
            usecols,
            names,
            header,
        )
        return par

    def get_snapshot(
        self, run_number: int, last_id: int, table_name: str
    ) -> Dict[str, Any]:
        if not is_recoverable(self._input._seq):
            raise ValueError("Not recoverable")
        ret = OrderedDict(
            file_seq=json.dumps(self._input._seq),
            file_cnt=self._input._file_cnt,
            encoding=json.dumps(self._input._encoding),
            compression=json.dumps(self._input._compression),
            remaining=self._remaining.decode("utf-8"),
            overflow_df=(
                ""
                if self._overflow_df is None
                else self._overflow_df.to_csv(index=False, header=False)
            ),
            offset=self._offset - len(self._input._dec_remaining),
            estimated_row_size=self._estimated_row_size,
            nb_cols=self._nb_cols,
            run_number=run_number,
            last_id=last_id,
            table_name=table_name,
            names=json.dumps(
                self._names.tolist()
                if isinstance(self._names, np.ndarray)
                else self._names
            ),
            usecols=json.dumps(self._usecols),
        )
        ret.update(check=hash(tuple(ret.values())))
        return ret

    def read(self, n: int, flush: bool = False) -> List[pd.DataFrame]:
        assert n > 0
        ret: List[pd.DataFrame] = []
        n_ = n
        if self._overflow_df is not None:
            len_df = len(self._overflow_df)
            # assert len_df < n
            if len_df > n:
                ret.append(self._overflow_df.iloc[:n])
                self._overflow_df = self._overflow_df.iloc[n:]
                # print("previous overflow partly consumed : ", n, " rows")
                return ret
            # else
            # print("previous overflow entirely consumed: ", len_df, " rows")
            n_ = n - len_df
            ret.append(self._overflow_df)
            self._overflow_df = None
            if n_ < n * MARGIN:  # almost equals
                return ret
        assert n_ > 0
        # it remains n_ rows to read
        row_cnt = 0
        # at_least_n = int(n_*(1-MARGIN))
        retries = 0
        while row_cnt < n_:  # at_least_n:
            row_size = self._estimated_row_size
            recovery_n = n_
            n_ = n_ - row_cnt
            if flush:
                nb_rows = n_
            elif self._names is None and self._usecols is not None:
                # try not to read a lot of rows because we are going to read
                # all columns, in order to know their list
                nb_rows = min(n_, HEADER_CHUNK)
            else:
                nb_rows = max(n_, self._chunksize)
            size = nb_rows * row_size
            try:
                # do not raise StopIteration, only returns b''
                new_csv, bytes_ = self._input.read(size)
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
                break  # end of file
            last_nl = bytes_.rfind(NL)  # stop after the last NL
            if last_nl == -1:  # NL not found => we read less than an entire row
                self._remaining += bytes_
                continue
            csv_bytes = self._remaining + bytes_[: last_nl + 1]
            self._remaining = bytes_[last_nl + 1 :]
            if not csv_bytes:
                break
            if new_csv or self._names is None:
                header = self._header
                if (
                    self._header in (0, "infer")
                    and self._names is None
                    and self._usecols
                    and (
                        not callable(self._usecols)  # TODO not sure what to do
                        and isinstance(self._usecols[0], str)
                    )
                ):
                    # csv_bytes begins with the column names
                    first_row_size = csv_bytes.find(b"\n") + 1
                    self._names = pd.read_csv(
                        BytesIO(csv_bytes[:first_row_size])
                    ).columns.values
                names = None
            else:
                header = None
                names = self._names
            kwds = {
                k: v
                for (k, v) in self._pd_kwds.items()
                if k not in ["header", "names", "usecols"]
            }
            read_df: pd.DataFrame = pd.read_csv(
                BytesIO(csv_bytes),
                header=header,
                names=names,  # type: ignore
                usecols=self._usecols,  # type: ignore
                **kwds
            )
            if self._names is None:
                self._names = read_df.columns.values
                if self._usecols:
                    if callable(self._usecols):
                        f_ = self._usecols
                        self._usecols = [c for c in read_df.columns if f_(c)]
                    read_df = read_df.loc[:, self._usecols]
            if self._nb_cols is None:
                self._nb_cols = len(read_df.columns)
            elif self._nb_cols != len(read_df.columns):
                raise ValueError(
                    "Wrong number of cols "
                    "{} instead of {}".format(len(read_df.columns), self._nb_cols)
                )
            len_df = len(read_df)
            if len_df:
                self._estimated_row_size = len(csv_bytes) // len_df
            if len_df <= n_:
                ret.append(read_df)
                row_cnt += len_df
            else:  # overflow (we read too much lines)
                self._overflow_df = read_df.iloc[n_:]
                ret.append(read_df.iloc[:n_])
                break
        return ret

    def is_flushed(self) -> bool:
        return self._overflow_df is None


class InputSource:
    """
    Always use InputSource.create() instead of InputSource() because __init__() is not awaitable
    """

    def __init__(
        self,
        inp: Any,
        encoding: Optional[str],
        file_cnt: int = 0,
        compression: Optional[str] = None,
        dec_remaining: bytes = b"",
        timeout: Optional[float] = None,
        start_byte: int = 0,
        usecols: Optional[List[str]] = None,
    ):
        """
        NB: all inputs are supposed to have the same type, encoding, compression
        TODO: check that for encoding and compression
        """
        # if self._input_stream is not None:
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
        # if compression is not None and start_byte:
        #    raise ValueError("Cannot open a compressed file with a positive offset")
        if file_cnt >= len(inp):
            raise ValueError("File counter out of range")
        self._seq: List[str] = inp
        self._file_cnt: int = file_cnt
        self._usecols: Optional[List[str]] = usecols
        self._encoding: Optional[str]
        self._compression: Optional[str]
        self._input_size: int
        self._timeout: Optional[float] = None
        self._decompressor_class: Optional[type] = None
        self._decompressor: Optional[Decompressor] = None
        self._dec_remaining: bytes
        self._dec_offset: int
        self._stream: IOBase

    @staticmethod
    def create(
        inp: Any,
        encoding: Optional[str],
        file_cnt: int = 0,
        compression: Optional[str] = None,
        dec_remaining: bytes = b"",
        timeout: Optional[float] = None,
        start_byte: int = 0,
        usecols: Optional[List[str]] = None,
    ) -> InputSource:
        isrc = InputSource(
            inp,
            encoding,
            file_cnt,
            compression,
            dec_remaining,
            timeout,
            start_byte,
            usecols=usecols,
        )
        compression = _infer_compression(isrc.filepath, compression)
        offs = 0 if compression else start_byte

        istream, encoding, compression, size = filepath_to_buffer(
            isrc.filepath,
            encoding=encoding,
            compression=compression,
            timeout=timeout,
            start_byte=offs,
        )
        isrc._encoding = encoding
        isrc._compression = compression
        isrc._input_size = size
        isrc._timeout = None  # for tests
        isrc._decompressor_class = None
        isrc._decompressor = None
        isrc._dec_remaining = dec_remaining
        isrc._dec_offset = 0
        # isrc._compressed_offset = 0
        isrc._stream = istream
        if isrc._compression is not None:
            isrc._decompressor_class = decompressors[isrc._compression]
            isrc._decompressor = isrc._decompressor_class()
            isrc._read_compressed(start_byte)  # seek
        return isrc

    @property
    def filepath(self) -> str:
        return self._seq[self._file_cnt]

    def switch_to_next(self) -> bool:
        """
        """
        # print("Switch to next")
        if self._file_cnt >= len(self._seq) - 1:
            return False
        self._file_cnt += 1
        istream, encoding, compression, size = filepath_to_buffer(
            self.filepath, encoding=self._encoding, compression=self._compression
        )
        if self._encoding != encoding:
            raise ValueError("all files must have the same encoding")
        if self._compression != compression:
            raise ValueError("all files must have the same compression")
        self._input_size = size
        self._timeout = None  # for tests
        self._decompressor_class = None
        self._decompressor = None
        self._dec_remaining = b""
        self._dec_offset = 0
        # self._compressed_offset = 0
        if self._compression is not None:
            self._decompressor_class = decompressors[self._compression]
            self._decompressor = self._decompressor_class()
        self._stream = istream
        return True

    def reopen(self, start_byte: int = 0) -> IOBase:
        if self._stream is not None:
            self.close()
        if self._compression is None:
            istream, encoding, compression, size = filepath_to_buffer(
                filepath=self.filepath,
                encoding=self._encoding,
                compression=self._compression,
                timeout=self._timeout,
                start_byte=start_byte,
            )
            self._stream = istream
            self._input_size = size
            return istream
        istream, encoding, compression, size = filepath_to_buffer(
            filepath=self.filepath,
            encoding=self._encoding,
            compression=self._compression,
            timeout=self._timeout,
            start_byte=0,
        )
        self._stream = istream
        if self._decompressor_class:
            self._decompressor = cast(Decompressor, self._decompressor_class())
        else:
            self._decompressor = NoDecompressor()
        if self._dec_offset != start_byte:
            raise ValueError("PB: {}!={}".format(self._dec_offset, start_byte))
        self._read_compressed(start_byte)  # seek
        return istream

    def tell(self) -> int:
        if self._compression is None:
            return self._stream.tell()
        else:
            return self._dec_offset

    def read(self, n: int) -> Tuple[bool, bytes]:
        """
        returns new_file_flag, some_bytes
        in case of multi-file with headers
        is important to know when a new file begins in order to skip
        its header
        """
        if self._compression is None:
            ret = self._stream.read(n)
        else:
            ret = self._read_compressed(n)
        if ret or not n:
            return False, ret
        tell_ = self._stream.tell()
        if tell_ < self._input_size:
            raise ValueError(
                "Inconsistent read: empty string"
                " when position {} < size {}".format(tell_, self._input_size)
            )
        if self.switch_to_next():
            _, r = self.read(n)
            return True, r
        else:
            return False, b""

    def _read_compressed(self, n: int) -> bytes:
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
        assert self._decompressor is not None
        while cnt < n_:
            max_length = 1024 * 100
            chunk = self._stream.read(int(max_length))
            if not len(chunk):
                break_ = True
            bytes_ = b""
            try:
                bytes_ = self._decompressor.decompress(chunk)
            except EOFError:
                break_ = True
            buff.write(bytes_)
            len_bytes = len(bytes_)
            cnt += len_bytes
            if break_:
                break
        self._dec_offset += cnt
        ret = buff.getvalue()
        self._dec_remaining = b""
        if len(ret) > n:
            self._dec_remaining = ret[n:]
            ret = ret[:n]
        return ret

    def close(self) -> None:
        if self._stream.closed:
            return
        try:
            self._stream.close()
            # pylint: disable=bare-except
        except Exception:
            pass
        self._input_encoding = None
        self._input_compression = None
        self._input_size = 0


def get_first_row(fd: InputSource) -> bytes:
    ret = BytesIO()
    guard = ROW_MAX_LENGTH_GUESS
    for _ in range(guard):
        _, c = fd.read(1)
        ret.write(c)
        if c == b"\n":
            break
    else:
        print("Warning: row longer than {}".format(guard))
    return ret.getvalue()


def read_csv(
    input_source: InputSource, silent_before: int = 0, **csv_kwds: Any
) -> Parser:
    pd_kwds = dict(csv_kwds)
    chunksize = pd_kwds["chunksize"]
    del pd_kwds["chunksize"]
    # pd_kwds['encoding'] = input_source._encoding
    first_row = get_first_row(input_source)
    usecols = None
    if "usecols" in pd_kwds:
        usecols = pd_kwds.pop("usecols")
    header: Union[str, None, int] = "infer"
    if "header" in pd_kwds:
        header = pd_kwds.pop("header")
        assert header is None or header == 0
    return Parser.create(
        input_source,
        remaining=first_row,
        estimated_row_size=len(first_row),
        pd_kwds=pd_kwds,
        chunksize=chunksize,
        usecols=usecols,
        header=header,
    )


def recovery(
    snapshot: Dict[str, Any], previous_file_seq: Union[str, List[str]], **csv_kwds: Any
) -> Parser:
    print("RECOVERY ...")
    pd_kwds = dict(csv_kwds)
    chunksize = pd_kwds["chunksize"]
    del pd_kwds["chunksize"]
    file_seq = json.loads(snapshot["file_seq"])
    if isinstance(previous_file_seq, str):
        previous_file_seq = [previous_file_seq]
    if previous_file_seq != file_seq[: len(previous_file_seq)]:
        # we tolerate a new file_seq longer than the previous
        raise ValueError("File sequence changed, recovery aborted!")
    file_cnt = snapshot["file_cnt"]
    encoding = json.loads(snapshot["encoding"])
    compression = json.loads(snapshot["compression"])
    remaining = snapshot["remaining"].encode("utf-8")
    overflow_df = snapshot["overflow_df"].encode("utf-8")
    offset = snapshot["offset"]
    estimated_row_size = snapshot["estimated_row_size"]
    nb_cols = snapshot["nb_cols"]
    names = json.loads(snapshot["names"])
    usecols = json.loads(snapshot["usecols"])
    if usecols is not None:
        assert "usecols" in csv_kwds and csv_kwds["usecols"] == usecols
    # dec_remaining = snapshot['dec_remaining'].encode('utf-8')
    if overflow_df:
        overflow_df = pd.read_csv(BytesIO(overflow_df), **pd_kwds)
        if nb_cols != len(overflow_df.columns):
            raise ValueError(
                "Inconsistent snapshot: wrong number of cols in df {} instead of {}".format(
                    len(overflow_df.columns), nb_cols
                )
            )
    else:
        overflow_df = None
    input_source = InputSource.create(
        file_seq,
        encoding=encoding,
        compression=compression,
        file_cnt=file_cnt,
        start_byte=offset,
        timeout=None,
    )
    pd_kwds = dict(csv_kwds)
    chunksize = pd_kwds["chunksize"]
    del pd_kwds["chunksize"]
    # if 'header' in pd_kwds:
    #     header = pd_kwds.pop('header')
    return Parser.create(
        input_source,
        remaining=remaining,
        estimated_row_size=estimated_row_size,
        offset=offset,
        overflow_df=overflow_df,
        pd_kwds=pd_kwds,
        chunksize=chunksize,
        names=names,
        usecols=usecols,
        header=None,
    )
