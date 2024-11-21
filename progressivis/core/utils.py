from __future__ import annotations

import os
import os.path
import io
import bz2
import zlib
import lzma
import tempfile
import json as js
import re
from itertools import tee
from functools import wraps
import functools as ft
import pyarrow as pa
import threading
import inspect
from urllib.parse import urlparse as parse_url
from urllib.parse import parse_qs

import numpy as np
import keyword
import uuid
from .pintset import PIntSet
from ..core import aio

import collections.abc as collections_abc  # only works on python 3.3+

from multiprocessing import Process
from pandas.io.common import is_url  # type: ignore
import pandas as pd

import requests
import s3fs  # type: ignore
import stat as st


from typing import (
    Optional,
    Set,
    Union,
    Iterable,
    Any,
    Dict,
    List,
    Tuple,
    cast,
    Callable,
    Iterator,
    Sized,
    Awaitable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from progressivis.core.scheduler import Scheduler
    from progressivis.core.module import Module, ReturnRunStep
    from progressivis.table.table_base import BasePTable
    from progressivis.utils.psdict import PDict

    PatchRunStepCallable = Callable[[int, int, float], ReturnRunStep]

integer_types = (int, np.integer)
Items = Union[collections_abc.MutableMapping[Any, Any], List[Any], Any]


# def is_int(n: Any) -> TypeGuard[int]:
def is_int(n: Any) -> bool:
    return isinstance(n, integer_types)


# def is_str(s: Any) -> TypeGuard[str]:
def is_str(s: Any) -> bool:
    return isinstance(s, str)


# def is_dict(s: Any) -> TypeGuard[dict]:
def is_dict(s: Any) -> bool:
    return isinstance(s, dict)


# def is_slice(s: Any) -> TypeGuard[slice]:
def is_slice(s: Any) -> bool:
    return isinstance(s, slice)


# def is_iterable(it: Any) -> TypeGuard[Iterable[Any]]:
def is_iterable(it: Any) -> bool:
    return isinstance(it, Iterable)


# def is_iter_str(it: Any) -> TypeGuard[Iterable[str]]:
def is_iter_str(it: Any) -> bool:
    if not is_iterable(it):
        return False
    for s in it:
        if not is_str(s):
            return False
    return True


def nn(x: Any) -> bool:
    return x is not None


def len_none(item: Optional[Sized]) -> int:
    return 0 if item is None else len(item)


def pairwise(iterator: Iterator[Any]) -> Iterable[Any]:
    a, b = tee(iterator)
    next(b, None)
    return zip(a, b)


def is_sorted(
    iterator: Iterator[Any], compare: Optional[Callable[[Any, Any], bool]] = None
) -> bool:
    if compare is None:

        def _compare(a: Any, b: Any) -> bool:
            return cast(bool, a <= b)

        compare = _compare

    return all(compare(a, b) for a, b in pairwise(iterator))


def remove_nan(d: Items) -> Any:
    if isinstance(d, float) and np.isnan(d):
        return None
    if isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, float) and np.isnan(v):
                d[i] = None
            else:
                remove_nan(cast(Items, v))
    elif isinstance(d, collections_abc.MutableMapping):
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
            else:
                remove_nan(cast(Items, v))
    return d


def find_nan_etc(d: Any) -> None:
    if isinstance(d, float) and np.isnan(d):
        return
    if isinstance(d, np.integer):
        print("numpy int: %s" % (d))
        return
    if isinstance(d, np.bool_):
        return
    if isinstance(d, np.ndarray):
        print("numpy array: %s" % (d))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, float) and np.isnan(v):
                print("numpy nan at %d in: %s" % (i, d))
            elif isinstance(v, np.integer):
                print("numpy int: %s at %d in %s" % (v, i, d))
            elif isinstance(v, np.bool_):
                print("numpy bool: %d in %s" % (i, d))
            elif isinstance(v, np.ndarray):
                print("numpy array: %d in %s" % (i, d))
            else:
                find_nan_etc(v)
    elif isinstance(d, collections_abc.Mapping):
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                print("numpy nan at %s in: %s" % (k, d))
            elif isinstance(v, np.integer):
                print("numpy int: %s in %s" % (k, d))
            elif isinstance(v, np.bool_):
                print("numpy bool: %s in %s" % (k, d))
            elif isinstance(v, np.ndarray):
                print("numpy array: %s in %s" % (k, d))
            else:
                find_nan_etc(v)


def remove_nan_etc(d: Any) -> Any:
    if isinstance(d, float) and np.isnan(d):
        return None
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    if isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, float) and np.isnan(v):
                d[i] = None
            elif isinstance(v, np.integer):
                d[i] = int(v)
            elif isinstance(v, np.bool_):
                d[i] = bool(v)
            else:
                d[i] = remove_nan_etc(v)
    elif isinstance(d, collections_abc.MutableMapping):
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
            elif isinstance(v, np.integer):
                d[k] = int(v)
            elif isinstance(v, np.bool_):
                d[k] = bool(v)
            elif isinstance(v, np.ndarray):
                d[k] = remove_nan_etc(v.tolist())
            else:
                d[k] = remove_nan_etc(v)
    return d


class AttributeDict:
    def __init__(self, d: Dict[str, Any]):
        self.d = d

    def __getattr__(self, attr: str) -> Any:
        return self.__dict__["d"][attr]

    def __getitem__(self, key: str) -> Any:
        return self.__getattribute__("d")[key]

    def __dir__(self) -> List[Any]:
        return list(self.__getattribute__("d").keys())


ID_RE = re.compile(r"[_A-Za-z][_a-zA-Z0-9]*")


def is_valid_identifier(s: str) -> bool:
    m = ID_RE.match(s)
    return bool(m and m.end(0) == len(s) and not keyword.iskeyword(s))


def fix_identifier(c: str) -> str:
    m = ID_RE.match(c)
    if m is None:
        c = "_" + c
        m = ID_RE.match(c)
    while m and m.end(0) != len(c):
        c = c[: m.end(0)] + "_" + c[m.end(0) + 1 :]
        m = ID_RE.match(c)
    return c


def gen_columns(n: int) -> List[str]:
    return ["_" + str(i) for i in range(1, n + 1)]


def type_fullname(o: Any) -> str:
    module = inspect.getmodule(o)
    # module = o.__class__.__module__
    if module is None or module == inspect.getmodule(str):
        return type(o).__name__
    return str(module) + "." + type(o).__name__


def indices_len(ind: Union[None, Sized, slice]) -> int:
    if isinstance(ind, slice):
        if ind.step is None or ind.step == 1:
            assert isinstance(ind.stop, int) and isinstance(ind.start, int)
            return ind.stop - ind.start
        else:
            return len(range(*ind.indices(ind.stop)))
    if ind is None:
        return 0
    # assert isinstance(ind, PIntSet), f"wrong type {type(ind)}"
    return len(ind)


def fix_loc(indices: Any) -> Any:
    if isinstance(indices, slice):
        return slice(indices.start, indices.stop - 1)  # semantic of slice .loc
    return indices


# See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float


def next_pow2(v: int) -> int:
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    return v + 1


def indices_to_slice(indices: PIntSet) -> Union[slice, PIntSet]:
    if len(indices) == 0:
        return slice(0, 0)
    s = None
    e = 0
    for i in indices:
        if s is None:
            s = e = i
        elif i == e or i == e + 1:
            e = i
        else:
            return indices  # not sliceable
    return slice(s, e + 1)


# def _first_slice(indices):
#     # assert isinstance(indices, PIntSet)
#     ei = enumerate(indices, indices[0])
#     mask = np.equal(*zip(*ei))
#     arr = np.array(indices)
#     head = arr[mask]
#     tail = arr[len(head) :]
#     return head, tail


# def iter_to_slices(indices, fix_loc=False):
#     tail = np.sort(indices)
#     last = tail[-1]
#     incr = 0 if fix_loc else 1
#     slices = []
#     while len(tail):
#         head, tail = _first_slice(tail)
#         stop = head[-1] + incr if head[-1] < last else None
#         slices.append(slice(head[0], stop, 1))
#     return slices


def norm_slice(sl: slice, fix_loc: bool = False, stop: Optional[int] = None) -> slice:
    if sl.start is not None and sl.step == 1:
        return sl
    start = 0 if sl.start is None else sl.start
    step = 1 if sl.step is None else sl.step
    if stop is None:
        stop = sl.stop
    if fix_loc:
        assert isinstance(stop, int)
        stop += 1
    return slice(start, stop, step)


def is_full_slice(sl: slice) -> bool:
    if not isinstance(sl, slice):
        return False
    nsl = norm_slice(sl)
    return nsl.start == 0 and nsl.step == 1 and nsl.stop is None


def inter_slice(this: Optional[slice], that: Optional[slice]) -> Any:
    # Union[None, slice, PIntSet, Tuple[...]]:
    bz = PIntSet([])
    if this is None:
        assert that is not None
        return bz, bz, norm_slice(that)
    if that is None:
        assert this is not None
        return norm_slice(this), bz, bz
    if isinstance(this, slice) and isinstance(that, slice):
        this = norm_slice(this)
        that = norm_slice(that)
        if this == that:
            return bz, this, bz
        if this.step == 1 and this.step == 1:
            if this.start >= that.start and this.stop <= that.stop:
                return bz, this, PIntSet(that) - PIntSet(this)
            if that.start >= this.start and that.stop <= this.stop:
                return PIntSet(this) - PIntSet(that), that, bz
            if this.stop <= that.start or that.stop <= this.start:
                return this, bz, that
            if this.start < that.start:
                left = this
                right = that
            else:
                left = that
                right = this
            common_ = slice(max(left.start, right.start), min(left.stop, right.stop), 1)
            only_left = slice(left.start, common_.start)
            only_right = slice(common_.stop, right.stop)
            if left == this:
                return only_left, common_, only_right
            else:
                return only_right, common_, only_left
        # else: # TODO: can we improve it when step >1 ?
    thisbm = PIntSet.aspintset(this)
    thatbm = PIntSet.aspintset(that)
    commonbm = thisbm & thatbm
    only_this = thisbm - thatbm
    only_that = thatbm - thisbm
    return only_this, commonbm, only_that


def slice_to_array(sl: Any) -> Any:
    if isinstance(sl, slice):
        # return PIntSet(range(*sl.indices(sl.stop)))
        return PIntSet(sl)
    return sl


def slice_to_pintset(sl: slice, stop: Optional[int] = None) -> PIntSet:
    stop = sl.stop if stop is None else stop
    assert isinstance(stop, int)
    return PIntSet(range(*sl.indices(stop)))


def slice_to_arange(sl: Union[slice, np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
    if isinstance(sl, slice):
        assert is_int(sl.stop)
        return cast(np.ndarray[Any, Any], np.arange(*sl.indices(sl.stop)))
    if isinstance(sl, np.ndarray):
        return sl
    raise ValueError(f"Unhandled value {sl}")
    # return np.array(sl)


def get_random_name(prefix: str) -> str:
    return prefix + str(uuid.uuid4()).split("-")[-1]


def all_string(it: Iterable[Any]) -> bool:
    return all([isinstance(elt, str) for elt in it])


def all_int(it: Iterable[Any]) -> bool:
    return all([isinstance(elt, integer_types) for elt in it])


def all_string_or_int(it: Iterable[Any]) -> bool:
    return all_string(it) or all_int(it)


def all_bool(it: Union[np.ndarray[Any, Any], Iterable[Any]]) -> bool:
    if hasattr(it, "dtype"):
        assert isinstance(it, np.ndarray)
        return bool(it.dtype == np.bool_)
    return all([isinstance(elt, bool) for elt in it])


def are_instances(
    it: Union[np.ndarray[Any, Any], Iterable[Any]], type_: Union[type, Tuple[type, ...]]
) -> bool:
    if hasattr(it, "dtype"):
        assert isinstance(it, np.ndarray)
        return bool(
            it.dtype in type_ if isinstance(type_, tuple) else it.dtype == type_
        )
    return all([isinstance(elt, type_) for elt in it])


def is_fancy(key: Any) -> bool:
    return (isinstance(key, np.ndarray) and key.dtype == np.int64) or isinstance(
        key, Iterable
    )


def fancy_to_mask(
    indexes: Any,
    array_shape: Tuple[int, ...],
    mask: Optional[np.ndarray[Any, Any]] = None,
) -> np.ndarray[Any, Any]:
    if mask is None:
        mask = np.zeros(array_shape, dtype=np.bool_)
    else:
        mask.fill(0)
    mask[indexes] = True
    return mask


def mask_to_fancy(mask: Any) -> Any:
    return np.where(mask)


def is_none_alike(x: Any) -> bool:
    if isinstance(x, slice) and x == slice(None, None, None):
        return True
    return x is None


def are_none(*args: Any) -> bool:
    for e in args:
        if e is not None:
            return False
    return True


class RandomBytesIO:
    def __init__(
        self,
        cols: int = 1,
        size: Optional[int] = None,
        rows: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._pos = 0
        self.closed = False  # for pyarrow
        self._reminder = ""
        self._cols = cols
        if size is not None and rows is not None:
            raise ValueError("'size' and 'rows' " "can not be supplied simultaneously")
        self._generator = self.get_row_generator(**kwargs)
        self._yield_size = len(next(self._generator))
        self._size: int
        self._rows: int
        if size is not None:
            rem = size % self._yield_size
            if rem:
                self._size = size - rem + self._yield_size
                self._rows = size // self._yield_size + 1
            else:
                self._size = size
                self._rows = size // self._yield_size
        elif rows is not None:
            self._rows = rows
            self._size = rows * self._yield_size
        else:
            raise ValueError(
                "One of 'size' and 'rows' "
                "must be supplied (put 0 "
                "for an infinite loop)"
            )

    # WARNING: the choice of the mask must guarantee a fixed size for the rows
    def get_row_generator(
        self,
        mask: str = "{: 8.7e}",
        loc: float = 0.5,
        scale: float = 0.8,
        seed: int = 1234,
    ) -> Iterator[str]:
        row_mask = ",".join([mask] * self._cols) + "\n"
        np.random.seed(seed=seed)
        while True:
            yield row_mask.format(
                *np.random.normal(loc=loc, scale=scale, size=self._cols)
            )

    def read(self, n: int = 0) -> bytes:
        if n == 0:
            n = self._size
        if self._pos > self._size - 1:
            return b""
        if self._pos + n > self._size:
            n = self._size - self._pos
        self._pos += n
        if n == len(self._reminder):
            ret = self._reminder
            self._reminder = ""
            return ret.encode("utf-8")
        if n < len(self._reminder):
            ret = self._reminder[:n]
            self._reminder = self._reminder[n:]
            return ret.encode("utf-8")
        # n > len(self._reminder)
        n2 = n - len(self._reminder)
        rem = n2 % self._yield_size
        n_yield = n2 // self._yield_size
        if rem:
            n_yield += 1
        s = "".join([elt for _, elt in zip(range(n_yield), self._generator)])
        raw_str = self._reminder + s
        ret = raw_str[:n]
        self._reminder = raw_str[n:]
        return ret.encode("utf-8")

    def tell(self) -> int:
        return self._pos

    def size(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self._reminder:
            ret = self._reminder
            self._reminder = ""
            self._pos += len(ret)
            return ret
        if self._pos + self._yield_size > self._size:
            raise StopIteration
        self._pos += self._yield_size
        return next(self._generator)

    def readline(self) -> str:
        try:
            return self.__next__()
        except StopIteration:
            return ""

    def readlines(self) -> List[str]:
        return list(self)

    def save(self, file_name: str, force: bool = False) -> None:
        if os.path.exists(file_name) and not force:
            raise ValueError("File {} already exists!".format(file_name))
        with open(file_name, "wb") as fd:
            for row in self:
                fd.write(bytes(row, encoding="ascii"))

    def __str__(self) -> str:
        return "<{} cols={}, rows={} bytes={}>".format(
            type(self), self._cols, self._rows, self._size
        )

    def __repr__(self) -> str:
        return self.__str__()


def _make_csv_fifo_impl(rand_io: RandomBytesIO, file_name: str) -> None:
    rand_io.save(file_name, force=True)


def make_csv_fifo(rand_io: RandomBytesIO, file_name: Optional[str] = None) -> str:
    if file_name is None:
        dir_name = tempfile.mkdtemp(prefix="p10s_rand_")
        file_name = os.path.join(dir_name, "buffer.csv")
    elif os.path.exists(file_name):
        raise ValueError("File {} already exists!".format(file_name))
    os.mkfifo(file_name)
    p = Process(target=_make_csv_fifo_impl, args=(rand_io, file_name))
    p.start()
    return file_name


def del_tmp_csv_fifo(file_name: str) -> None:
    if not file_name.startswith("/tmp/p10s_rand_"):
        raise ValueError("Not in /tmp/p10s_rand_... {}".format(file_name))
    mode = os.stat(file_name).st_mode
    if not st.S_ISFIFO(mode):
        raise ValueError("Not a FIFO {}".format(file_name))
    dn = os.path.dirname(file_name)
    os.remove(file_name)
    os.rmdir(dn)


def is_s3_url(url: str) -> bool:
    """Check for an s3, s3n, or s3a url"""
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in ["s3", "s3n", "s3a"]


def _is_buffer_url(url: str) -> bool:
    res = parse_url(url)
    return res.scheme == "buffer"


def _url_to_buffer(url: str) -> RandomBytesIO:
    res = parse_url(url)
    if res.scheme != "buffer":
        raise ValueError("Wrong buffer url: {}".format(url))
    dict_ = parse_qs(res.query, strict_parsing=True)
    kwargs = dict([(k, int(e[0])) for (k, e) in dict_.items()])
    return RandomBytesIO(**kwargs)


#
# from pandas-dev:  _strip_schema, s3_get_filepath_or_buffer
#


def _strip_schema(url: str) -> str:
    """Returns the url without the s3:// part"""
    result = parse_url(url)
    return result.netloc + result.path


def s3_get_filepath_or_buffer(
    filepath_or_buffer: Any,
    encoding: Optional[str] = None,
    compression: Optional[str] = None,
    custom_fs: Any = None,
) -> Any:
    from botocore.exceptions import NoCredentialsError  # type: ignore

    try:
        # pylint: disable=unused-argument
        fs = custom_fs or s3fs.S3FileSystem(anon=False)
        filepath_or_buffer = fs.open(_strip_schema(filepath_or_buffer))
    except (OSError, NoCredentialsError):
        # boto3 has troubles when trying to access a public file
        # when credentialed...
        # An OSError is raised if you have credentials, but they
        # aren't valid for that bucket.
        # A NoCredentialsError is raised if you don't have creds
        # for that bucket.
        fs = custom_fs or s3fs.S3FileSystem(anon=True)
        filepath_or_buffer = fs.open(_strip_schema(filepath_or_buffer))
    return filepath_or_buffer, None, compression


def filepath_to_buffer(
    filepath: Any,
    encoding: Optional[str] = None,
    compression: Optional[str] = None,
    timeout: Optional[float] = None,
    start_byte: int = 0,
    fs: Any = None,
) -> Tuple[io.IOBase, Optional[str], Optional[str], int]:
    if not is_str(filepath):
        # if start_byte:
        #    filepath.seek(start_byte)
        size = filepath.size() if hasattr(filepath, "size") else 0
        return cast(io.IOBase, filepath), encoding, compression, size
    if is_url(filepath):
        headers = None
        if start_byte:
            headers = {"Range": "bytes={}-".format(start_byte)}
        req = requests.get(filepath, stream=True, headers=headers, timeout=timeout)
        content_encoding = req.headers.get("Content-Encoding", None)
        if content_encoding == "gzip":
            compression = "gzip"
        size = req.headers.get("Content-Length", 0)
        # return HttpDesc(req.raw, filepath), encoding, compression, int(size)
        return cast(io.IOBase, req.raw), encoding, compression, int(size)
    if is_s3_url(filepath):
        reader, encoding, compression = s3_get_filepath_or_buffer(
            filepath, encoding=encoding, compression=compression, custom_fs=fs
        )
        return cast(io.IOBase, reader), encoding, compression, reader.size
    if _is_buffer_url(filepath):
        buffer = _url_to_buffer(filepath)
        return cast(io.IOBase, buffer), encoding, compression, buffer.size()
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        raise ValueError("wrong filepath: {}".format(filepath))
    size = os.stat(filepath).st_size
    stream = io.FileIO(filepath)
    if start_byte:
        stream.seek(start_byte)
    return stream, encoding, compression, size


_compression_to_extension = {
    "gzip": ".gz",
    "bz2": ".bz2",
    "zip": ".zip",
    "xz": ".xz",
}


def _infer_compression(
    filepath_or_buffer: Any, compression: Optional[str]
) -> Optional[str]:
    """
    From Pandas.
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buf :
        a path (str) or buffer
    compression : str or None
        the compression method including None for no compression and 'infer'

    Returns
    -------
    string or None :
        compression method

    Raises
    ------
    ValueError on invalid compression specified
    """

    # No compression has been explicitly specified
    if compression is None:
        return None
    if not is_str(filepath_or_buffer):
        return None
    # Infer compression from the filename/URL extension
    if compression == "infer":
        for compression, extension in _compression_to_extension.items():
            if filepath_or_buffer.endswith(extension):
                return compression
        return None

    # Compression has been specified. Check that it's valid
    if compression in _compression_to_extension:
        return compression

    msg = "Unrecognized compression type: {}".format(compression)
    valid = ["infer", "None"] + sorted(_compression_to_extension)
    msg += "\nValid compression types are {}".format(valid)
    raise ValueError(msg)


def estimate_row_size(filepath: str, length: int = 1_000_000) -> tuple[int, int]:
    compression: str | None = _infer_compression(filepath, "infer")
    stream, encoding, compression, size = filepath_to_buffer(
        filepath, encoding=None, compression=compression
    )
    buff = stream.read(length)
    if compression is None:
        decoded = buff
    elif compression == "bz2":
        decoded = bz2.BZ2Decompressor().decompress(buff)
    elif compression == "xz":
        decoded = lzma.LZMADecompressor().decompress(buff)
    elif compression == "gzip":
        decoded = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16).decompress(buff)
    else:
        raise ValueError(f"Unknown compression {compression}")
    n_buff_rows = decoded.count(b"\n")
    row_len = len(buff) // n_buff_rows
    n_rows = size // row_len
    return n_rows, row_len


def get_physical_base(t: Any) -> Any:
    # TODO: obsolete, to be removed
    return t


def normalize_columns(raw_columns: Union[List[Any], pd.Index[Any]]) -> List[str]:
    uniq: Set[str] = set()
    columns: List[str] = []
    for i, c in enumerate(raw_columns, 1):
        if not isinstance(c, str):
            c = str(c)
        c = fix_identifier(c)
        while c in uniq:
            c += "_" + str(i)
        columns.append(c)
    return columns


def force_valid_id_columns(df: pd.DataFrame) -> None:
    df.columns = normalize_columns(df.columns)  # type: ignore


def force_valid_id_columns_pa(rb: pa.RecordBatch) -> pa.RecordBatch:
    columns: List[str] = normalize_columns(rb.schema.names)
    if columns == rb.schema.names:
        return rb
    return pa.RecordBatch.from_arrays(rb.columns, names=columns)


class Dialog:
    def __init__(self, module: Module, started: bool = False):
        self._module = module
        self.bag: Dict[str, Any] = dict()
        self._started: bool = started

    def set_started(self, v: bool = True) -> Dialog:
        self._started = v
        return self

    def set_output_table(self, res: Union[None, BasePTable, PDict]) -> Dialog:
        self._module.result = res  # type: ignore
        return self

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def output_table(self) -> Union[BasePTable, PDict, None]:
        return self._module.result  # type: ignore


def spy(*args: Any, **kwargs: Any) -> None:
    import time

    f = open(kwargs.pop("file"), "a")
    print(time.time(), *args, file=f, flush=True, **kwargs)
    f.close()


class ModulePatch:
    def __init__(self, name: str) -> None:
        self._name = name
        self.applied: bool = False

    def patch_condition(self, m: Module) -> bool:
        if self.applied:
            return False
        return self._name == m.name

    def before_run_step(self, m: Module, *args: Any, **kwargs: Any) -> None:
        pass

    def after_run_step(self, m: Module, *args: Any, **kwargs: Any) -> None:
        pass


def patch_this(
    to_decorate: PatchRunStepCallable, module: Module, patch: ModulePatch
) -> PatchRunStepCallable:
    """
    patch decorator
    """

    def patch_decorator(
        to_decorate: Callable[[PatchRunStepCallable], PatchRunStepCallable]
    ) -> Callable[..., Any]:
        """
        This is the actual decorator. It brings together the function to be
        decorated and the decoration stuff
        """

        @wraps(to_decorate)
        def patch_wrapper(*args: Any, **kwargs: Any) -> PatchRunStepCallable:
            """
            This function is the decoration
            run_step(self, run_number, step_size, howlong)
            """
            patch.before_run_step(module, *args, **kwargs)
            ret = to_decorate(*args, **kwargs)
            patch.after_run_step(module, *args, **kwargs)
            return ret

        return patch_wrapper

    return patch_decorator(to_decorate)  # type: ignore


def decorate(scheduler: Scheduler, patch: ModulePatch) -> None:
    def decorate_module(m: Module, patch: ModulePatch) -> None:
        assert hasattr(m, "run_step")
        m.run_step = patch_this(  # type: ignore
            to_decorate=m.run_step, module=m, patch=patch
        )
        patch.applied = True

    for m in scheduler.modules().values():
        if not isinstance(m, Module):
            continue  # for mypy
        if patch.patch_condition(m):
            decorate_module(m, patch)


class JSONEncoderNp(js.JSONEncoder):
    "Encode numpy objects"

    def default(self, o: Any) -> Any:
        "Handle default encoding."
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):  # np.float32 don't inherit from float
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, PIntSet):
            return list(o)
        return js.JSONEncoder.default(self, o)

    @staticmethod
    def dumps(*args: Any, **kwargs: Any) -> str:
        return js.dumps(*args, cls=JSONEncoderNp, **kwargs)

    @staticmethod
    def loads(*args: Any, **kwargs: Any) -> Any:
        return js.loads(*args, **kwargs)

    @staticmethod
    def cleanup(*args: Any, **kwargs: Any) -> Any:
        s = JSONEncoderNp.dumps(*args, **kwargs)
        return JSONEncoderNp.loads(s)


async def asynchronize(f: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    # cf. https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
    loop = aio.get_running_loop()
    fun = ft.partial(f, *args, **kwargs)
    return await loop.run_in_executor(None, fun)


def gather_and_run(*args: Awaitable[Any]) -> None:
    """
    this function avoids the use on the "%gui asyncio" magic in notebook
    """

    async def gath() -> Any:
        await aio.gather(*args)

    def func_() -> None:
        loop = aio.new_event_loop()
        aio.set_event_loop(loop)
        loop.run_until_complete(gath())
        loop.close()

    thread = threading.Thread(target=func_, args=())
    thread.start()


def is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return bool(
            get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore
        )
    except ImportError:
        pass
    print("not in notebook")
    return False


def filter_cols(
    df: BasePTable, columns: Optional[List[str]] = None, indices: Optional[Any] = None
) -> BasePTable:
    """
    Return the specified table filtered by the specified indices and
    limited to the columns of interest.
    """
    if columns is None:
        if indices is None:
            return df
        return cast(BasePTable, df.loc[indices])
    if columns is None:
        return df
    if indices is None:
        indices = slice(0, None)
    return df.loc[indices, columns]  # type: ignore
