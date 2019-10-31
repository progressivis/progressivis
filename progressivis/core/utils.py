
import numpy as np
import keyword
import re
from itertools import tee
import uuid
from functools import wraps
from progressivis.core.bitmap import bitmap
try:
    import collections.abc as collections_abc  # only works on python 3.3+
except ImportError:
    import collections as collections_abc

from multiprocessing import Process
from pandas.io.common import _is_url
try:
    from pandas.io.common import _is_s3_url
except ImportError:  # pandas >=0.23.0
    from pandas.io.common import is_s3_url
    _is_s3_url = is_s3_url

import requests
import os
import os.path
import io
import tempfile
import s3fs
import stat as st

integer_types = (int, np.integer)


def is_int(n):
    return isinstance(n, integer_types)


def is_str(s):
    return isinstance(s, str)


def is_iterable(it):
    return isinstance(it, collections_abc.Iterable)


def is_iter_str(it):
    if not is_iterable(it):
        return False
    for s in it:
        if not is_str(s):
            return False
    return True


def len_none(l):
    return 0 if l is None else len(l)


def pairwise(iterator):
    a, b = tee(iterator)
    next(b, None)
    return zip(a, b)


def is_sorted(iterator, compare=None):
    if compare is None:
        compare = lambda a, b: a <= b
    return all(compare(a, b) for a, b in pairwise(iterator))


def remove_nan(d):
    if isinstance(d, float) and np.isnan(d):
        return None
    if isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, float) and np.isnan(v):
                d[i] = None
            else:
                remove_nan(v)
    elif isinstance(d, collections_abc.Mapping):
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
            else:
                remove_nan(v)
    return d


def find_nan_etc(d):
    if isinstance(d, float) and np.isnan(d):
        return None
    if isinstance(d, np.integer):
        print('numpy int: %s' % (d))
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    if isinstance(d, np.ndarray):
        print('numpy array: %s' % (d))
    if isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, float) and np.isnan(v):
                print('numpy nan at %d in: %s' % (i, d))
            elif isinstance(v, np.integer):
                print('numpy int: %s at %d in %s' % (v, i, d))
            elif isinstance(v, np.bool_):
                print('numpy bool: %d in %s' % (i, d))
            elif isinstance(v, np.ndarray):
                print('numpy array: %d in %s' % (i, d))
            else:
                find_nan_etc(v)
    elif isinstance(d, collections_abc.Mapping):
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                print('numpy nan at %s in: %s' % (k, d))
            elif isinstance(v, np.integer):
                print('numpy int: %s in %s' % (k, d))
            elif isinstance(v, np.bool_):
                print('numpy bool: %s in %s' % (k, d))
            elif isinstance(v, np.ndarray):
                print('numpy array: %s in %s' % (k, d))
            else:
                find_nan_etc(v)


def remove_nan_etc(d):
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
    elif isinstance(d, collections_abc.Mapping):
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


class AttributeDict(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, attr):
        return self.__dict__['d'][attr]

    def __getitem__(self, key):
        return self.__getattribute__('d')[key]

    def __dir__(self):
        return list(self.__getattribute__('d').keys())


ID_RE = re.compile(r'[_A-Za-z][_a-zA-Z0-9]*')


def is_valid_identifier(s):
    m = ID_RE.match(s)
    return bool(m and m.end(0) == len(s) and
                not keyword.iskeyword(s))


def fix_identifier(c):
    m = ID_RE.match(c)
    if m is None:
        c = '_' + c
        m = ID_RE.match(c)
    while m.end(0) != len(c):
        c = c[:m.end(0)] + '_' + c[m.end(0)+1:]
        m = ID_RE.match(c)
    return c


def gen_columns(n):
    return ["_"+str(i) for i in range(1, n+1)]


def type_fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__


def indices_len(ind):
    if isinstance(ind, slice):
        if ind.step is None or ind.step == 1:
            return ind.stop-ind.start
        else:
            return len(range(*ind.indices(ind.stop)))
    if ind is None:
        return 0
    return len(ind)


def fix_loc(indices):
    if isinstance(indices, slice):
        return slice(indices.start, indices.stop-1)  # semantic of slice .loc
    return indices

# See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float


def next_pow2(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    return v+1


def indices_to_slice(indices):
    if len(indices) == 0:
        return slice(0, 0)
    s = e = None
    for i in indices:
        if s is None:
            s = e = i
        elif i == e or i == e+1:
            e = i
        else:
            return indices  # not sliceable
    return slice(s, e+1)


def _first_slice(indices):
    # assert isinstance(indices, bitmap)
    ei = enumerate(indices, indices[0])
    mask = np.equal(*zip(*ei))
    arr = np.array(indices)
    head = arr[mask]
    tail = arr[len(head):]
    return head, tail


def iter_to_slices(indices, fix_loc=False):
    tail = np.sort(indices)
    last = tail[-1]
    incr = 0 if fix_loc else 1
    slices = []
    while len(tail):
        head, tail = _first_slice(tail)
        stop = head[-1]+incr if head[-1] < last else None
        slices.append(slice(head[0], stop, 1))
    return slices


def norm_slice(sl):
    if (sl.start is not None and sl.step == 1):
        return sl
    start = 0 if sl.start is None else sl.start
    step = 1 if sl.step is None else sl.step
    return slice(start, sl.stop, step)


def is_full_slice(sl):
    if not isinstance(sl, slice):
        return False
    nsl = norm_slice(sl)
    return nsl.start == 0 and nsl.step == 1 and nsl.stop is None


def inter_slice(this, that):
    bz = bitmap([])
    if this is None:
        return bz, bz, norm_slice(that)
    if that is None:
        return norm_slice(this), bz, bz
    if isinstance(this, slice) and isinstance(that, slice):
        this = norm_slice(this)
        that = norm_slice(that)
        if this == that:
            return bz, this, bz
        if this.step == 1 and this.step == 1:
            if this.start >= that.start and this.stop <= that.stop:
                return bz, this, bitmap(that)-bitmap(this)
            if that.start >= this.start and that.stop <= this.stop:
                return bitmap(this)-bitmap(that),  that, bz
            if this.stop <= that.start or that.stop <= this.start:
                return this, bz, that
            if this.start < that.start:
                left = this
                right = that
            else:
                left = that
                right = this
            common_ = slice(max(left.start, right.start),
                            min(left.stop, right.stop), 1)
            only_left = slice(left.start, common_.start)
            only_right = slice(common_.stop, right.stop)
            if left == this:
                return only_left, common_, only_right
            else:
                return only_right, common_, only_left
        # else: # TODO: can we improve it when step >1 ?
    else:
        if not isinstance(this, bitmap):
            this = bitmap(this)
        if not isinstance(that, bitmap):
            that = bitmap(that)
        common_ = this & that
        only_this = this - that
        only_that = that - this
    return only_this, common_, only_that


def slice_to_array(sl):
    if isinstance(sl, slice):
        # return bitmap(range(*sl.indices(sl.stop)))
        return bitmap(sl)
    return sl


def slice_to_bitmap(sl):
    ret = slice_to_array(sl)
    if isinstance(ret, bitmap):
        return ret
    return bitmap(ret)


def slice_to_arange(sl):
    if isinstance(sl, slice):
        return np.arange(*sl.indices(sl.stop))
    if isinstance(sl, np.ndarray):
        return sl
    return np.array(sl)


def get_random_name(prefix):
    return prefix+str(uuid.uuid4()).split('-')[-1]


def all_string(it):
    return all([isinstance(elt, str) for elt in it])


def all_int(it):
    return all([isinstance(elt, integer_types) for elt in it])


def all_string_or_int(it):
    return all_string(it) or all_int(it)


def all_bool(it):
    if hasattr(it, 'dtype'):
        return it.dtype == bool
    return all([isinstance(elt, bool) for elt in it])


def are_instances(it, type_):
    if hasattr(it, 'dtype'):
        return it.dtype in type_ if isinstance(type_, tuple) else\
          it.type == type_
    return all([isinstance(elt, type_) for elt in it])


def is_fancy(key):
    return (isinstance(key, np.ndarray) and key.dtype == np.int64) or \
      isinstance(key, collections_abc.Iterable)


def fancy_to_mask(indexes, array_shape, mask=None):
    if mask is None:
        mask = np.zeros(array_shape, dtype=np.bool)
    else:
        mask.fill(0)
    mask[indexes] = True
    return mask


def mask_to_fancy(mask):
    return np.where(mask)


def is_none_alike(x):
    if isinstance(x, slice) and x == slice(None, None, None):
        return True
    return x is None


def are_none(*args):
    for e in args:
        if e is not None:
            return False
    return True


class RandomBytesIO(object):
    def __init__(self, cols=1, size=None, rows=None, **kwargs):
        self._pos = 0
        self._reminder = ""
        self._cols = cols
        if size is not None and rows is not None:
            raise ValueError("'size' and 'rows' "
                             "can not be supplied simultaneously")
        self._generator = self.get_row_generator(**kwargs)
        self._yield_size = len(next(self._generator))
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
            raise ValueError("One of 'size' and 'rows' "
                             "must be supplied (put 0 "
                             "for an infinite loop)")

    # WARNING: the choice of the mask must guarantee a fixed size for the rows
    def get_row_generator(self, mask='{: 8.7e}',
                          loc=0.5, scale=0.8, seed=1234):
        row_mask = ','.join([mask]*self._cols)+'\n'
        np.random.seed(seed=seed)
        while True:
            yield row_mask.format(*np.random.normal(loc=loc, scale=scale,
                                                    size=self._cols))

    def read(self, n=0):
        if n == 0:
            n = self._size
        if self._pos > self._size - 1:
            return b''
        if self._pos + n > self._size:
            n = self._size - self._pos
        self._pos += n
        if n == len(self._reminder):
            ret = self._reminder
            self._reminder = ""
            return ret.encode('utf-8')
        if n < len(self._reminder):
            ret = self._reminder[:n]
            self._reminder = self._reminder[n:]
            return ret.encode('utf-8')
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
        return ret.encode('utf-8')

    def tell(self):
        return self._pos

    def size(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        if self._reminder:
            ret = self._reminder
            self._reminder = ''
            self._pos += len(ret)
            return ret
        if self._pos + self._yield_size > self._size:
            raise StopIteration
        self._pos += self._yield_size
        return next(self._generator)

    def next(self):
        return self.__next__()

    def readline(self):
        try:
            return self.__next__()
        except StopIteration:
            return ''

    def readlines(self):
        return list(self)

    def save(self, file_name, force=False):
        if os.path.exists(file_name) and not force:
            raise ValueError("File {} already exists!".format(file_name))
        with open(file_name, 'wb') as fd:
            for row in self:
                fd.write(bytes(row, encoding='ascii'))

    def __str__(self):
        return "<{} cols={}, rows={} bytes={}>".format(type(self), self._cols,
                                                       self._rows, self._size)

    def __repr__(self):
        return self.__str__()


def _make_csv_fifo_impl(rand_io, file_name):
    rand_io.save(file_name, force=True)


def make_csv_fifo(rand_io, file_name=None):
    if file_name is None:
        dir_name = tempfile.mkdtemp(prefix='p10s_rand_')
        file_name = os.path.join(dir_name, 'buffer.csv')
    elif os.path.exists(file_name):
        raise ValueError("File {} already exists!".format(file_name))
    os.mkfifo(file_name)
    p = Process(target=_make_csv_fifo_impl, args=(rand_io, file_name))
    p.start()
    return file_name


def del_tmp_csv_fifo(file_name):
    if not file_name.startswith('/tmp/p10s_rand_'):
        raise ValueError("Not in /tmp/p10s_rand_... {}".format(file_name))
    mode = os.stat(file_name).st_mode
    if not st.S_ISFIFO(mode):
        raise ValueError("Not a FIFO {}".format(file_name))
    dn = os.path.dirname(file_name)
    os.remove(file_name)
    os.rmdir(dn)



from urllib.parse import urlparse as parse_url
from urllib.parse import parse_qs


def _is_buffer_url(url):
    res = parse_url(url)
    return res.scheme == 'buffer'


def _url_to_buffer(url):
    #import pdb; pdb.set_trace()
    res = parse_url(url)
    if res.scheme != 'buffer':
        raise ValueError("Wrong buffer url: {}".format(url))
    dict_ = parse_qs(res.query, strict_parsing=1)
    kwargs = dict([(k, int(e[0])) for (k, e) in dict_.items()])
    return RandomBytesIO(**kwargs)

#
# from pandas-dev:  _strip_schema, s3_get_filepath_or_buffer
#


def _strip_schema(url):
    """Returns the url without the s3:// part"""
    result = parse_url(url)
    return result.netloc + result.path


def s3_get_filepath_or_buffer(filepath_or_buffer, encoding=None,
                              compression=None):
    # pylint: disable=unused-argument
    fs = s3fs.S3FileSystem(anon=False)
    from botocore.exceptions import NoCredentialsError
    try:
        filepath_or_buffer = fs.open(_strip_schema(filepath_or_buffer))
    except (OSError, NoCredentialsError):
        # boto3 has troubles when trying to access a public file
        # when credentialed...
        # An OSError is raised if you have credentials, but they
        # aren't valid for that bucket.
        # A NoCredentialsError is raised if you don't have creds
        # for that bucket.
        fs = s3fs.S3FileSystem(anon=True)
        filepath_or_buffer = fs.open(_strip_schema(filepath_or_buffer))
    return filepath_or_buffer, None, compression


def filepath_to_buffer(filepath, encoding=None,
                       compression=None, timeout=None, start_byte=0):
    if not is_str(filepath):
        # if start_byte:
        #    filepath.seek(start_byte)
        return filepath, encoding, compression, filepath.size()
    if _is_url(filepath):
        headers = None
        if start_byte:
            headers = {"Range": "bytes={}-".format(start_byte)}
        req = requests.get(filepath, stream=True, headers=headers,
                           timeout=timeout)
        content_encoding = req.headers.get('Content-Encoding', None)
        if content_encoding == 'gzip':
            compression = 'gzip'
        size = req.headers.get('Content-Length', 0)
        # return HttpDesc(req.raw, filepath), encoding, compression, int(size)
        return req.raw, encoding, compression, int(size)
    if _is_s3_url(filepath):
        reader, encoding, compression = s3_get_filepath_or_buffer(
            filepath,
            encoding=encoding,
            compression=compression)
        return reader, encoding, compression, reader.size
    if _is_buffer_url(filepath):
        buffer = _url_to_buffer(filepath)
        return buffer, encoding, compression, buffer.size()
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        raise ValueError("wrong filepath: {}".format(filepath))
    size = os.stat(filepath).st_size
    stream = io.FileIO(filepath)
    if start_byte:
        stream.seek(start_byte)
    return stream, encoding, compression, size


_compression_to_extension = {
    'gzip': '.gz',
    'bz2': '.bz2',
    'zip': '.zip',
    'xz': '.xz',
}


def _infer_compression(filepath_or_buffer, compression):
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
    if compression == 'infer':
        for compression, extension in _compression_to_extension.items():
            if filepath_or_buffer.endswith(extension):
                return compression
        return None

    # Compression has been specified. Check that it's valid
    if compression in _compression_to_extension:
        return compression

    msg = 'Unrecognized compression type: {}'.format(compression)
    valid = ['infer', None] + sorted(_compression_to_extension)
    msg += '\nValid compression types are {}'.format(valid)
    raise ValueError(msg)


def get_physical_base(t):
    return t if t.base is None else get_physical_base(t.base)


def force_valid_id_columns(df):
    uniq = set()
    columns = []
    i = 0
    for c in df.columns:
        i += 1
        if not isinstance(c, str):
            c = str(c)
        c = fix_identifier(c)
        while c in uniq:
            c += ('_' + str(i))
        columns.append(c)
    df.columns = columns


class _Bag(object):
    pass


class Dialog(object):
    def __init__(self, module, started=False):
        self._module = module
        self.bag = _Bag()
        self._started = started

    def set_started(self, v=True):
        self._started = v
        return self

    def set_output_table(self, res):
        #with self._module.lock:
        self._module._table = res
        return self

    @property
    def is_started(self):
        return self._started

    @property
    def output_table(self):
        return self._module._table


def spy(*args, **kwargs):
    import time
    f = open(kwargs.pop('file'), "a")
    print(time.time(), *args, file=f, flush=True, **kwargs)
    f.close()


def patch_this(to_decorate, module, patch):
    """
    patch decorator
    """
    def patch_decorator(to_decorate):
        """
        This is the actual decorator. It brings together the function to be
        decorated and the decoration stuff
        """
        @wraps(to_decorate)
        def patch_wrapper(*args, **kwargs):
            """
            This function is the decoration
            run_step(self, run_number, step_size, howlong)
            """
            patch.before_run_step(module, *args, **kwargs)
            ret = to_decorate(*args, **kwargs)
            patch.after_run_step(module, *args, **kwargs)
            return ret
        return patch_wrapper
    return patch_decorator(to_decorate)


class ModulePatch(object):
    def __init__(self, name):
        self._name = name
        self.applied = False

    def patch_condition(self, m):
        if self.applied:
            return False
        return self._name == m.name

    def before_run_step(self, m, *args, **kwargs):
        pass

    def after_run_step(self, m, *args, **kwargs):
        pass


def decorate_module(m, patch):
    assert hasattr(m, 'run_step')
    m.run_step = patch_this(to_decorate=m.run_step, module=m, patch=patch)
    patch.applied = True


def decorate(scheduler, patch):
    for m in scheduler.modules().values():
        if patch.patch_condition(m):
            decorate_module(m, patch)
