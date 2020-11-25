"""
Base class for columns.
"""
import operator
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

from progressivis.core.config import get_option
from progressivis.core.utils import integer_types


logger = logging.getLogger(__name__)

class _Loc(object):
    # pylint: disable=too-few-public-methods
    def __init__(self, column):
        self.column = column

    def parse_loc(self, loc):
        "Nomalize the loc"
        if isinstance(loc, tuple):
            raise ValueError('Location accessor not implemented for key "%s"' % loc)
        return self.column.id_to_index(loc)

    def __delitem__(self, loc):
        index = self.parse_loc(loc)
        del self.column[index]

    def __getitem__(self, loc):
        index = self.parse_loc(loc)
        return self.column[index]

    def __setitem__(self, loc, value):
        index = self.parse_loc(loc)
        self.column[index] = value

class BaseColumn(metaclass=ABCMeta):
    """
    Base class for columns.
    """
    # pylint: disable=too-many-public-methods
    def __init__(self, name, index=None, base=None):
        self._index = index
        self._name = name
        self._base = base

    @property
    def loc(self):
        "Return the accessor by id"
        return _Loc(self)

    @property
    def name(self):
        "Return the name"
        return self._name

    @property
    def index(self):
        "Retur the index"
        return self._index

    @property
    def base(self):
        "Return the base column or None"
        return self._base

    @abstractproperty
    def size(self):
        "Return the size of the column"
        pass

    @abstractproperty
    def fillvalue(self):
        "Return the default value of elements created when the column is enlarged"
        pass

    def id_to_index(self, loc, as_slice=True):
        "Convert an identifier to an index"
        # pylint: disable=unused-argument
        if self.index:
            #return self.index[loc]
            return self.index.id_to_index(loc, as_slice)
        raise KeyError('Cannot map key %s to index'% loc)

    def update(self):
        """
        Synchronize the column size with its index if needed.
        This method can be called safely any time, it will
        do nothing if nothing needs to be done.
        """
        self.resize(self.index.size)

    def __repr__(self):
        return str(self)+self.info_contents()

    def __str__(self):
        classname = self.__class__.__name__
        rep = '%s("%s", dshape=%s)[%d]' % (classname, self.name, str(self.dshape), len(self))
        return rep

    def info_contents(self):
        "Return a string describing the contents of the column"
        length = len(self)
        rep = ''
        max_rows = get_option('display.max_rows')
        for row in range(min(max_rows, length)):
            rep += ("\n    %d: %s"%(self.id_to_index(row), self[row]))
        if max_rows and length > max_rows:
            rep += "\n..."
        return rep

    def has_freelist(self):
        # pylint: disable=no-self-use
        "Return True of the column manages a free list"
        return False

    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        return np.nditer(self.value)

    @abstractmethod
    def __getitem__(self, index):
        pass

    def tolist(self):
        "Return a list from the values of the column"
        return self.values.tolist()

    def read_direct(self, array, source_sel=None, dest_sel=None):
        """ Read data from column into an existing NumPy array.

        Selections must be the output of numpy.s_[<args>] or slice.
        """
        if dest_sel:
            array[dest_sel] = self[...] if source_sel is None else self[source_sel]
        else:
            array[:] = self[:] if source_sel is None else self[source_sel]

    @abstractmethod
    def __setitem__(self, index, val):
        pass

    @abstractproperty
    def value(self):
        "Return a numpy array-compatible object containing the values"
        pass

    @property
    def values(self):
        "Synonym with value"
        return self.value

    @abstractmethod
    def resize(self, newsize):
        "Resize this column"
        pass

    @abstractproperty
    def shape(self):
        "Return the shape of that column"
        pass

    @abstractmethod
    def set_shape(self, shape):
        "Set the shape of that column. The semantics can be different than that of numpy."
        pass

    @abstractproperty
    def maxshape(self):
        "Return the maximum shape"
        pass

    @abstractproperty
    def dtype(self):
        "Return the dtype"
        pass

    @abstractproperty
    def dshape(self):
        "Return the datashape"
        pass

    @abstractproperty
    def chunks(self):
        "Return the chunk size"
        pass

    @abstractmethod
    def __delitem__(self, index):
        pass

    def last(self):
        "Return the last element"
        length = self.size
        if length == 0:
            return None
        return self[length-1] # the last element is never -1

    def index_to_chunk(self, index):
        "Convert and index to its chunk index"
        if isinstance(index, integer_types):
            index = (index, )
        chunks = self.chunks
        return [i // c for i, c in zip(index, chunks)]

    def chunk_to_index(self, chunk):
        "Convert a chunk index to its index"
        if isinstance(chunk, integer_types):
            chunk = (chunk, )
        chunks = self.chunks
        return [i * c for i, c in zip(chunk, chunks)]

    @property
    def changes(self):
        "Return the ChangeManager associated with the index of this column"
        if self._index is None:
            return None
        return self._index.changes

    @changes.setter
    def changes(self, changemanager):
        "Set the ChangeManager associated with the index of this column"
        if self.index is None:
            raise RuntimeError('Column has no index')
        self.index.changes = changemanager

    def compute_updates(self, start, now, mid=None, cleanup=True):
        "Return the updates of this column managed by the index"
        if self.index is None:
            return None
        return self.index.compute_updates(start, now, mid, cleanup)

    def unary(self, operation, **kwargs):
        "Unary function manager"
        axis = kwargs.pop('axis', 0)
        keepdims = kwargs.pop('keepdims', False)
        # ignore other kwargs, maybe raise error in the future
        return operation(self.value, axis=axis, keepdims=keepdims)

    def binary(self, operation, other, **kwargs):
        "Binary function manager"
        axis = kwargs.pop('axis', 0)
        assert axis == 0
        isscalar = (np.isscalar(other) or isinstance(other, np.ndarray))
        if isscalar:
            value = operation(self.value, other)
        else:
            value = operation(self.value, other.value)
        return value

    def __abs__(self, **kwargs):
        return self.unary(np.abs, **kwargs)

    def __add__(self, other):
        return self.binary(operator.add, other)

    def __radd__(self, other):
        return other.binary(operator.add, self)

    def __and__(self, other):
        return self.binary(operator.and_, other)

    def __rand__(self, other):
        return other.binary(operator.and_, self)

    # def __div__(self, other):
    #     return self.binary(operator.div, other)

    # def __rdiv__(self, other):
    #     return other.binary(operator.div, self)

    def __eq__(self, other):
        return self.binary(operator.eq, other)

    def __gt__(self, other):
        return self.binary(operator.gt, other)

    def __ge__(self, other):
        return self.binary(operator.ge, other)

    def __invert__(self):
        return self.unary(np.invert)

    def __lshift__(self, other):
        return self.binary(operator.lshift, other)

    def __rlshift__(self, other):
        return other.binary(operator.lshift, self)

    def __lt__(self, other):
        return self.binary(operator.lt, other)

    def __le__(self, other):
        return self.binary(operator.le, other)

    def __mod__(self, other):
        return self.binary(operator.mod, other)

    def __rmod__(self, other):
        return other.binary(operator.mod, self)

    def __mul__(self, other):
        return self.binary(operator.mul, other)

    def __rmul__(self, other):
        return other.binary(operator.mul, self)

    def __ne__(self, other):
        return self.binary(operator.ne, other)

    def __neg__(self):
        return self.unary(np.neg)

    def __or__(self, other):
        return self.binary(operator.or_, other)

    def __pos__(self):
        return self

    def __ror__(self, other):
        return other.binary(operator.or_, self)

    def __pow__(self, other):
        return self.binary(operator.pow, other)

    def __rpow__(self, other):
        return other.binary(operator.pow, self)

    def __rshift__(self, other):
        return self.binary(operator.rshift, other)

    def __rrshift__(self, other):
        return other.binary(operator.rshift, self)

    def __sub__(self, other):
        return self.binary(operator.sub, other)

    def __rsub__(self, other):
        return other.binary(operator.sub, self)

    def __truediv__(self, other):
        return self.binary(operator.truediv, other)

    def __rtruediv__(self, other):
        return other.binary(operator.truediv, self)

    def __floordiv__(self, other):
        return self.binary(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return other.binary(operator.floordiv, self)

    def __xor__(self, other):
        return self.binary(operator.xor, other)

    def __rxor__(self, other):
        return other.binary(operator.xor, self)

    def any(self, **kwargs):
        "Return True if any element is not False"
        return self.unary(np.any, **kwargs)

    def all(self, **kwargs):
        "Return True if all the elements are True"
        return self.unary(np.all, **kwargs)

    def min(self, **kwargs):
        "Return the min value"
        return self.unary(np.min, **kwargs)

    def max(self, **kwargs):
        "Return the max value"
        return self.unary(np.max, **kwargs)
