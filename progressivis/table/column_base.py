"""
Base class for columns.
"""
from __future__ import annotations

import operator
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

from progressivis.core.config import get_option

from typing import Any, Optional, List, TYPE_CHECKING, Callable, Union, cast

if TYPE_CHECKING:
    from .table_base import IndexTable, Tuple, TableChanges
    from .dshape import DataShape


logger = logging.getLogger(__name__)


class _Loc:
    # pylint: disable=too-few-public-methods
    def __init__(self, column: BaseColumn) -> None:
        self.column = column

    def parse_loc(self, loc: Any) -> Any:
        "Nomalize the loc"
        if isinstance(loc, tuple):
            raise ValueError(f'Location accessor not implemented for key "{loc}"')
        return self.column.id_to_index(loc)

    def __delitem__(self, loc: Any) -> None:
        index = self.parse_loc(loc)
        del self.column[index]

    def __getitem__(self, loc: Any) -> Any:
        index = self.parse_loc(loc)
        return self.column[index]

    def __setitem__(self, loc: Any, value: Any) -> None:
        index = self.parse_loc(loc)
        self.column[index] = value


class BaseColumn(metaclass=ABCMeta):
    """
    Base class for columns.
    """

    # pylint: disable=too-many-public-methods
    def __init__(self,
                 name: str,
                 index: IndexTable,
                 base: Optional[BaseColumn] = None):
        self._index = index
        self._name = name
        self._base = base

    @property
    def loc(self) -> Any:
        "Return the accessor by id"
        return _Loc(self)

    @property
    def name(self) -> str:
        "Return the name"
        return self._name

    @property
    def index(self) -> IndexTable:
        "Retur the index"
        return self._index

    @property
    def base(self) -> Optional[BaseColumn]:
        "Return the base column or None"
        return self._base

    @abstractproperty
    def size(self) -> int:
        "Return the size of the column"
        pass

    @abstractproperty
    def fillvalue(self) -> Any:
        """
        Return the default value for elements created
        when the column is enlarged
        """
        pass

    def id_to_index(self, loc: Any, as_slice: bool = True) -> Any:
        "Convert an identifier to an index"
        # pylint: disable=unused-argument
        return self.index.id_to_index(loc, as_slice)

    def update(self) -> None:
        """
        Synchronize the column size with its index if needed.
        This method can be called safely any time, it will
        do nothing if nothing needs to be done.
        """
        self.resize(self.index.size)

    def __repr__(self) -> str:
        return str(self) + self.info_contents()

    def __str__(self) -> str:
        classname = self.__class__.__name__
        rep = '%s("%s", dshape=%s)[%d]' % (
            classname,
            self.name,
            str(self.dshape),
            len(self),
        )
        return rep

    def info_contents(self) -> str:
        "Return a string describing the contents of the column"
        length = len(self)
        rep = ""
        max_rows = get_option("display.max_rows")
        for row in range(min(max_rows, length)):
            rep += "\n    %d: %s" % (self.id_to_index(row), self[row])
        if max_rows and length > max_rows:
            rep += "\n..."
        return rep

    def has_freelist(self) -> bool:
        # pylint: disable=no-self-use
        "Return True of the column manages a free list"
        return False

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self):
        return np.nditer(self.value)

    @abstractmethod
    def __getitem__(self, index: Any) -> Any:
        pass

    def tolist(self) -> List[Any]:
        "Return a list from the values of the column"
        return self.values.tolist()

    def read_direct(self,
                    array: np.ndarray,
                    source_sel: Any = None,
                    dest_sel: Any = None):
        """ Read data from column into an existing NumPy array.

        Selections must be the output of numpy.s_[<args>] or slice.
        """
        if dest_sel:
            array[dest_sel] = self[...] if source_sel is None else self[source_sel]
        else:
            array[:] = self[:] if source_sel is None else self[source_sel]

    @abstractmethod
    def __setitem__(self, index: Any, val: Any) -> None:
        pass

    @abstractproperty
    def value(self) -> np.ndarray:
        "Return a numpy array-compatible object containing the values"
        pass

    @property
    def values(self) -> np.ndarray:
        "Synonym with value"
        return self.value

    @abstractmethod
    def resize(self, newsize: int) -> None:
        "Resize this column"
        pass

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        "Return the shape of that column"
        pass

    @abstractmethod
    def set_shape(self, shape: Tuple[int, ...]) -> None:
        """Set the shape of that column.
        The semantics can be different than that of numpy."""
        pass

    @abstractproperty
    def maxshape(self) -> Tuple[int, ...]:
        "Return the maximum shape"
        pass

    @abstractproperty
    def dtype(self) -> np.dtype:
        "Return the dtype"
        pass

    @abstractproperty
    def dshape(self) -> DataShape:
        "Return the datashape"
        pass

    @abstractproperty
    def chunks(self) -> Tuple[int, ...]:
        "Return the chunk size"
        pass

    @abstractmethod
    def __delitem__(self, index: Any) -> None:
        pass

    def last(self) -> Any:
        "Return the last element"
        length = self.size
        if length == 0:
            return None
        return self[length - 1]  # the last element is never -1

    # def index_to_chunk(self, index):
    #     "Convert and index to its chunk index"
    #     if isinstance(index, integer_types):
    #         index = (index, )
    #     chunks = self.chunks
    #     return [i // c for i, c in zip(index, chunks)]

    # def chunk_to_index(self, chunk):
    #     "Convert a chunk index to its index"
    #     if isinstance(chunk, integer_types):
    #         chunk = (chunk, )
    #     chunks = self.chunks
    #     return [i * c for i, c in zip(chunk, chunks)]

    @property
    def changes(self) -> Optional[TableChanges]:
        "Return the ChangeManager associated with the index of this column"
        if self._index is None:
            return None
        return self._index.changes

    @changes.setter
    def changes(self, tablechange: Optional[TableChanges]):
        "Set the ChangeManager associated with the index of this column"
        if self.index is None:
            raise RuntimeError("Column has no index")
        self.index.changes = tablechange

    def compute_updates(self, start: int, now: int, mid: str, cleanup: bool = True):
        "Return the updates of this column managed by the index"
        if self.index is None:
            return None
        return self.index.compute_updates(start, now, mid, cleanup)

    def unary(self,
              operation=Callable[[np.ndarray, int, bool], np.ndarray],
              **kwargs):
        "Unary function manager"
        axis = kwargs.pop("axis", 0)
        keepdims = kwargs.pop("keepdims", False)
        # ignore other kwargs, maybe raise error in the future
        return operation(self.value, axis=axis, keepdims=keepdims)

    def binary(self,
               operation: Callable[[np.ndarray, Union[np.ndarray, int, float, bool]], np.ndarray],
               other: Union[np.ndarray, BaseColumn, int, float, bool],
               **kwargs) -> np.ndarray:
        "Binary function manager"
        axis = kwargs.pop("axis", 0)
        assert axis == 0
        # if isinstance(other, (int, float, bool, np.ndarray)):
        value: np.ndarray
        if isinstance(other, (np.ndarray, int, float, bool)):
            value = operation(self.value, other)
        elif isinstance(other, BaseColumn):
            value = operation(self.value, other.value)
        else:
            raise ValueError(f"Invalid type {type(other)}")
        return value

    def __abs__(self, **kwargs) -> np.ndarray:
        return self.unary(np.abs, **kwargs)

    def __add__(self, other) -> np.ndarray:
        return self.binary(operator.add, other)

    def __radd__(self, other) -> np.ndarray:
        return other.binary(operator.add, self)

    def __and__(self, other) -> np.ndarray:
        return self.binary(operator.and_, other)

    def __rand__(self, other) -> np.ndarray:
        return other.binary(operator.and_, self)

    def __eq__(self, other):  # type: ignore
        return self.binary(operator.eq, other)

    def __gt__(self, other) -> np.ndarray:
        return self.binary(operator.gt, other)

    def __ge__(self, other) -> np.ndarray:
        return self.binary(operator.ge, other)

    def __invert__(self) -> np.ndarray:
        return self.unary(np.invert)

    def __lshift__(self, other) -> np.ndarray:
        return self.binary(operator.lshift, other)

    def __rlshift__(self, other) -> np.ndarray:
        return other.binary(operator.lshift, self)

    def __lt__(self, other) -> np.ndarray:
        return self.binary(operator.lt, other)

    def __le__(self, other) -> np.ndarray:
        return self.binary(operator.le, other)

    def __mod__(self, other) -> np.ndarray:
        return self.binary(operator.mod, other)

    def __rmod__(self, other) -> np.ndarray:
        return other.binary(operator.mod, self)

    def __mul__(self, other) -> np.ndarray:
        return self.binary(operator.mul, other)

    def __rmul__(self, other) -> np.ndarray:
        return other.binary(operator.mul, self)

    def __ne__(self, other) -> bool:
        return any(self.binary(operator.ne, other))

    def __or__(self, other) -> np.ndarray:
        return self.binary(operator.or_, other)

    def __pos__(self) -> BaseColumn:
        return self

    def __ror__(self, other) -> np.ndarray:
        return other.binary(operator.or_, self)

    def __pow__(self, other) -> np.ndarray:
        return self.binary(operator.pow, other)

    def __rpow__(self, other) -> np.ndarray:
        return other.binary(operator.pow, self)

    def __rshift__(self, other) -> np.ndarray:
        return self.binary(operator.rshift, other)

    def __rrshift__(self, other) -> np.ndarray:
        return other.binary(operator.rshift, self)

    def __sub__(self, other) -> np.ndarray:
        return self.binary(operator.sub, other)

    def __rsub__(self, other) -> np.ndarray:
        return other.binary(operator.sub, self)

    def __truediv__(self, other) -> np.ndarray:
        return self.binary(operator.truediv, other)

    def __rtruediv__(self, other) -> np.ndarray:
        return other.binary(operator.truediv, self)

    def __floordiv__(self, other) -> np.ndarray:
        return self.binary(operator.floordiv, other)

    def __rfloordiv__(self, other) -> np.ndarray:
        return other.binary(operator.floordiv, self)

    def __xor__(self, other) -> np.ndarray:
        return self.binary(operator.xor, other)

    def __rxor__(self, other) -> np.ndarray:
        return other.binary(operator.xor, self)

    def any(self, **kwargs) -> bool:
        "Return True if any element is not False"
        return self.unary(np.any, **kwargs)

    def all(self, **kwargs) -> bool:
        "Return True if all the elements are True"
        return self.unary(np.all, **kwargs)

    def min(self, **kwargs) -> Any:
        "Return the min value"
        axis = cast(int, kwargs.pop("axis", 0))
        keepdims = cast(bool, kwargs.pop("keepdims", False))
        return self.value.min(axis=axis, keepdims=keepdims)

    def max(self, **kwargs) -> Any:
        "Return the max value"
        axis = cast(int, kwargs.pop("axis", 0))
        keepdims = cast(bool, kwargs.pop("keepdims", False))
        return self.value.max(axis=axis, keepdims=keepdims)
