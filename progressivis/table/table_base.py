"""Base class for PTables
"""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Mapping, Iterable
import operator
import logging
import numpy as np
import pandas as pd
import datetime as dt
from progressivis.core.index_update import IndexUpdate
from progressivis.core.utils import (
    integer_types,
    norm_slice,
    is_slice,
    is_int,
    is_str,
    all_int,
    all_string,
    all_string_or_int,
    all_bool,
    indices_len,
    remove_nan,
    is_none_alike,
    get_physical_base,
)
from progressivis.core.config import get_option
from progressivis.core.pintset import PIntSet
from .dshape import dshape_print, dshape_create, DataShape, EMPTY_DSHAPE
from .tablechanges import PTableChanges as PTableChanges

import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import (
    Union,
    Any,
    Optional,
    Dict,
    Set,
    List,
    Tuple,
    TYPE_CHECKING,
    Callable,
    Sequence,
    overload,
    Iterator,
)

if TYPE_CHECKING:
    from .column_base import BasePColumn
    from .row import Row
    from .compute import Computed
    BinaryRet = Union[Dict[str, np.ndarray[Any, Any]], "BasePTable"]
    ColIndexer = Union[int, np.integer[Any], str]

Shape = Tuple[int, ...]
Indexer = Union[Any]  # improve later
UnaryRet = Dict[str, Any]

logger = logging.getLogger(__name__)


FAST = 1


def _to_datetime(arr: np.ndarray[Any, Any]) -> List[dt.datetime]:
    return [dt.datetime(*elt) for elt in arr[:]]


class _BaseLoc:
    # pylint: disable=too-few-public-methods
    def __init__(self, this_table: BasePTable, as_loc: bool = True) -> None:
        self._table = this_table
        self._as_loc = as_loc

    def parse_key(self, key: Indexer) -> Tuple[Any, Any, Any]:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError(f"getitem not implemented for {key=}")
            index, col_key = key
        else:
            index, col_key = key, slice(None)
        locs = None
        if self._as_loc:  # i.e loc mode
            locs = index
            index = self._table._any_to_pintset(index)
        return index, col_key, locs

    def parse_key_to_pintset(self, key: Indexer) -> Tuple[Any, Any, Any]:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError(f"getitem not implemented for {key=}")
            raw_index, col_key = key
        else:
            raw_index, col_key = key, slice(None)
        index = self._table._any_to_pintset(raw_index)
        return index, col_key, raw_index


class _Loc(_BaseLoc):
    # pylint: disable=too-few-public-methods
    def __delitem__(self, key: Indexer) -> None:
        index, col_key, raw_index = self.parse_key_to_pintset(key)
        if not is_none_alike(col_key):
            raise ValueError(f"Cannot delete {key=}")
        self._table._drop(index, raw_index)

    @overload
    def __getitem__(self, key: int) -> Optional[Row]:
        ...

    @overload
    def __getitem__(
        self, key: Tuple[int, Union[slice, List[str], List[int]]]
    ) -> Optional[Row]:
        ...

    @overload
    def __getitem__(self, key: Tuple[int, str]) -> Optional[Any]:
        ...

    @overload
    def __getitem__(
        self, key: Union[PIntSet, np.ndarray[Any, Any], slice, List[str], List[int]]
    ) -> Optional[BasePTable]:
        ...

    @overload
    def __getitem__(
        self,
        key: Tuple[
            Union[PIntSet, np.ndarray[Any, Any], slice],
            Union[int, str, slice, List[str]],
        ],
    ) -> Optional[BasePTable]:
        ...

    def __getitem__(self, key: Indexer) -> Any:
        index, col_key, raw_index = self.parse_key_to_pintset(key)
        if not (is_slice(raw_index) or index in self._table.index):
            diff_ = index - self._table.index
            raise KeyError(f"Non existing indices {diff_}")
        if isinstance(raw_index, integer_types):
            row = self._table.row(int(raw_index))
            if row is not None and col_key != slice(None):
                return row[col_key]
            return row
        elif isinstance(index, Iterable):
            base = self._table.get_original_base()
            btab = BasePTable(selection=raw_index, base=base)
            columns, columndict = self._table.make_projection(col_key, btab)
            btab._columns = columns
            btab._columndict = columndict
            btab._dshape = dshape_create(
                "{" + ",".join([f"{c.name}:{c.dshape}" for c in btab._columns]) + "}"
            )
            btab._masked = self._table
            return btab
        raise ValueError('getitem not implemented for index "%s"', index)

    def __setitem__(self, key: Indexer, value: Any) -> Any:
        index, col_key, raw_index = self.parse_key_to_pintset(key)
        if isinstance(raw_index, integer_types):
            index = raw_index
        self._table.setitem_2d(index, col_key, value)


class _At(_BaseLoc):
    # pylint: disable=too-few-public-methods
    def _parse_key(self, key: Indexer) -> Tuple[Any, Any]:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('getitem not implemented for key "%s"' % key)
            index, col_key = key
        else:
            raise KeyError(f"Invalid key {key}")
        if not is_int(index):
            raise KeyError(f"Invalid row key {index}")
        if not (is_str(col_key) or is_int(col_key)):
            raise KeyError(f"Invalid column key {col_key}")
        return index, col_key

    def __getitem__(self, key: Indexer) -> Any:
        index, col_key = self._parse_key(key)
        if index not in self._table.index:
            raise KeyError(f"Not existing indice {index}")
        return self._table[col_key][index]

    def __setitem__(self, key: Indexer, value: Any) -> None:
        index, col_key = self._parse_key(key)
        if not is_int(index):
            raise KeyError(f"Invalid row key {index}")
        if not (is_str(col_key) or is_int(col_key)):
            raise ValueError('At setitem not implemented for column key "%s"' % col_key)
        self._table[col_key][index] = value


class BasePTable(metaclass=ABCMeta):
    # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """
    Base class for ``progressivis`` tables and table-views (:class:`PTable <progressivis.PTable>`, :class:`PTableSelectedView <progressivis.table.PTableSelectedView>` etc.)

    .. warning::
        Do not instanciate this class directly!

    """

    def __init__(
        self,
        base: Optional[BasePTable] = None,
        selection: Union[PIntSet, slice] = slice(0, None),
        columns: Optional[List[BasePColumn]] = None,
        columndict: Optional[Dict[str, int]] = None,
        computed: Optional[Dict[str, Computed]] = None,
    ) -> None:
        """
        """
        self._base: Optional[BasePTable] = (
            base if (base is None or base._base is None) else base._base
        )
        self._selection: Union[slice, PIntSet] = selection
        self._columns: List[BasePColumn] = [] if columns is None else columns
        self._columndict: Dict[str, int] = dict() if columndict is None else columndict
        self._loc = _Loc(self, True)
        self._at = _At(self, True)
        self._masked = base
        self._dshape: DataShape = EMPTY_DSHAPE
        self.computed = {} if computed is None else computed

    def _drop(
        self, index: Any, raw_index: Optional[Any] = None, truncate: bool = False
    ) -> None:
        pass

    def drop(
        self, index: Union[slice, Sequence[int]], truncate: bool = False
    ) -> None:
        """
        Remove rows by specifying their indices.

        Args:
           index: indices of rows to be dropped
           truncate: manage dropped indices:

               * if ``True`` then the dropped indices > greater (non dropped) index can be reused to index the rows added later
               * If ``False`` no dropped index will be reused

               .. warning::
                   in any case, dropped indices < greater (non dropped) index will not be reused
        """
        self._drop(index, truncate=truncate)

    @property
    def loc(self):  # type: ignore
        # NB: do not specify here the return type _Loc (prevent sphinx to show _Loc which is private)
        """
        Return a `locator` object for accessing a group of rows and columns by indices
        and column names.
        The following two syntax are allowed:

        - ``.loc[row-selection]``
        - ``.loc[row-selection, column-selection]``

        Allowed inputs for row selection are:

        - A single index
        - A list or array of indices, e.g. ``[1, 3, 5]``.
        - A slice object with indices, e.g. ``1:5``.

        Allowed inputs column row selection are:

        - A single column name or index
        - A list or array of column names or indices, e.g. ``['b', 'd', 'f']`` or ``[1, 3, 5]``.
        - A slice object with column name or indices, e.g. ``'b':'f'`` or ``1:5``.

        .. warning:: Just like **pandas** dataframes slices (but contrary to usual python slices), **both** the `start` and the `stop` bounds are included in the interval

        Examples
        --------
        **Getting values**

        >>> from progressivis import PTable
        >>> data = dict(i=[29, 45, 12, 20, 70],
        ...             j=[-95, -47, -11, -83, -68],
        ...             s=["t", "a", "b", "l", "e"],
        ...             f=[0.741, 0.0812, 0.284, 0.775, 0.884],
        ...             g=[-0.320, -0.031, -0.717, -0.863, -0.8087]
        ... )
        >>> pt = PTable("pt", data=data)
        >>> pt
        PTable("pt", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[5]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   0|          29|         -95|           t|       0.741|       -0.32|
                   1|          45|         -47|           a|      0.0812|      -0.031|
                   2|          12|         -11|           b|       0.284|      -0.717|
                   3|          20|         -83|           l|       0.775|      -0.863|
                   4|          70|         -68|           e|       0.884|     -0.8087|
        >>>

        **Single row**

        >>> pt.loc[2]
        <progressivis.table.row.Row object at 0x7f8907fe2c10>
        >>> pt.loc[2].to_dict()
        {'i': 12, 'j': -11, 's': 'b', 'f': 0.284, 'g': -0.717}
        >>>

        **Single row/single column**

        >>> pt.loc[2, "f"]
        0.284
        >>>

        **Slicing rows, keeping all columns**

        >>> pt.loc[1:3]
        BasePTable("anonymous", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[3]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   1|          45|         -47|           a|      0.0812|      -0.031|
                   2|          12|         -11|           b|       0.284|      -0.717|
                   3|          20|         -83|           l|       0.775|      -0.863|
        >>>

        **Fancy indexing on rows, keeping all columns**

        >>> pt.loc[[1, 3]]
        BasePTable("anonymous", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[2]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   1|          45|         -47|           a|      0.0812|      -0.031|
                   3|          20|         -83|           l|       0.775|      -0.863|
        >>>

        **All rows, single column**

        >>> pt.loc[:, "f"]
        BasePTable("anonymous", dshape="{f: float64}")[5]
           Index    |     f      |
                   0|       0.741|
                   1|      0.0812|
                   2|       0.284|
                   3|       0.775|
                   4|       0.884|
        >>>

        **All rows, list of names for columns**

        >>> pt.loc[:, ["j", "f"]]
        BasePTable("anonymous", dshape="{j: int32, f: float64}")[5]
           Index    |     j      |     f      |
                   0|         -95|       0.741|
                   1|         -47|      0.0812|
                   2|         -11|       0.284|
                   3|         -83|       0.775|
                   4|         -68|       0.884|
        >>>

        **All rows, list of indices for columns**

        >>> pt.loc[:, [1, 3]]
        BasePTable("anonymous", dshape="{j: int32, f: float64}")[5]
           Index    |     j      |     f      |
                   0|         -95|       0.741|
                   1|         -47|      0.0812|
                   2|         -11|       0.284|
                   3|         -83|       0.775|
                   4|         -68|       0.884|
        >>>

        **All rows, range of names (slicing) for columns**

        >>> pt.loc[:, "j":"f"]
        BasePTable("anonymous", dshape="{j: int32, s: string, f: float64}")[5]
           Index    |     j      |     s      |     f      |
                   0|         -95|           t|       0.741|
                   1|         -47|           a|      0.0812|
                   2|         -11|           b|       0.284|
                   3|         -83|           l|       0.775|
                   4|         -68|           e|       0.884|
        >>>

        **All rows, range of indices (slicing) for columns**

        >>> pt.loc[:, 1:3]
        BasePTable("anonymous", dshape="{j: int32, s: string, f: float64}")[5]
           Index    |     j      |     s      |     f      |
                   0|         -95|           t|       0.741|
                   1|         -47|           a|      0.0812|
                   2|         -11|           b|       0.284|
                   3|         -83|           l|       0.775|
                   4|         -68|           e|       0.884|
        >>>


        **Setting values**

        **Setting unique value**

        >>> pt.loc[3, "f"] = 0.0
        >>> pt
        PTable("pt", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[5]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   0|          29|         -95|           t|       0.741|       -0.32|
                   1|          45|         -47|           a|      0.0812|      -0.031|
                   2|          12|         -11|           b|       0.284|      -0.717|
                   3|          20|         -83|           l|         0.0|      -0.863|
                   4|          70|         -68|           e|       0.884|     -0.8087|
        >>>

        **Broadcasting a value over a column**


        >>> pt.loc[:, "f"] = 0.
        >>> pt
        PTable("pt", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[5]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   0|          29|         -95|           t|         0.0|       -0.32|
                   1|          45|         -47|           a|         0.0|      -0.031|
                   2|          12|         -11|           b|         0.0|      -0.717|
                   3|          20|         -83|           l|         0.0|      -0.863|
                   4|          70|         -68|           e|         0.0|     -0.8087|

        **Broadcasting a value over a list of columns**

        >>> pt.loc[:, ["i", "j"]] = 42
        >>> pt
        PTable("pt", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[5]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   0|          42|          42|           t|         0.0|       -0.32|
                   1|          42|          42|           a|         0.0|      -0.031|
                   2|          42|          42|           b|         0.0|      -0.717|
                   3|          42|          42|           l|         0.0|      -0.863|
                   4|          42|          42|           e|         0.0|     -0.8087|

        **Setting a row**

        >>> pt.loc[2, :] = [42, -42, "B", 4.2, -4.2]
        >>> pt
        PTable("pt", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[5]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   0|          42|          42|           t|         0.0|       -0.32|
                   1|          42|          42|           a|         0.0|      -0.031|
                   2|          42|         -42|           B|         4.2|        -4.2|
                   3|          42|          42|           l|         0.0|      -0.863|
                   4|          42|          42|           e|         0.0|     -0.8087|
        >>>

        **Setting some columns in a row**

        >>> pt.loc[3, ["i", "s"]] = [0, "L"]
        >>> pt
        PTable("pt", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[5]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   0|          42|          42|           t|         0.0|       -0.32|
                   1|          42|          42|           a|         0.0|      -0.031|
                   2|          42|         -42|           B|         4.2|        -4.2|
                   3|           0|          42|           L|         0.0|      -0.863|
                   4|          42|          42|           e|         0.0|     -0.8087|
        >>>

        **Setting many values in a column**

        >>> pt.loc[1:3, "i"] = [43, 44, 45]
        >>> pt
        PTable("pt", dshape="{i: int32, j: int32, s: string, f: float64, g: float64}")[5]
           Index    |     i      |     j      |     s      |     f      |     g      |
                   0|          42|          42|           t|         0.0|       -0.32|
                   1|          43|          42|           a|         0.0|      -0.031|
                   2|          44|         -42|           B|         4.2|        -4.2|
                   3|          45|          42|           L|         0.0|      -0.863|
                   4|          42|          42|           e|         0.0|     -0.8087|
        >>>

        """
        return self._loc

    @property
    def at(self):  # type: ignore
        # NB: do not specify here the return type _At (prevent sphinx to show _At)
        # pylint: disable=invalid-name
        """
        Return an object for indexing values using ids
        """
        return self._at

    def __repr__(self) -> str:
        return str(self) + self.info_contents()

    def __str__(self) -> str:
        classname = self.__class__.__name__
        length = len(self)
        return '%s("%s", dshape="%s")[%d]' % (
            classname,
            self.name,
            dshape_print(self.dshape),
            length,
        )

    def get_original_base(self) -> BasePTable:
        if self._base is None:
            return self
        return self._base

    def info_row(self, row: int, width: int) -> str:
        "Return a description for a row, used in `repr`"
        row_id = row if row in self.index else -1
        rep = "{0:{width}}|".format(row_id, width=width)
        for name in self.columns:
            col = self[name]
            try:
                v = str(col[row])
            except ValueError:
                if row_id == -1:
                    v = "?" * width
                else:
                    raise
            if len(v) > width:
                if col.dshape == "string":
                    v = v[0 : width - 3] + "..."
                else:
                    v = v[0 : width - 1] + "."
            rep += "{0:>{width}}|".format(v, width=width)
        return rep

    def info_contents(self) -> str:
        "Return a description of the contents of this table"
        length = len(self)
        rep = ""
        max_rows = min(length, get_option("display.max_rows"))
        if max_rows == 0:
            return ""
        if max_rows < length:
            head = max_rows // 2
            tail = head
        else:
            head = length
            tail = None
        width = get_option("display.column_space")

        rep += "\n{0:^{width}}|".format("Index", width=width)
        for name in self.columns:
            if len(name) > width:
                name = name[0:width]
            rep += "{0:^{width}}|".format(name, width=width)

        for row in self.index[:head]:
            rep += "\n"
            rep += self.info_row(row, width)

        if tail:
            rep += "\n...(%d)..." % length
            for row in self.index[-tail:]:
                rep += "\n"
                rep += self.info_row(row, width)
        return rep

    def info_raw_contents(self) -> str:
        "Return a description of the contents of this table including unselected rows"
        length = self.last_id + 1  # len(self)
        rep = ""
        max_rows = min(length, get_option("display.max_rows"))
        if max_rows == 0:
            return ""
        if max_rows < length:
            head = max_rows // 2
            tail = length - max_rows // 2
        else:
            head = length
            tail = None
        width = get_option("display.column_space")

        rep += "\n{0:^{width}}|".format("Index", width=width)
        for name in self.columns:
            if len(name) > width:
                name = name[0:width]
            rep += "{0:^{width}}|".format(name, width=width)

        for row in range(head):
            rep += "\n"
            rep += self.info_row(row, width)

        if tail:
            rep += "\n...(%d)..." % length
            for row in range(tail, length):
                rep += "\n"
                rep += self.info_row(row, width)
        return rep

    def index_to_mask(self) -> np.ndarray[Any, Any]:
        return np.array(((elt in self.index) for elt in range(self.last_id + 1)))

    def index_to_array(self) -> np.ndarray[Any, Any]:
        return np.array(self.index, dtype="int32")

    def __iter__(self) -> Iterator[str]:
        return iter(self._columndict.keys())

    @property
    def size(self) -> int:
        "Return the size of this table, which is the number of rows"
        return self.nrow

    @property
    def is_identity(self) -> bool:
        "Return True if the index is using the identity mapping"
        sl = self.index.to_slice_maybe()
        if not isinstance(sl, slice):
            return False
        # return sl == slice(0, self.last_id+1, None)
        return bool(sl.start == 0)

    @property
    def last_id(self) -> int:
        "Return the last id of this table"
        # if not self._index:
        #    assert self._last_id == -1
        assert self._base
        return self._base.last_id

    @property
    def last_xid(self) -> int:
        "Return the last eXisting id of this table"
        assert self._base
        return self._base.last_xid  # only for refreshing self._last_id

    def width(self, colnames: Optional[List[Union[int, str]]] = None) -> int:
        """Return the number of effective width (number of columns) of the table

        Since a column can be multidimensional, the effective width of a table
        is the sum of the effective width of each of its columns.

        Parameters
        ----------
        colnames : list or `None`
            The optional list of columns to use for counting, or all the
            columns when not specified or `None`.
        """
        columns = (
            self._columns if colnames is None else [self[name] for name in colnames]
        )
        width = 0
        for col in columns:
            width += col.shape[1] if len(col.shape) > 1 else 1
        return width

    @property
    def shape(self) -> Shape:
        "Return the shape of this table as if it were a numpy array"
        return self.size, self.width()

    def to_json(self, **kwds: Any) -> Dict[str, Any]:
        "Return a dictionary describing the contents of this columns."
        if "orient" not in kwds:
            self.to_dict(orient="dict", **kwds)
        elif kwds["orient"] == "dict":
            self.to_dict(**kwds)
        return self.to_dict(**kwds)  # type: ignore

    def make_projection(
        self, cols: Optional[List[str]], index: Any
    ) -> Tuple[List[BasePColumn], Dict[str, int]]:
        from .column_selected import PColumnSelectedView

        dict_ = self._make_columndict_projection(cols)
        columns: List[BasePColumn] = [
            PColumnSelectedView(base=c, index=index)
            for (i, c) in enumerate(self._columns)
            if i in dict_.values()
        ]
        columndict: Dict[str, int] = dict(zip(dict_.keys(), range(len(dict_))))
        if not self.computed:
            return columns, columndict
        cols_as_set: Set[str] = set()
        if is_none_alike(cols):
            cols_as_set = set(self.computed.keys())
        elif isinstance(cols, str):
            cols_as_set = set([cols])
        elif isinstance(cols, Iterable) and all_string(cols):
            cols_as_set = set(cols)
        else:
            logger.warning(f"computed columns will be ignored with selection {cols}")
        comp_cols: List[BasePColumn] = [
            meta._make_computed(index, aka, self)
            for (aka, meta) in self.computed.items()
            if aka in cols_as_set and aka not in self._columndict.keys()
        ]
        columns += comp_cols
        cc_dict: Dict[str, int] = {
            k: i
            for (i, k) in enumerate(
                [e for e in self.computed.keys() if e in cols_as_set], len(dict_)
            )
        }
        columndict.update(cc_dict)
        return columns, columndict

    def _make_columndict_projection(
        self,
        cols: Union[
            None, slice, int, str, List[str], List[Union[int, np.integer[Any]]]
        ],
    ) -> Dict[str, int]:
        if is_none_alike(cols):
            return self._columndict
        if isinstance(cols, slice):
            if is_int(cols.start):
                assert is_int(cols.stop) or cols.stop is None
            else:
                assert cols.start is None or (is_str(cols.start) and cols.start in self._columndict)
                assert cols.stop is None or (is_str(cols.stop) and cols.stop in self._columndict)
                cols = slice(self._columndict[cols.start] if is_str(cols.start) else None,
                             self._columndict[cols.stop] if is_str(cols.stop) else None)
            stop_ = cols.stop
            if stop_ is None:
                stop_ = len(self._columndict)
            else:
                stop_ += 1  # pandas alike slicing
                assert stop_ <= len(self._columndict)
            nsl = norm_slice(cols, stop=stop_)
            return dict(
                {
                    k: v
                    for (i, (k, v)) in enumerate(self._columndict.items())
                    if i in range(*nsl.indices(nsl.stop))
                }
            )
        if isinstance(cols, (int, integer_types)) or isinstance(cols, str):
            cols = [cols]  # type: ignore
        if isinstance(cols, Iterable):
            if all_int(cols):
                return dict(
                    {
                        k: v
                        for (i, (k, v)) in enumerate(self._columndict.items())
                        if i in cols
                    }
                )
            if all_string(cols):
                return dict({k: v for (k, v) in self._columndict.items() if k in cols})
        raise ValueError(f"Invalid column projection {cols}")

    @overload
    def to_dict(
        self,
        orient: Union[Literal["dict"], Literal["split"]] = "dict",
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ...

    @overload
    def to_dict(
        self,
        orient: Union[
            Literal["list"], Literal["rows"], Literal["datatable"], Literal["records"]
        ],
        columns: Optional[List[str]] = None,
    ) -> List[Any]:
        ...

    @overload
    def to_dict(
        self, orient: Literal["index"], columns: Optional[List[str]] = None
    ) -> Dict[int, Any]:
        ...

    def to_dict(self, orient: str = "dict", columns: Optional[List[str]] = None) -> Any:
        # pylint: disable=too-many-branches
        """
        Return a dictionary describing the contents of this columns.

        Parameters
        ----------
        orient : {'dict', 'list', 'split', 'rows', 'datatable', 'records', 'index'}
            TODO
        columns : list or `None`
            TODO
        """
        ret: Dict[str, Any]
        ret2: List[Any]
        if columns is None:
            columns = self.columns
        if orient == "dict":
            ret = dict()
            for name in columns:
                col = self[name]
                ret[name] = {
                    int(k): v for (k, v) in dict(zip(self.index, col.tolist())).items()
                }  # because a custom JSONEncoder cannot fix it
            return ret
        if orient == "list":
            ret = dict()
            for name in columns:
                col = self[name]
                ret[name] = col.tolist()
            return ret
        if orient == "split":
            ret = {"index": list(self.index), "columns": columns}
            data = []
            cols = [self[c] for c in columns]
            for i in self.index:
                line = []
                for col in cols:
                    line.append(get_physical_base(col).loc[i])
                data.append(line)
            ret["data"] = data
            return ret
        if orient == "datatable":
            # not a pandas compliant mode but useful for JS DataPTable
            ret2 = []
            for i in self.index:
                line = [i]
                for name in columns:
                    col = self[name]
                    line.append(remove_nan(get_physical_base(col).loc[i]))
                ret2.append(line)
            return ret2
        if orient in ("rows", "records"):
            ret2 = []
            for i in self.index:
                line2 = {}
                for name in columns:
                    col = self[name]
                    line2[name] = get_physical_base(col).loc[i]
                ret2.append(line2)
            return ret2
        if orient == "index":
            ret3: Dict[int, Any] = dict()
            for id_ in self.index:
                line2 = {}
                for name in columns:
                    col = self[name]
                    line2[name] = col.loc[id_]
                ret3[int(id_)] = line2
            return ret3
        raise ValueError(f"to_dict({orient}) not implemented")

    def to_csv(
        self, filename: str, columns: Optional[List[str]] = None, sep: str = ","
    ) -> None:  # TODO: to be improved
        if columns is None:
            columns = self.columns
        with open(filename, "wb") as f:
            for i in self.index:
                row = []
                for name in columns:
                    col = self[name]
                    row.append(str(remove_nan(get_physical_base(col).loc[i])))
                f.write(sep.join(row).encode("utf-8"))
                f.write(b"\n")

    def to_df(self, to_datetime: Sequence[str] = ()) -> pd.DataFrame:
        def _proc(col: str) -> Any:
            if col in to_datetime:
                return _to_datetime(self[col].loc[:])
            return self[col].loc[:]

        return pd.DataFrame({col: _proc(col) for col in self.columns})

    def column_offsets(
        self, columns: List[str], shapes: Optional[List[Tuple[int, ...]]] = None
    ) -> List[int]:
        """Return the offsets of each column considering columns can have
        multiple dimensions
        """
        if shapes is None:
            shapes = [self[c].shape for c in columns]
        offsets = [0]
        dim2 = 0
        for shape in shapes:
            dims = len(shape)
            if dims > 2:
                raise ValueError(
                    "Cannot convert table to numpy array because" "of shape %s", shape
                )
            dim2 += dims
            offsets.append(dim2)
        return offsets

    @property
    def columns(self) -> List[str]:
        "Return the list of column names in this table"
        return list(self._columndict.keys())

    def _column(self, name: ColIndexer) -> BasePColumn:
        if isinstance(name, integer_types):
            return self._columns[name]
        return self._columns[self._columndict[name]]

    def column_index(self, name: Union[str, int]) -> int:
        "Return the index of the specified column in this table"
        if isinstance(name, integer_types):
            return name
        return self._columndict[name]

    def index_to_id(self, ix: Any) -> Any:
        """Return the ids of the specified indices
        NB: useless for this implementation. kept for compat.

        Parameters
        ----------
        ix: the specification of an index or a list of indices
            The list can be specified with multiple formats: integer, list,
            numpy array, Iterable, or slice.  A similar format is return,
            except that slices and Iterables may return expanded as lists or
            arrays.
        """
        if is_int(ix):
            return ix
        locs = self._any_to_pintset(ix)
        return locs

    def id_to_index(self, loc: Any, as_slice: bool = True) -> Any:
        # to be reimplemented with LRU-dict+pyroaring
        """Return the indices of the specified id or ids
        NB: useless for this implementation. kept for compat.

        Parameters
        ----------
        loc : an id or list of ids
            The format can be: integer, list, numpy array, Iterable, or slice.
            Note that a PIntSet is an list, and array, and a PIntSet are all
            Iterables but are managed in an efficient way.
        as_slice : boolean
            If True, try to convert the result into a slice if possible and
            not too expensive.
        """
        return self.index_to_id(loc)

    def _compute_index(self) -> PIntSet:
        assert self._base is not None
        res = self._base._any_to_pintset(self._selection)
        prev = self._masked
        while prev is not None:
            bm = self._base._any_to_pintset(prev._selection)
            res &= bm
            prev = prev._masked
        return res

    @property
    def index(self) -> PIntSet:
        "Return the object in change of indexing this table"
        # return self._base._any_to_pintset(self._mask)
        return self._compute_index()

    @property
    def ncol(self) -> int:
        "Return the number of columns (same as `len(table.columns()`)"
        return len(self._columns)

    @property
    def nrow(self) -> int:
        "Return the number of rows (same as `len(table)`)"
        return len(self.index)

    def __len__(self) -> int:
        return self.nrow

    def _slice_to_pintset(
        self, sl: slice, fix_loc: bool = True, existing_only: bool = True
    ) -> PIntSet:
        stop = sl.stop or self.last_xid
        nsl = norm_slice(sl, fix_loc, stop=stop)
        ret = PIntSet(nsl)
        if existing_only:
            ret &= self.index
        return ret

    def _any_to_pintset(
        self,
        locs: Union[PIntSet, int, np.integer[Any], Iterable[Any], slice],
        copy: bool = True,
        fix_loc: bool = True,
        existing_only: bool = True,
    ) -> PIntSet:
        if isinstance(locs, PIntSet):
            return locs[:] if copy else locs
        if isinstance(locs, integer_types):
            return PIntSet([locs])
        if isinstance(locs, Iterable):
            if all_bool(locs):
                assert isinstance(locs, Iterable)
                return PIntSet(np.nonzero(locs))  # type: ignore
            else:
                return PIntSet(locs)
        if isinstance(locs, slice):
            return self._slice_to_pintset(locs, fix_loc, existing_only)
        raise KeyError(f"Invalid type {type(locs)} for key {locs}")

    @property
    def name(self) -> str:
        "Return the name of this table"
        return "anonymous"

    @property
    def dshape(self) -> DataShape:
        """
        Return the datashape of this table
        """
        return self._dshape

    @property
    def base(self) -> Optional[BasePTable]:
        "Return the base table for views, or None if the table is not a view"
        return self._base

    @property
    def changes(self) -> Optional[PTableChanges]:
        "Return the PTableChange manager associated with this table or None"
        return self._changes

    @changes.setter
    def changes(self, tablechange: Optional[PTableChanges]) -> None:
        "Set the PTableChange manager, or unset with None"
        self._changes = tablechange

    def reset_updates(self, mid: str) -> None:
        if self._changes:
            self._changes.reset(mid)

    def compute_updates(
        self, start: int, now: int, mid: str, cleanup: bool = True
    ) -> Optional[IndexUpdate]:
        """Compute the updates (delta) that happened to this table since the last call.

        Parameters
        ----------
        start:
            Start is interpreted as a virtual time for `last time`
        now:
            Start is interpreted as a virtual time for `now`
        mid:
            An identifier for the object that will ask for updates,
            usually the name of a slot.

        Returns
        -------
        :
            None or an IndexUpdate structure which describes the list of rows created, updated, and deleted.
        """
        if self._changes:
            self._flush_cache()
            updates = self._changes.compute_updates(start, now, mid, cleanup=cleanup)
            if updates is None:
                updates = IndexUpdate(created=PIntSet(self.index))
            return updates
        return None

    @overload
    def __getitem__(self, key: Union[int, str]) -> BasePColumn:
        ...

    @overload
    def __getitem__(
        self,
        key: Union[
            List[Any], Tuple[Any, Any], np.ndarray[Any, Any], slice, Iterable[int]
        ],
    ) -> Tuple[BasePColumn, ...]:
        ...

    def __getitem__(self, key: Any) -> Any:
        # hack, use t[['a', 'b'], 1] to get a list instead of a PTableView
        fast = False
        if isinstance(key, tuple):
            key = key[0]  # i.e. columns
            fast = True
        if isinstance(key, (str, integer_types)):
            return self._column(key)
        elif isinstance(key, Iterable):
            if fast:
                return (self._column(c) for c in key)
        elif isinstance(key, slice):
            if fast:
                indices = self._col_slice_to_indices(key)
                return (self._column(c) for c in range(*indices))
        raise ValueError('getitem not implemented for key "%s"' % key)

    def row(self, loc: int) -> Optional[Row]:
        "Return a Row object wrapping the loc"
        return self.last(loc)

    def iterrows(self) -> Iterator[Optional[Row]]:
        "Return an iterator returning rows and their ids"
        raise NotImplementedError("iterrow not implemented in BasePTable")

    @overload
    def last(self, key: Optional[int] = None) -> Optional[Row]:
        ...

    @overload
    def last(self, key: str) -> Any:
        ...

    @overload
    def last(self, key: Sequence[Union[int, str]]) -> List[Any]:
        ...

    def last(self, key: Any = None) -> Any:
        "Return the last row"
        length = len(self)
        if length == 0:
            # raise KeyError("No value in table")
            return None
        if key is None or isinstance(key, integer_types):
            from .row import Row

            return Row(self, key if key is None else int(key))
        if isinstance(key, str):
            return self._column(key)[self.last_xid]
        if all_string_or_int(key):
            index = self.last_xid
            return [self._column(c)[index] for c in key]
        raise ValueError('last not implemented for key "%s"' % key)

    def __delitem__(self, key: Indexer) -> None:
        bm = self._any_to_pintset(key, fix_loc=False, existing_only=False)
        if not bm:
            return
        res = self._any_to_pintset(self._selection)
        # if not (bm in self.index or isinstance(key, slice)):
        #    raise ValueError('Invalid locs')
        if isinstance(key, slice):
            if not (bm.min() in self.index and bm.max() in res):
                raise ValueError("Invalid locs")  # when key is a slice we accept holes
        elif bm not in self.index:  # when key is not a slice it must be exhaustive
            raise ValueError("Invalid locs")
        if isinstance(key, Iterable):
            assert bm in res
        res -= bm
        self._selection = res

    def setitem_2d(
        self, rowkey: Any, colkey: Union[ColIndexer, Iterable[int], slice], values: Any
    ) -> None:
        if isinstance(colkey, (str, integer_types)):
            self._setitem_key(colkey, rowkey, values)
        elif isinstance(colkey, Iterable):
            self._setitem_iterable(colkey, rowkey, values)
        elif isinstance(colkey, slice):
            self._setitem_slice(colkey, rowkey, values)
        else:
            raise ValueError("Unhandled key %s", colkey)

    @overload
    def __setitem__(
        self,
        key: Union[int, str],
        values: Union[BasePColumn, np.ndarray[Any, Any], Iterable[Any]],
    ) -> None:
        ...

    @overload
    def __setitem__(
        self,
        key: Union[
            List[Any], Tuple[Any, Any], np.ndarray[Any, Any], slice, Iterable[int]
        ],
        values: Union[
            np.ndarray[Any, Any],
            BasePTable,
            Sequence[Union[BasePColumn, np.ndarray[Any, Any], Iterable[Any]]],
        ],
    ) -> None:
        ...

    def __setitem__(self, colkey: Any, values: Any) -> None:
        if isinstance(colkey, tuple):
            raise ValueError(
                "Adding new columns ({}) via __setitem__"
                " not implemented".format(colkey)
            )
        if isinstance(colkey, (str, integer_types)):
            # NB: on Pandas, only strings are accepted!
            self._setitem_key(colkey, None, values)
        elif isinstance(colkey, Iterable):
            if not all_string_or_int(colkey):
                raise ValueError("setitem not implemented for %s key" % colkey)
            self._setitem_iterable(colkey, None, values)
        elif isinstance(colkey, slice):
            self._setitem_slice(colkey, None, values)
        else:
            raise ValueError("Unhandled key %s", colkey)

    def _setitem_key(self, colkey: ColIndexer, rowkey: Indexer, values: Any) -> None:
        if is_none_alike(rowkey) and len(values) != len(self):
            raise ValueError(
                "Length of values (%d) different "
                "than length of table (%d)" % (len(values), len(self))
            )
        column = self._column(colkey)
        if is_none_alike(rowkey):
            column[self.index] = values
        else:
            column[rowkey] = values

    def _setitem_iterable(
        self, colkey: Iterable[Union[int, str]], rowkey: Indexer, values: Any
    ) -> None:
        # pylint: disable=too-many-branches
        colnames: List[Union[int, str]] = list(colkey)
        len_colnames = len(colnames)
        if not isinstance(values, Iterable):
            values = np.repeat(values, len_colnames)
        if isinstance(values, Mapping):
            for (k, v) in values.items():
                column = self._column(k)
                if is_none_alike(rowkey):
                    column[self.index] = v
                else:
                    column[rowkey] = v
        elif hasattr(values, "shape"):
            shape = values.shape
            if len(shape) > 1 and shape[1] != self.width(colnames):
                # and not isinstance(values, BasePTable):
                raise ValueError(
                    "Shape [1] (width)) of columns and " "value shape do not match"
                )

            if rowkey is None:
                rowkey = self.index.to_slice_maybe()  # slice(None, None)
            for i, colname in enumerate(colnames):
                column = self._column(colname)
                if len(column.shape) > 1:
                    wid = column.shape[1]
                    column[rowkey, 0:wid] = values[i : i + wid]
                else:  # i.e. len(column.shape) == 1
                    if isinstance(values, BasePTable):
                        column[rowkey] = values[i]
                    elif len(shape) == 1:  # values is a row
                        column[rowkey] = values[i]
                    else:
                        column[rowkey] = values[:, i]
        else:
            for i, colname, v in zip(range(len_colnames), colnames, values):
                column = self._column(colname)
                if is_none_alike(rowkey):
                    column[self.index] = v
                else:
                    column[rowkey] = values[i]

    def _col_slice_to_indices(self, colkey: slice) -> range:
        if isinstance(colkey.start, str):
            start = self.column_index(colkey.start)
            end = self.column_index(colkey.stop)
            colkey = slice(start, end + 1, colkey.step)
        return range(*colkey.indices(self.ncol))

    def _setitem_slice(self, colkey: slice, rowkey: Indexer, values: Any) -> None:
        indices = self._col_slice_to_indices(colkey)
        self._setitem_iterable(indices, rowkey, values)

    def columns_common_dtype(
        self, columns: Optional[List[str]] = None
    ) -> np.dtype[Any]:
        """Return the dtype that BasePTable.to_array would return.

        Parameters
        ----------
        columns: a list or None
            the columns to extract or, if None, all the table columns
        """
        if columns is None:
            columns = self.columns
        dtypes = [self[c].dtype for c in columns]
        return np.result_type(*dtypes)

    def to_array(
        self,
        locs: Indexer = None,
        columns: Optional[List[str]] = None,
        # returns_indices=False,
        ret: Optional[np.ndarray[Any, Any]] = None,
    ) -> np.ndarray[Any, Any]:
        """Convert this table to a numpy array

        Parameters
        ----------
        locs:
            The rows to extract.  Locs can be specified with multiple formats:
            integer, list, numpy array, Iterable, or slice.
        columns:
            the columns to extract or, if None, all the table columns
        return_indices:
            if True, returns a tuple with the indices of the returned values
            as indices, followed by the array
        ret:
            if None, the returned array is allocated, otherwise, ret is reused.
            It should be an array of the right dtype and size otherwise it is
            ignored.
        """
        if columns is None:
            columns = self.columns
        assert columns is not None
        shapes = [self[c].shape for c in columns]
        offsets = self.column_offsets(columns, shapes)
        dtype = self.columns_common_dtype(columns)
        indices = None
        # TODO split the copy in chunks
        if locs is None:
            indices = self.index
        elif isinstance(locs, slice):
            indices = self._slice_to_pintset(locs)
            # indices = self._any_to_pintset(locs)
        else:
            indices = locs
        shape: Shape = (indices_len(indices), offsets[-1])
        arr: np.ndarray[Any, Any]
        if isinstance(ret, np.ndarray) and ret.shape == shape and ret.dtype == dtype:
            arr = ret
        else:
            arr = np.empty(shape, dtype=dtype)
        for i, column in enumerate(columns):
            col = self._column(column)
            shape = shapes[i]
            if len(shape) == 1:
                col.read_direct(arr, indices, dest_sel=np.s_[:, offsets[i]])
            else:
                col.read_direct(
                    arr, indices, dest_sel=np.s_[:, offsets[i] : offsets[i + 1]]
                )
        # if returns_indices:
        #     return indices, arr
        return arr

    def unary(self, op: Any, **kwargs: Any) -> UnaryRet:
        axis = kwargs.get("axis", 0)
        # get() is cheaper than pop(), it avoids to update unused kwargs
        keepdims = kwargs.get("keepdims", False)
        # ignore other kwargs, maybe raise error in the future
        res: Dict[str, Any] = dict()
        for col in self._columns:
            value = op(col.values, axis=axis, keepdims=keepdims)
            res[col.name] = value
        return res

    def raw_unary(self, op: Any, **kwargs: Any) -> Dict[str, Any]:
        res: Dict[str, Any] = dict()
        for col in self._columns:
            value = op(col.values, **kwargs)
            res[col.name] = value
        return res

    def binary(
        self,
        op: Callable[
            [np.ndarray[Any, Any], Union[np.ndarray[Any, Any], int, float, bool]],
            np.ndarray[Any, Any],
        ],
        other: BasePTable,
        **kwargs: Any,
    ) -> BinaryRet:
        axis = kwargs.pop("axis", 0)
        assert axis == 0
        res: Dict[str, np.ndarray[Any, Any]] = {}
        if isinstance(other, (np.ndarray, int, float, bool)):
            for col in self._columns:
                name = col.name
                value = op(col, other)
                res[name] = value
        else:
            for col in self._columns:
                name = col.name
                value = op(col.value, other[name].value)
                res[name] = value
        return res

    def __abs__(self, **kwargs: Any) -> UnaryRet:
        return self.unary(np.abs, **kwargs)

    def __add__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.add, other)

    def __radd__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.add, self)

    def __and__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.and_, other)

    def __rand__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.and_, self)

    # def __div__(self, other):
    #     return self.binary(operator.div, other)

    # def __rdiv__(self, other):
    #     return other.binary(operator.div, self)

    def __eq__(self, other: Any) -> Any:
        if not isinstance(other, BasePTable):
            return False
        return self.binary(operator.eq, other)

    def __gt__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.gt, other)

    def __ge__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.ge, other)

    def __invert__(self) -> UnaryRet:
        return self.unary(np.invert)

    def __lshift__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.lshift, other)

    def __rlshift__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.lshift, self)

    def __lt__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.lt, other)

    def __le__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.le, other)

    def __mod__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.mod, other)

    def __rmod__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.mod, self)

    def __mul__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.mul, other)

    def __rmul__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.mul, self)

    def __ne__(self, other: Any) -> Any:
        if not isinstance(other, BasePTable):
            return False
        return self.binary(operator.ne, other)

    def __neg__(self) -> Dict[str, Any]:
        return self.unary(np.negative)

    def __or__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.or_, other)

    def __pos__(self) -> BasePTable:
        return self

    def __ror__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.or_, self)

    def __pow__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.pow, other)

    def __rpow__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.pow, self)

    def __rshift__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.rshift, other)

    def __rrshift__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.rshift, self)

    def __sub__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.sub, other)

    def __rsub__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.sub, self)

    def __truediv__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.truediv, other)

    def __rtruediv__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.truediv, self)

    def __floordiv__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.floordiv, other)

    def __rfloordiv__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.floordiv, self)

    def __xor__(self, other: BasePTable) -> BinaryRet:
        return self.binary(operator.xor, other)

    def __rxor__(self, other: BasePTable) -> BinaryRet:
        return other.binary(operator.xor, self)

    def any(self, **kwargs: Any) -> UnaryRet:
        return self.unary(np.any, **kwargs)

    def all(self, **kwargs: Any) -> UnaryRet:
        return self.unary(np.all, **kwargs)

    def min(self, **kwargs: Any) -> UnaryRet:
        return self.unary(np.nanmin, **kwargs)  # avoids propagation of Nan

    def max(self, **kwargs: Any) -> UnaryRet:
        return self.unary(np.nanmax, **kwargs)  # avoids propagation of Nan

    def var(self, **kwargs: Any) -> UnaryRet:
        return self.raw_unary(np.var, **kwargs)

    def argmin(self, **kwargs: Any) -> UnaryRet:
        argmin_ = self.raw_unary(np.argmin, **kwargs)
        if self.is_identity:
            return argmin_
        index_array = self.index_to_array()
        for k, v in argmin_.items():
            argmin_[k] = index_array[v]
        return argmin_

    def argmax(self, **kwargs: Any) -> UnaryRet:
        argmax_ = self.raw_unary(np.argmax, **kwargs)
        if self.is_identity:
            return argmax_
        index_array = self.index_to_array()
        for k, v in argmax_.items():
            argmax_[k] = index_array[v]
        return argmax_

    def idxmin(self, **kwargs: Any) -> UnaryRet:
        res = self.argmin(**kwargs)
        for c, ix in res.items():
            res[c] = self.index_to_id(ix)
        return res

    def idxmax(self, **kwargs: Any) -> UnaryRet:
        res = self.argmax(**kwargs)
        for c, ix in res.items():
            res[c] = self.index_to_id(ix)
        return res

    def remove_module(self, mid: str) -> None:
        # TODO
        pass

    def _normalize_locs(self, locs: Indexer) -> PIntSet:
        return self._any_to_pintset(locs)
        """if locs is None:
            if bool(self._freelist):
                locs = iter(self)
            else:
                locs = iter(self.dataset)
        elif isinstance(locs, integer_types):
            locs = [locs]
        return PIntSet(locs)"""

    def equals(self, other: BasePTable) -> bool:
        if self is other:
            return True
        return bool(np.all(self._columns == other._columns))

    def get_panene_data(
        self, cols: Optional[List[str]] = None
    ) -> List[np.ndarray[Any, Any]]:
        if cols is None:
            cols = self.columns
        return [self[key].values for key in cols]

    def cxx_api_raw_cols(self, cols: Optional[List[str]] = None) -> Tuple[Any, Any]:
        tbl = self.base or self
        if cols is None:
            cols = tbl.columns
        return cols, [tbl[c].dataset.base for c in cols]  # type: ignore

    def cxx_api_raw_cols2(self, cols: Optional[List[str]] = None) -> List[Any]:
        tbl = self.base or self
        if cols is None:
            cols = tbl.columns
        return [tbl[c].dataset.base for c in cols]  # type: ignore

    def cxx_api_info_index(self) -> Tuple[bool, PIntSet, int]:
        # ix = Int64HashPTable() if self.is_identity else self._ids._ids_dict._ht
        # return self.is_identity, ix, self.last_id
        return False, self.index, self.last_id

    def _flush_cache(self) -> None:
        pass


class IndexPTable(BasePTable):
    """
    Base class for physical tables (currently :class:`PTable <progressivis.PTable>`)

    It implements index management.

    .. warning::
        Do not instanciate this class directly!

    """
    def __init__(self, index: Optional[PIntSet] = None) -> None:
        super().__init__()
        self._index: PIntSet = PIntSet() if index is None else index
        self._cached_index: Optional[PIntSet] = None  # hack
        self._last_id: int = -1
        self._changes = None

    # begin(Change management)
    def _flush_cache(self) -> None:
        self._cached_index = None

    def touch(self, index: Optional[PIntSet] = None) -> None:
        if index is self._cached_index:
            return
        self._cached_index = index
        self.add_updated(index)

    @property
    def index(self) -> PIntSet:
        "Return the object in change of indexing this table"
        return PIntSet(self._index)  # returns a copy to prevent unwanted

    @index.setter
    def index(self, indices: Any) -> None:
        "Modify the object in change of indexing this table"
        raise NotImplementedError("Cannot change index")
        # indices = self._any_to_pintset(indices)
        # if indices not in self._observed.index:
        #     raise ValueError(f"Not existing indices {indices-self._observed.index}")
        # created_ = indices - self._index
        # if created_:
        #     self.add_created(created_)
        # deleted_ = self._index - indices
        # if deleted_:
        #     self.add_deleted(deleted_)
        # self._index = indices
        # return self._index

    @property
    def last_id(self) -> int:
        "Return the last id of this table"
        # if not self._index:
        #    assert self._last_id == -1
        if self.index and self._last_id < self.index.max():
            self._last_id = self.index.max()
        return self._last_id

    def iterrows(self) -> Iterator[Optional[Row]]:
        "Return an iterator returning rows and their ids"
        return map(self.row, iter(self._index))

    @property
    def last_xid(self) -> int:
        "Return the last eXisting id of this table"
        self.last_id  # only for refreshing self._last_id
        return self.index.max()

    def add_created(self, locs: Any) -> None:
        # self.notify_observers('created', locs)
        # TODO simplify tablechanges to ignore add_created etc. when no bookmark exist
        if self._changes is None:
            self._changes = PTableChanges()
        locs = self._normalize_locs(locs)
        self._changes.add_created(locs)

    def add_updated(self, locs: Any) -> None:
        # self.notify_observers('updated', locs)
        if self._changes is None:
            self._changes = PTableChanges()
        locs = self._normalize_locs(locs)
        self._changes.add_updated(locs)

    def add_deleted(self, locs: Any) -> None:
        # self.notify_observers('deleted', locs)
        if self._changes is None:
            self._changes = PTableChanges()
        locs = self._normalize_locs(locs)
        self._changes.add_deleted(locs)

    def __delitem__(self, key: Any) -> None:
        bm = self._any_to_pintset(key, fix_loc=False, existing_only=False)
        if not bm:
            return
        # if not (bm in self.index or isinstance(key, slice)):
        #    raise ValueError('Invalid locs')
        if isinstance(key, slice):
            if not (bm.min() in self._index and bm.max() in self._index):
                raise ValueError("Invalid locs")  # when key is a slice we accept holes
        elif bm not in self.index:  # when key is not a slice it must be exhaustive
            raise ValueError("Invalid locs")
        if isinstance(key, Iterable):
            assert bm in self._index
        self._index -= bm
        self.add_deleted(bm)

    def _drop(
        self, index: Any, raw_index: Optional[Any] = None, truncate: bool = False
    ) -> None:
        """
        Args:
           index: is a normalized index. It is always a sequence or a slice
           raw_index: is the index as it is defined by the caller. It could be an integer
           truncate: if ``True`` then the dropped indices > last_id can be reused to index
            the rows added later.
           if ``False`` no dropped index will be reused
           **NB:** in any case, dropped indices < last_id will not be reused
        """
        if raw_index is None:
            raw_index = index
        if isinstance(raw_index, (int, np.integer)):
            # self.__delitem__(raw_index)
            self._index.remove(int(raw_index))
        else:
            # self.__delitem__(index)
            index = self._any_to_pintset(index)
            self._index -= index
        if truncate:  # useful 4 csv recovery
            self._last_id = self._index.max() if self._index else -1
        self.add_deleted(index)
        if self._storagegroup is not None:  # type: ignore
            self._storagegroup.release(index)  # type: ignore

    def _resize_rows(self, newsize: int, index: Optional[Any] = None) -> None:
        # self._ids.resize(newsize, index)
        created = PIntSet()
        if index is not None:
            index = self._any_to_pintset(index)
            created = index - self._index
            if index and index.min() > self.last_id:
                self._index |= index
            else:
                # assert index in self._index #TODO: check with JDF
                # self._index = index
                self._index |= index
        else:
            # assert self._is_identity
            if newsize >= self.last_id + 1:
                new_ids = PIntSet(range(self.last_id + 1, newsize))
                created = new_ids - self._index
                self._index |= new_ids
            else:
                self._index &= PIntSet(range(0, newsize))
        if created:
            self.add_created(created)


class PTableSelectedView(BasePTable):
    """
    Virtual table built on top of a :class:`PTable <progressivis.PTable>` or a :class:`PTableSelectedView <progressivis.PTableSelectedView>`
    """
    def __init__(
        self,
        base: BasePTable,
        selection: Union[PIntSet, slice] = slice(0, None),
        columns: Optional[List[str]] = None,
        computed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            base: the table (stored or virtual) on which the current view is built
            selection: indices to be part of the view
            columns: selection of columns to be included in the view
            computed: computed columns (as :py:class:`dict` keys) and their associated expressions (as  :py:class:`dict` values)
        """
        super().__init__(base, columns=None, selection=selection)
        assert self._base
        if computed is not None:
            self._base.computed.update(computed)
        if base.columns:
            columns_ = columns if columns else base.columns
            comp_ = set() if computed is None else set(computed.keys())
            allowed = set(base.columns) | comp_
            columns_ = [c for c in columns_ if c in allowed]
            assert columns_
        cols, coldict = self._base.make_projection(columns_, self)
        self._columns = cols
        self._columndict = coldict
        self._dshape = dshape_create(
            "{" + ",".join([f"{c.name}:{c.dshape}" for c in self._columns]) + "}"
        )

    @property
    def selection(self) -> PIntSet:
        return PIntSet(self._selection)

    @selection.setter
    def selection(self, sel: Union[PIntSet, slice]) -> None:
        if isinstance(sel, PIntSet):
            self._selection = sel.copy()
        elif isinstance(sel, slice):
            self._selection = sel
        else:
            raise ValueError("Selection must be a PIntSet or a slice")
