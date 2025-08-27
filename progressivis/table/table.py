"""
Main PTable class
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import logging
import numpy as np
import pandas as pd
import numexpr as ne
import pyarrow as pa
from progressivis.core.utils import (
    integer_types,
    get_random_name,
    slice_to_pintset,
    all_int,
    are_instances,
    gen_columns,
)

# try:
#    from progressivis.utils.fast import indices_to_slice
# except ImportError:
from progressivis.core.utils import indices_to_slice

from progressivis.storage import Group
from .dshape import (
    DataShape,
    dshape_create,
    dshape_table_check,
    dshape_fields,
    dshape_to_shape,
    dshape_extract,
    dshape_compatible,
    dshape_from_dtype,
    EMPTY_DSHAPE,
)
from . import metadata
from .table_base import IndexPTable, BasePTable
from .column import PColumn

from progressivis.core.pintset import PIntSet

from typing import Any, Dict, Optional, Union, cast, Tuple, Callable, List
from ..core.pv_types import Index, Data

Chunks = Union[None, int, Dict[str, Union[int, Tuple[int, ...]]]]

logger = logging.getLogger(__name__)

__all__ = ["PTable"]


def _get_slice_df(df: Data, sl: Index) -> Data:
    return df.iloc[sl]


def _get_slice_arr(arr: Data, sl: Index) -> Data:
    return arr[sl]


class PTable(IndexPTable):
    """Create a PTable data structure, made of a collection of columns.

    A PTable is similar to Python Pandas or R DataFrame, but
    column-based and supporting fast addition of items.

    Example:
        >>> from progressivis import PTable
        >>> t = PTable('my-table', dshape='{a: int32, b: float32, c: bool}', create=True)
        >>> len(t)
        0
        >>> t.columns
        ['a', 'b', 'c']
    """

    def __init__(
        self,
        name: Optional[str],
        data: Any = None,
        dshape: Optional[Union[str, DataShape]] = None,
        fillvalues: Optional[Dict[str, Any]] = None,
        storagegroup: Optional[Group] = None,
        chunks: Optional[Chunks] = None,
        create: Optional[bool] = None,
        indices: Optional[Index] = None,
    ) -> None:
        """
        Args:
            name: the name of the table
            data: optional container: contained sata that will be appended to the table. \
                It can be of multiple types:

                * :class:`PTable <progressivis.PTable>`: another table is used to fill-up this table
                * :class:`pandas.DataFrame`: a Pandas DataFrame is copied to this table
                * :class:`numpy.ndarray`: a numpy array is copied. The dshape should be provided

            dshape: data shape such as ``{'a': int32, 'b': float64, 'c': string}``
                The column names and types as specified by the `datashape` library.
            fillvalues: the default values of the columns specified as a dictionary
                Each column is created with a default ``fillvalue``. This parameter can
                specify the fillvalue of each column with 3 formats:
                a single value) which will be used by all the column creations
                a dictionary) associating a column name to a value
                the '*' entry in a dictionary) for defaulting the fillvalue when not specified
            storagegroup: a factory used to create the columns
                When not specified or `None` the default ``storage`` is used.
                Otherwise, a ``storagegroup`` is specified in ``Group``.
            chunks: the specification of the chunking of columns when the storagegroup
                supports it.
                Like the ``fillvalue`` argument, it can be one value or a dict.
            create: forces the creation of the table.
                Because the the storagegroup allows persistence, a table with the same name \
                may exist in the storagegroup. With ``create=False``, the previous value \
                is loaded, whereas with ``create=True`` it is replaced.
            indices: the indices of the rows appended when data is specified, in case the
                table contents has to be joined with another table.

        """
        # pylint: disable=too-many-arguments, too-many-branches
        super().__init__()
        if not (fillvalues is None or isinstance(fillvalues, Mapping)):
            raise ValueError(
                "Invalid fillvalues (%s) should be None or a dictionary" % fillvalues
            )
        if not (chunks is None or isinstance(chunks, (integer_types, Mapping))):
            raise ValueError(
                "Invalid chunks (%s) should be None or a dictionary" % chunks
            )
        if data is not None:
            if create is not None and create is not True:
                logger.warning("creating a PTable with data and create=False")
            create = True

        self._chunks = chunks
        # self._nrow = 0
        self._name: str = get_random_name("table_") if name is None else name
        # TODO: attach all randomly named tables to a dedicated, common parent node
        if not (storagegroup is None or isinstance(storagegroup, Group)):
            raise ValueError(
                "Invalid storagegroup (%s) should be None or a Group" % storagegroup
            )
        if storagegroup is None:
            assert Group.default
            storagegroup = Group.default(self._name, create=create)
        if storagegroup is None:
            raise RuntimeError("Cannot get a valid default storage Group")
        self._storagegroup = storagegroup
        if dshape is None:
            if data is None:
                self._dshape = EMPTY_DSHAPE
            else:
                data = self.parse_data(data)
                self._dshape = dshape_extract(data) or EMPTY_DSHAPE
        else:
            self._dshape = dshape_create(dshape)
            assert dshape_table_check(self._dshape)
        if create and self._dshape is EMPTY_DSHAPE:
            raise ValueError("Cannot create a table without a dshape")
        if self._dshape is EMPTY_DSHAPE or (
            not create and metadata.ATTR_TABLE in self._storagegroup.attrs
        ):
            self._load_table()
        else:
            self._create_table(fillvalues or {})
        if data is not None:
            self.append(data, indices=indices)

    @property
    def name(self) -> str:
        return self._name

    def _chunks_for(self, name: str) -> Union[None, int, Tuple[int, ...]]:
        chunks = self._chunks
        if chunks is None:
            return None
        if isinstance(chunks, (integer_types, tuple)):
            return chunks
        if name in chunks:
            return chunks[name]
        if "*" in chunks:
            return chunks["*"]
        return None

    @property
    def storagegroup(self) -> Group:
        "Return the storagegroup form this column"
        return self._storagegroup

    def _load_table(self) -> None:
        node = self._storagegroup
        if metadata.ATTR_TABLE not in node.attrs:
            raise ValueError('Group "%s" is not a PTable', self.name)
        version = node.attrs[metadata.ATTR_VERSION]
        if version != metadata.VALUE_VERSION:
            raise ValueError('Invalid version "%s" for PTable', version)
        nrow = node.attrs[metadata.ATTR_NROWS]
        self._dshape = dshape_create(node.attrs[metadata.ATTR_DATASHAPE])
        assert dshape_table_check(self._dshape)
        self._index = PIntSet.deserialize(self._storagegroup.attrs[metadata.ATTR_INDEX])
        self._last_id = node.attrs[metadata.ATTR_LAST_ID]
        for (name, dshape) in dshape_fields(self._dshape):
            column = self._create_column(name)
            column.load_dataset(
                dshape=dshape_create(dshape), nrow=nrow, shape=dshape_to_shape(dshape)
            )

    def _create_table(self, fillvalues: Any) -> None:
        node = self.storagegroup
        node.attrs[metadata.ATTR_TABLE] = self.name
        node.attrs[metadata.ATTR_VERSION] = metadata.VALUE_VERSION
        node.attrs[metadata.ATTR_DATASHAPE] = str(self._dshape)
        node.attrs[metadata.ATTR_NROWS] = 0
        node.attrs[metadata.ATTR_INDEX] = self._index.serialize()
        node.attrs[metadata.ATTR_LAST_ID] = self.last_id
        # create internal id dataset
        # self._ids = IdPColumn(table=self, storagegroup=self.storagegroup)
        # self._ids.create_dataset(dshape=None, fillvalue=-1)
        for (name, dshape) in dshape_fields(self._dshape):
            assert name not in self._columndict
            shape = dshape_to_shape(dshape)
            fillvalue = fillvalues.get(name, None)
            chunks = self._chunks_for(name)
            # TODO compute chunks according to the shape
            column = self._create_column(name)
            column.create_dataset(
                dshape=dshape_create(dshape),
                chunks=chunks,
                fillvalue=fillvalue,
                shape=shape,
            )

    def _create_column(self, name: str) -> PColumn:
        column = PColumn(name, self, storagegroup=self.storagegroup)
        index = len(self._columns)
        self._columndict[name] = index
        self._columns.append(column)
        return column

    def __contains__(self, colname: str) -> bool:
        return colname in self._columndict

    def _drop(
        self, index: Any, raw_index: Optional[Any] = None, truncate: bool = False
    ) -> None:
        super()._drop(index, raw_index, truncate)
        self._storagegroup.attrs[metadata.ATTR_INDEX] = self._index.serialize()
        self._storagegroup.attrs[metadata.ATTR_LAST_ID] = self.last_id

    def truncate(self) -> None:
        if len(self):
            self.drop(slice(None, None, None), truncate=True)

    def _resize_rows(self, newsize: int, index: Optional[Any] = None) -> None:
        super()._resize_rows(newsize, index)
        self._storagegroup.attrs[metadata.ATTR_INDEX] = self._index.serialize()
        self._storagegroup.attrs[metadata.ATTR_LAST_ID] = self.last_id

    def resize(
        self, newsize: int, index: Optional[Union[PIntSet, List[int]]] = None
    ) -> None:
        # NB: newsize means how many active rows the table must contain
        if index is not None:
            index = PIntSet.aspintset(index)
            newsize_ = index.max() + 1 if index else 0
            if newsize < newsize_:
                logger.warning(f"Wrong newsize={newsize}, fixed to {newsize_}")
                newsize = newsize_
        assert newsize is not None
        delta = newsize - len(self.index)
        # if delta < 0:
        #    return
        newsize = self.last_id + delta + 1
        crt_index = PIntSet(self._index)
        self._resize_rows(newsize, index)
        del_index = crt_index - self._index
        if del_index:
            self.add_deleted(del_index)
        if delta < 0:
            return
        self._storagegroup.attrs[metadata.ATTR_NROWS] = newsize
        assert newsize is not None
        for column in self._columns:
            col = cast(PColumn, column)
            col._resize(newsize)

    def _allocate(
        self, count: int, index: Optional[Union[PIntSet, List[int]]] = None
    ) -> PIntSet:
        start = self.last_id + 1
        index = (
            PIntSet(range(start, start + count))
            if index is None
            else PIntSet.aspintset(index)
        )
        newsize = max(index.max(), self.last_id) + 1
        self.add_created(index)
        self._storagegroup.attrs[metadata.ATTR_NROWS] = newsize
        for column in self._columns:
            col = cast(PColumn, column)
            col._resize(newsize)
        self._resize_rows(newsize, index)
        return index

    def touch_rows(self, loc: Any = None) -> None:
        "Signals that the values at loc have been changed"
        self.touch(loc)

    def parse_data(self, data: Any, indices: Optional[Any] = None) -> Any:
        if data is None:
            return None
        if isinstance(data, Mapping):
            if are_instances(data.values(), (np.ndarray, list)):
                return data  # PTable can parse this
        if isinstance(data, (np.ndarray, Mapping)):
            # delegate creation of structured data to pandas for now
            data = pd.DataFrame(data, columns=self.columns, index=indices)
        return data  # hoping it works

    def append(self, data: Any, indices: Optional[Any] = None) -> None:
        """
        Append rows of the tabular ``data`` (i.e. :class:`PTable <progressivis.PTable>`,
        :class:`pandas.DataFrame`, :class:`pyarrow.RecordBatch` or :py:class:`dict` of \
        arrays) to the  end of ``self``.
        The data has to be compatible. It can be from multiple sources
        [more details needed].

        Args:
            data: data to be appended
            indices: allows to force indices for the appended rows
        """
        if data is None:
            return
        if data is self:
            data = data.to_dict(orient="list")
        data = self.parse_data(data, indices)
        """
        The following _get_slice() definition is motivated by the warning below and
        because data is not always a DataFrame:
        FutureWarning: The behavior of `series[i:j]` with an integer-dtype index is deprecated.
        In a future version, this will be treated as *label-based* indexing,
        consistent with e.g. `series[i]` lookups. To retain the old behavior,
        use `series.iloc[i:j]`. To get the future behavior, use `series.loc[i:j]`
        """
        _get_slice: Callable[[Data, Index], Data]
        if isinstance(data, pd.DataFrame):
            _get_slice = _get_slice_df
        else:
            _get_slice = _get_slice_arr
        dshape = dshape_extract(data)
        if not dshape_compatible(dshape, self.dshape):
            raise ValueError(f"{dshape} incompatible data shape in append")
        length = -1
        all_arrays = True

        def _len(c: Any) -> int:
            if isinstance(data, BasePTable):
                return len(c.value)
            return len(c)

        for colname in self:
            fromcol = data[colname]
            if length == -1:
                length = _len(fromcol)
            elif length != _len(fromcol):
                raise ValueError("Cannot append ragged values")
            all_arrays |= isinstance(fromcol, np.ndarray)
        if length == 0:
            return
        if isinstance(indices, slice):
            indices = slice_to_pintset(indices)
        if indices is not None and len(indices) != length:
            raise ValueError("Bad index length (%d/%d)", len(indices), length)
        init_indices = indices
        prev_last_id = self.last_id
        indices = self._allocate(length, indices)
        if isinstance(data, BasePTable):
            left_ind: Union[PIntSet, slice]
            if init_indices is None:
                start = prev_last_id + 1
                left_ind = slice(start, start + len(data) - 1)
            else:
                left_ind = indices
            self.loc[left_ind, :] = data
        elif all_arrays:
            from_ind = slice(0, length)
            raw_indices = indices  # still PIntSet here
            # indices = indices_to_slice(indices)
            indices = indices.to_slice_maybe()
            for colname in self:
                tocol = self._column(colname)
                fromcol = data[colname]
                # fromcol_ind = fromcol[from_ind]
                fromcol_ind = _get_slice(fromcol, from_ind)
                try:
                    tocol[indices] = fromcol_ind
                except ValueError:
                    if isinstance(fromcol, pa.TimestampArray):
                        for i, (k, elt) in enumerate(zip(raw_indices, fromcol_ind)):
                            dt = elt.as_py()
                            tocol[k] = (
                                dt.year,
                                dt.month,
                                dt.day,
                                dt.hour,
                                dt.minute,
                                dt.second,
                            )
                    elif isinstance(
                        fromcol, pd.Series
                    ) and fromcol.dtype.name.startswith("datetime"):
                        for i, (k, dt) in enumerate(zip(raw_indices, fromcol_ind)):
                            tocol[k] = (
                                dt.year,
                                dt.month,
                                dt.day,
                                dt.hour,
                                dt.minute,
                                dt.second,
                            )

                    else:
                        raise
        else:
            for colname in self:
                tocol = self._column(colname)
                fromcol = data[colname]
                for i in range(length):
                    tocol[indices[i]] = fromcol[i]

    def add(self, row: Any, index: Optional[Any] = None) -> int:
        """
        Add one row to the end of ``self``.

        Args:
            row: the row to be added (typically a :py:class:`dict` or a sequence)
            index: allows to force a the index value of the added row
        """
        assert len(row) == self.ncol

        if isinstance(row, Mapping):
            for colname in self:
                if colname not in row:
                    raise ValueError(f'Missing value for column "{colname}"')
        if index is None:
            index = self._allocate(1)
        else:
            index = self._allocate(1, [index])
        # start = self.id_to_index(index[0], as_slice=False)
        start = index[0]

        if isinstance(row, Mapping):
            for colname in self:
                tocol = self._column(colname)
                tocol[start] = row[colname]
        else:
            for colname, value in zip(self, row):
                tocol = self._column(colname)
                tocol[start] = value
        return start

    def binary(
        self,
        op: Callable[
            [np.ndarray[Any, Any], Union[np.ndarray[Any, Any], int, float, bool]],
            np.ndarray[Any, Any],
        ],
        other: BasePTable,
        **kwargs: Any,
    ) -> Union[Dict[str, np.ndarray[Any, Any]], BasePTable]:
        res = super().binary(op, other, **kwargs)
        if isinstance(res, BasePTable):
            return res
        return PTable(None, data=res, create=True)

    @staticmethod
    def from_array(
        array: np.ndarray[Any, Any],
        name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        offsets: Optional[Union[List[int], List[Tuple[int, int]]]] = None,
        dshape: Optional[Union[str, DataShape]] = None,
        **kwds: Any,
    ) -> PTable:
        """Offsets is a list of indices or pairs."""
        if offsets is None:
            offsets = [(i, i + 1) for i in range(array.shape[1])]
        elif offsets is not None:
            if all_int(offsets):
                ioffsets = cast(List[int], offsets)
                offsets = [
                    (ioffsets[i], ioffsets[i + 1]) for i in range(len(ioffsets) - 1)
                ]
            elif not all([isinstance(elt, tuple) for elt in offsets]):
                raise ValueError("Badly formed offset list %s", offsets)

        toffsets = cast(List[Tuple[int, int]], offsets)
        if columns is None:
            if dshape is None:
                columns = gen_columns(len(toffsets))
            else:
                dshape = dshape_create(dshape)
                columns = [fieldname for (fieldname, _) in dshape_fields(dshape)]
        if dshape is None:
            dshape_type = dshape_from_dtype(array.dtype)
            dims = [
                "" if (off[0] + 1 == off[1]) else "%d *" % (off[1] - off[0])
                for off in toffsets
            ]
            dshapes = [
                "%s: %s %s" % (column, dim, dshape_type)
                for column, dim in zip(columns, dims)
            ]
            dshape = "{" + ", ".join(dshapes) + "}"
        data = OrderedDict()
        for nam, off in zip(columns, toffsets):
            if off[0] + 1 == off[1]:
                data[nam] = array[:, off[0]]
            else:
                data[nam] = array[:, off[0]: off[1]]
        return PTable(name, data=data, dshape=str(dshape), **kwds)

    def eval(
        self,
        expr: str,
        inplace: bool = False,
        name: Optional[str] = None,
        result_object: Optional[str] = None,
        locs: Optional[Any] = None,
        as_slice: bool = True,
    ) -> Any:
        """Evaluate the ``expr`` on columns and return the result.

        Args:
            inplace: boolean, optional
                Apply the changes in place
            name: string
                used when a new table/view is created, otherwise ignored
            result_object: string
               Posible values for result_object: {'raw_numexpr', 'index', 'view', 'table'}
               When expr is conditional.
               Note: a result as 'view' is not guaranteed: it may be 'table' when the
               calculated index is not sliceable
               - 'table' or None when expr is an assignment
               Default values for result_object :
               - 'indices' when expr is conditional
               - NA i.e. always None when inplace=True, otherwise a new table is returned
        """
        if inplace and result_object:
            raise ValueError("incompatible options 'inplace' and 'result_object'")
        indices = locs
        if indices is None:
            indices = np.array(self.index)
        context = {
            k: self.to_array(locs=indices, columns=[k]).reshape(-1)
            for k in self.columns
        }
        is_assign = False
        try:
            res = ne.evaluate(expr, local_dict=context)
            if result_object is None:
                result_object = "index"
        except SyntaxError as err:
            # maybe an assignment ?
            try:
                l_col, r_expr = expr.split("=", 1)
                l_col = l_col.strip()
                if l_col not in self.columns:
                    raise err
                res = ne.evaluate(r_expr.strip(), local_dict=context)
                is_assign = True
            except Exception:
                raise err
            if result_object is not None and result_object != "table":
                raise ValueError(
                    "result_object={} is not valid when expr "
                    "is an assignment".format(result_object)
                )
        else:
            if result_object not in ("raw_numexpr", "index", "view", "table"):
                raise ValueError("result_object={} is not valid".format(result_object))
        if is_assign:
            if inplace:
                self[l_col] = res
                return

            def cval(key: Any) -> Any:
                return res if key == l_col else self[key].values

            data = [(cname, cval(cname)) for cname in self.columns]
            return PTable(name=name, data=OrderedDict(data), indices=self.index)
        # not an assign ...

        if res.dtype != "bool":
            raise ValueError("expr must be an assignment " "or a conditional expr.!")
        if inplace:
            raise ValueError("inplace eval of conditional expr " "not implemented!")
        if result_object == "raw_numexpr":
            return res
        indices = indices[res]
        if not as_slice and result_object == "index":
            return indices
        ix_slice = indices_to_slice(indices)
        if result_object == "index":
            return ix_slice
        if result_object == "view":
            return self.loc[ix_slice, :]
        # as a new table ...
        data = [(cname, self[cname].values[indices]) for cname in self.columns]
        return PTable(name=name, data=OrderedDict(data), indices=indices)
