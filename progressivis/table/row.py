from __future__ import annotations

from collections import MutableMapping
from progressivis.core.utils import integer_types, remove_nan_etc
from collections import OrderedDict


from typing import (
    Any, List, Tuple, Union, Dict, TYPE_CHECKING, overload, Sequence,
    Optional,
    Iterator
)

JSon = Dict[str, Any]

if TYPE_CHECKING:
    from .table_base import BaseTable
    from .dshape import DataShape
    import numpy as np


class Row(MutableMapping[str, Any]):
    """ Wraps a dictionary interace around a row of a Table.

    Parameters
    ----------
    table : Table to wrap
    index : the integer index of the row to wrap, or None if it has to remain the last
        row of the table

    Examples
    --------
    >>> from progressivis.table import Table, Row
    >>> table = Table('table', data={'a': [ 1, 2, 3], 'b': [10.1, 0.2, 0.3]})
    >>> row = Row(table) # wraps the last row
    >>> len(row)
    2
    >>> row['a']
    3
    """

    def __init__(self, table: BaseTable, index: Optional[int] = None) -> None:
        super(Row, self).__setattr__("table", table)
        if index is not None and not isinstance(index, integer_types):
            raise ValueError('index should be an integer, not "%s"' % str(index))
        super(Row, self).__setattr__("index", index)
        self.index: Optional[int]
        self.table: BaseTable

    @property
    def row(self) -> Optional[int]:
        table = self.table
        index = self.index
        return table.last_xid if index is None else index

    def __len__(self) -> int:
        table = self.table
        return table.ncol

    @overload
    def __getitem__(self, key: Union[int, str]) -> Any:
        ...

    @overload
    def __getitem__(self, key: Sequence[Union[int, str]]) -> Tuple[Any, ...]:
        ...

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (list, tuple)):
            return (self[k] for k in key)  # recursive call
        table = self.table
        return table.at[self.row, table.column_index(key)]

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        table = self.table
        table.at[self.row, table.column_index(key)] = value

    def __delitem__(self, key: Union[str, int]) -> None:
        raise ValueError("Cannot delete from a row")

    def dtype(self, key: Union[int, str]) -> np.dtype[Any]:
        table = self.table
        return table[key].dtype

    def dshape(self, key: Union[int, str]) -> DataShape:
        table = self.table
        return table[key].dshape

    def __iter__(self) -> Iterator[str]:
        table = self.table
        return iter(table)

    # def __reversed__(self) -> Iterator[str]:
    #     table = self.table
    #     return iter(table).reversed()  # type: ignore

    def __contains__(self, key: object) -> bool:
        table = self.table
        if isinstance(key, (int, str)):
            return key in table
        return False

    def to_dict(self, ordered: bool = False) -> Dict[str, Any]:
        if ordered:
            return OrderedDict(self)
        return dict(self)

    def to_json(self) -> JSon:
        d = dict(self)
        remove_nan_etc(d)
        return d

    def __getattr__(self, attr: str) -> Any:
        return self[attr]

    def __setattr__(self, attr: str, value: Any) -> None:
        self[attr] = value

    def __dir__(self) -> List[str]:
        table = self.table
        return list(table.columns)
