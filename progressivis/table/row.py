from __future__ import annotations

from collections import Mapping
from progressivis.core.utils import integer_types, remove_nan_etc
from collections import OrderedDict


from typing import Any, List, Tuple, Union, Dict
JSon = Dict[str, Any]


class Row(Mapping):
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

    def __init__(self, table, index: int = None):
        super(Row, self).__setattr__("table", table)
        if index is not None and not isinstance(index, integer_types):
            raise ValueError('index should be an integer, not "%s"' % str(index))
        super(Row, self).__setattr__("index", index)

    @property
    def row(self) -> int:
        table = self.table
        index = self.index
        return table.last_xid if index is None else index

    def __len__(self) -> int:
        table = self.table
        return table.ncol

    def __getitem__(self, key: Union[int, str, List[Union[int, str]], Tuple[Union[int, str], ...]]) -> Any:
        table = self.table
        if isinstance(key, (list, tuple)):
            return (self[k] for k in key)  # recursive call
        return table.at[self.row, table.column_index(key)]

    def __setitem__(self, key: Union[str, int], value: Any):
        table = self.table
        table.at[self.row, table.column_index(key)] = value

    def dtype(self, key: Union[int, str]):
        table = self.table
        return table[key].dtype

    def dshape(self, key: Union[int, str]):
        table = self.table
        return table[key].dshape

    def __iter__(self):
        table = self.table
        return iter(table)

    def __reversed__(self):
        table = self.table
        return table.iter().reversed()

    def __contains__(self, key: object):
        table = self.table
        if isinstance(key, (int, str)):
            return key in table
        return False

    def to_dict(self, ordered=False):
        if ordered:
            return OrderedDict(self)
        return dict(self)

    def to_json(self) -> JSon:
        return remove_nan_etc(dict(self))

    def __getattr__(self, attr: str) -> Any:
        return self[attr]

    def __setattr__(self, attr: str, value: Any) -> None:
        self[attr] = value

    def __dir__(self):
        table = self.table
        return list(table.columns)
