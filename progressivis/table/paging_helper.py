from __future__ import annotations

from progressivis.core.utils import remove_nan
from progressivis.core.pintset import PIntSet
from progressivis.table.table_base import BasePTable

from typing import List, Any


class PagingHelper:
    def __init__(self, tbl: BasePTable):
        self._table: BasePTable = tbl
        self._index: PIntSet = tbl.index
        # FIXME
        # if not isinstance(tbl, PTable):
        #     self._index = np.array(list(self._table.index))
        if not tbl.is_identity:
            self._index = self._table.index

    def get_page(self, start: int, end: int) -> List[List[Any]]:
        ret = []
        columns = self._table.columns
        for i in self._index[start:end]:
            row = [i]
            for name in columns:
                col = self._table[name]
                row.append(remove_nan(col.loc[i]))
            ret.append(row)
        return ret
