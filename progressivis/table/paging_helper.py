from __future__ import annotations

from progressivis.core.utils import remove_nan
from progressivis.core.bitmap import bitmap
from progressivis.table.table_base import BaseTable

from typing import List, Any


class PagingHelper:
    def __init__(self, tbl: BaseTable):
        self._table: BaseTable = tbl
        self._index: bitmap = tbl.index
        # FIXME
        # if not isinstance(tbl, Table):
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
