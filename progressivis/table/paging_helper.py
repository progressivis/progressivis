from .table import Table
import numpy as np
from progressivis.core.utils import remove_nan
class PagingHelper:
    def __init__(self, tbl):
        self._table = tbl
        self._index = tbl.index
        if not isinstance(tbl, Table):
            self._index = np.array(list(self._table.index))
        elif not tbl.is_identity:
            self._index = self._table.index

    def get_page(self, start, end):
        ret = []
        columns = self._table.columns
        for i in self._index[start:end]:
            row = [i]
            for name in columns:
                col = self._table[name]
                row.append(remove_nan(col.loc[i]))
            ret.append(row)
        return ret        
