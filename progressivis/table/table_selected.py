from __future__ import absolute_import, division, print_function

from progressivis.core.utils import (integer_types)

from .table_base import BaseTable
from .dshape import dshape_create
from ..core.bitmap import bitmap
from .column_selected import ColumnSelectedView
from .column_id_selected import IdColumnSelectedView

from collections import Iterable
import six

class InvalidOperationException(Exception): pass

class TableSelectedView(BaseTable):
    def __init__(self, base, row_selection, col_key=None, name=None):
        # should test if row_selection in indices
        assert isinstance(row_selection, bitmap)
        assert row_selection in base.index
        super(TableSelectedView, self).__init__(base=base)
        self._name = base.name if name is None else name
        self._ids = IdColumnSelectedView(base.index, row_selection)
        if col_key is None:
            col_key = base.columns
        if isinstance(col_key, (six.string_types, integer_types)):
            col_key = [col_key]
        elif isinstance(col_key, slice):
            for name in base.columns[col_key]:
                col = base[name]
                self._create_column(col.name, col)
        elif isinstance(col_key, Iterable):
            for col_id in col_key:
                col = base[col_id]
                self._create_column(col.name, col)
        else:
            raise ValueError('getitem not implemented for key "%s"' % col_key)
        self._dshape = dshape_create('{'\
                      +','.join(["{}:{}".format(c.name,c.dshape) for c in self._columns])\
                      +'}')

    @property
    def selection(self):
        return self._ids.selection

    @selection.setter
    def selection(self, value):
        assert isinstance(value, bitmap)
        self._ids.selection = value

    @property
    def nrow(self):
        return len(self._ids)

    def _create_column(self, name, base):
        column = ColumnSelectedView(base, self._ids, name=name)
        index = len(self._columns)
        self._columndict[name] = index
        self._columns.append(column)
        return column

    def resize(self, newsize, index=None):
        raise InvalidOperationException('Cannot resize a TableSelectedView')

    def __delitem__(self, key):
        self._ids -= key

    def drop(self, index, locs=None):
        if locs is None:
            locs = self.id_to_index(index)
        locs = bitmap.asbitmap(locs)
        sel = self.index.selection
        if locs in sel:
            self.index.selection = sel-locs
        else:
            raise ValueError('Invalid indices')
        
    @property
    def name(self):
        return self._name

    @property
    def dshape(self):
        return self.base.dshape
    
