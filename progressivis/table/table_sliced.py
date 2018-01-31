from __future__ import absolute_import, division, print_function

from progressivis.core.utils import (integer_types, norm_slice)
from .dshape import (dshape_create)

from .table_base import BaseTable
from .column_sliced import ColumnSlicedView
from .column_id_sliced import IdColumnSlicedView

from collections import Iterable
import six

import logging
logger = logging.getLogger(__name__)

        
# TODO return a name acceptable for a dataset
def quote_type_name(name):
    return name

class TableSlicedView(BaseTable):
    def __init__(self, table, row_key, col_key):
        assert isinstance(row_key, slice) and row_key.step in (None, 1)
        #TODO climb up to the root view instead of maintaining a chain of views
        super(TableSlicedView, self).__init__(base=table)
        self._name = table.name
        # self.get_subtable_name(table.name) # "tv_"+table.name+'_'+str(uuid.uuid4()).split('-')[-1] # quite ugly...
        #The base table should handle it fine
        #self._ids  = IdColumnSlicedView(update=None, target_column=table.index, view_slice=row_key)
        self._view_slice = norm_slice(row_key)
        self._ids = IdColumnSlicedView(table.index, self._view_slice)
        if isinstance(col_key, (six.string_types, integer_types)):
            col_key = [col_key]
        if isinstance(col_key, slice):
            for name in table.columns[col_key]:
                col = table._column(name)
                self._create_column(name, self._view_slice, col)
        elif isinstance(col_key, Iterable):
            for col_id in col_key:
                col = table._column(col_id)
                self._create_column(col.name, self._view_slice, col)
        else:
            raise ValueError('getitem not implemented for key "%s"' % col_key)
        self._dshape = dshape_create('{'\
                      +','.join(["{}:{}".format(c.name,c.dshape) for c in self._columns])\
                      +'}')

    @property
    def view_slice(self):
        view_slice = self._view_slice
        if view_slice.stop is None:
            view_slice = slice(view_slice.start, self.nrow, 1)
        return view_slice

    @property
    def nrow(self):
        view_slice = self._view_slice
        if view_slice.stop is None:
            return self._base.nrow - view_slice.start
        return view_slice.stop - view_slice.start

    @property
    def name(self):
        return self._name

    def _create_column(self, name, row_key, base):
        column = ColumnSlicedView(name, base=base, index=self._ids, view_slice=row_key)
        index = len(self._columns)
        #column.dataset = base.dataset[row_key]
        self._columndict[name] = index
        self._columns.append(column)
        return column

    ## def id_to_index(self, rid, as_slice=True): # to be reimplemented with LRU-dict+pyroaring
    ##     res = super(TableView, self).id_to_index(rid, as_slice)
    ##     return self._ids.get_shifted_key(res)

    def __delitem__(self, key):
        raise NotImplementedError('TODO')

    def drop(self, key):
        raise NotImplementedError('TODO')

    def __getitem__(self, key):
        if isinstance(key, (six.string_types, integer_types)):
            return self._column(key)
        elif isinstance(key, Iterable):
            return (self._column(c) for c in key)
        raise ValueError('getitem not implemented for key "%s"' % key)

    def resize(self, newsize, index=None):
        pass

    def __setitem__(self, key, values):
        raise NotImplementedError('Sorry, __setitem__ in a view is not implemented yet')


    @property
    def dshape(self):
        return self._dshape
    ## @staticmethod
    ## def from_pytable(pytable):
    ##     if not isinstance(pytable, pt.table.Table):
    ##         raise ValueError('{} is not a Pytables instance'.format(pytable))
    ##     name = pytable.name)
