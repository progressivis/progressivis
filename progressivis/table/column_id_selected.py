from __future__ import absolute_import, division, print_function

from .column_id import IdColumn
from .column_proxy import ColumnProxy

from ..core.bitmap import bitmap
from ..core.utils import is_none_alike

class IdColumnSelectedView(ColumnProxy):
    def __init__(self, index, selection):
        assert isinstance(selection, bitmap)
        super(IdColumnSelectedView, self).__init__(base=index, name=IdColumn.INTERNAL_ID)
        self._selection = selection
        

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, selection):
        self._selection = selection

    @property
    def index(self):
        return self

    def __repr__(self):
        return 'IdColumnSelectedView(%s,dshape=%s)' % (self.name, str(self.dshape))

    def id_to_index(self, loc, as_slice=True):
        if is_none_alike(loc):
            loc = self._selection
        elif not loc in self._selection:
            raise KeyError('Invalid key(s) %s', loc)
        return self.base.id_to_index(loc) # indices in base columns for now

    def __getitem__(self, index):
        ids = self.base[index]
        if not ids in self._selection:
            raise KeyError('Invalid key(s) %s', index)
        return ids
        
    def __setitem__(self, index, value):
        raise RuntimeError('Setting id index not supported')

    def __delitem__(self, key):
        self._selection -= bitmap.asbitmap(key)

    def __len__(self):
        return len(self._selection)

    def resize(self, _):
        pass # ignore

    @property
    def update_mask(self):
        return self._selection

    @property
    def changes(self):
        return self._base.changes

    @changes.setter
    def changes(self, c):
        self._base.changes = c

    def compute_updates(self, start, mid=None):
        mask = self.update_mask
        updates = self.base.compute_updates(start, mid)
        updates.created &= mask
        updates.updated &= mask
        updates.deleted &= mask
        return updates
