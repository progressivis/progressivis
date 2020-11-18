from .column_proxy import ColumnProxy

import logging
logger = logging.getLogger(__name__)


class ColumnSelectedView(ColumnProxy):
    def __init__(self, base, index, name=None):
        super(ColumnSelectedView, self).__init__(base, index=index, name=name)

    @property
    def shape(self):
        tshape = list(self._base.shape)
        tshape[0] = len(self)
        return tuple(tshape)

    def set_shape(self, shape):
        raise RuntimeError("set_shape not implemented for %s", type(self))

    @property
    def maxshape(self):
        tshape = list(self._base.maxshape)
        tshape[0] = len(self)
        return tuple(tshape)

    def __len__(self):
        return len(self.index)

    @property
    def value(self):
        return self._base[self.index.id_to_index(slice(None, None, None))]


    def __getitem__(self, index):
        #import pdb;pdb.set_trace()
        bm = self.index._any_to_bitmap(index)
        return self._base[bm]
