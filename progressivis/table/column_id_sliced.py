
from .column_sliced import ColumnSlicedView
from .column_id import IdColumn
from .loc import Loc

from ..core.bitmap import bitmap


class IdColumnSlicedView(ColumnSlicedView):
    def __init__(self, base, view_slice):
        super(IdColumnSlicedView, self).__init__(IdColumn.INTERNAL_ID, base,
                                                 None, view_slice)
        self._update_mask = None

    # def __delitem__(self, index):
    #     index = self.view_to_base(index)
    #     #pylint: disable=protected-access
    #     self._base._delete_ids(index)

    def __repr__(self):
        return 'IdColumnSlicedView(%s,dshape=%s)' % (
            self.name, str(self.dshape))

    def id_to_index(self, loc, as_slice=True):
        res = self._base.id_to_index(loc, as_slice)
        return self.base_to_view(res)

    def __contains__(self, loc):
        start = self._view_slice.start
        stop = self._view_slice.stop or self.base.size
        ids = self.base.index.id_to_index(loc, True)
        v = Loc.dispatch(ids)
        if v == Loc.INT:
            return start <= ids and ids < stop
        if v == Loc.SLICE:
            return start <= ids.start and ids.stop <= stop
        return ids in bitmap(self._view_slice)

    def resize(self, _):
        pass  # ignore

    @property
    def update_mask(self):
        if self._update_mask is None or \
          (self._view_slice.stop is None
           and self._update_mask.max() != self.size-1):
            self._update_mask = bitmap(self._base[self.view_slice])

        return self._update_mask


    def compute_updates(self, start, now, mid=None, cleanup=True):
        #TODO the mask should be maintained in ID space, not index space
        mask = self.update_mask
        updates = self._base.compute_updates(start, now, mid, cleanup=True)
        updates.created &= mask
        updates.updated &= mask
        updates.deleted &= mask
        return updates
