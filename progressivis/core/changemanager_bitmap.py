from __future__ import absolute_import, division, print_function

from .bitmap import bitmap
from .index_update import IndexUpdate
from .changemanager_base import BaseChangeManager
from .slot import Slot


class BitmapChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a DataFrame between runs.
    """
    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=False,
                 buffer_deleted=True):
        super(BitmapChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted)
        self._last_bm = None

    def reset(self, name=None):
        super(BitmapChangeManager, self).reset(name)
        self._last_bm = None

    def compute_updates(self, data):
        last_bm = self._last_bm
        changes = IndexUpdate()
        if last_bm is None:
            if self.created.buffer:
                changes.created.update(data)
        else:
            if self.created.buffer:
                changes.created.update(data-last_bm)
            if self.deleted.buffer:
                changes.deleted.update(last_bm-data)
        self._last_bm = bitmap(data)
        return changes

    def update(self, run_number, data, mid):
        # pylint: disable=unused-argument
        assert isinstance(data, bitmap)
        if data is None or (run_number != 0 and
                            run_number <= self._last_update):
            return

        changes = self.compute_updates(data)
        self._last_update = run_number
        self._row_changes.combine(changes,
                                  self.created.buffer,
                                  self.updated.buffer,
                                  self.deleted.buffer)


Slot.add_changemanager_type(bitmap, BitmapChangeManager)
