from __future__ import absolute_import, division, print_function

from .table_selected import TableSelectedView
from .changemanager_table import TableChangeManager
from ..core.changemanager_bitmap import BitmapChangeManager

class FakeSlot(object):
    def __init__(self, scheduler, data):
        self._scheduler = scheduler
        self._data = data       

    def scheduler(self):
        return self._scheduler

    def data(self):
        return self._data

class TableSelectedChangeManager(BitmapChangeManager):
    """
    Manage changes that occured in a TableSelectedView between runs.
    """
    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=False,
                 buffer_deleted=False,
                 manage_columns=True):
        data = slot.data()
        assert isinstance(data, TableSelectedView)
        bmslot = FakeSlot(slot.scheduler(), data.selection) # not required formally
        super(TableSelectedChangeManager, self).__init__(
            bmslot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            manage_columns)
        self._tablechange = TableChangeManager(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            manage_columns)

    def reset(self, mid=None):
        super(TableSelectedChangeManager, self).reset(mid)
        self._tablechange.reset(mid)

    def update(self, run_number, data, mid=None, cleanup=True):
        assert isinstance(data, TableSelectedView)
        #import pdb; pdb.set_trace()
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        table = data.base
        selection = data.selection
        super(TableSelectedChangeManager, self).update(run_number, selection, mid)
        self._tablechange.update(run_number, table, mid, cleanup)
        table_changes = self._tablechange.row_changes
        # Mask table changes with current selection. 
        table_changes.created &= data.selection
        table_changes.updated &= data.selection
        table_changes.deleted &= data.selection

        self._row_changes.combine(table_changes,
                                  self.created.buffer,
                                  self.updated.buffer,
                                  self.deleted.buffer)
        table_changes.clear()

        self._last_update = run_number


from ..core.slot import Slot
Slot.add_changemanager_type(TableSelectedView, TableSelectedChangeManager)
