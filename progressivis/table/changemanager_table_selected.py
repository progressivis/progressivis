"Change manager for SelectedTable"
from __future__ import absolute_import, division, print_function

from .table_selected import TableSelectedView
from .changemanager_table import TableChangeManager
from ..core.changemanager_bitmap import BitmapChangeManager
from ..core.slot import Slot

class FakeSlot(object):
    # pylint: disable=too-few-public-methods
    "Fake slot to provide data to inner change manager"
    __fields__ = ('scheduler', 'data')
    def __init__(self, scheduler, data):
        self.scheduler = scheduler
        self.data = data

class TableSelectedChangeManager(BitmapChangeManager):
    """
    Manage changes that occured in a TableSelectedView between runs.
    """
    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=False,
                 buffer_deleted=False):
        data = slot.data()
        assert isinstance(data, TableSelectedView)
        bmslot = FakeSlot(slot.scheduler(), data.selection) # not required formally
        super(TableSelectedChangeManager, self).__init__(
            bmslot,
            buffer_created,
            buffer_updated,
            buffer_deleted)
        self._tablechange = TableChangeManager(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted)

    def reset(self, name=None):
        super(TableSelectedChangeManager, self).reset(name)
        self._tablechange.reset(name)

    def update(self, run_number, data, mid):
        assert isinstance(data, TableSelectedView)
        #import pdb; pdb.set_trace()
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        table = data.base
        selection = data.selection
        super(TableSelectedChangeManager, self).update(run_number, selection, mid)
        self._tablechange.update(run_number, table, mid)
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


Slot.add_changemanager_type(TableSelectedView, TableSelectedChangeManager)
