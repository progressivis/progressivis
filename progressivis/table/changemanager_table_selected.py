"Change manager for SelectedTable"

from .table_base import TableSelectedView
from ..core.changemanager_base import BaseChangeManager
from ..core.changemanager_bitmap import BitmapChangeManager
from ..core.slot import Slot

class FakeSlot(object):
    # pylint: disable=too-few-public-methods
    "Fake slot to provide data to inner change manager"
    __fields__ = ('data')
    def __init__(self, data):
        self._data = data
    def data(self):
        return self._data

class TableSelectedChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a TableSelectedView between runs.
    """
    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=False,
                 buffer_deleted=False,
                 buffer_exposed=False,
                 buffer_masked=False):
        data = slot.data()
        assert isinstance(data, TableSelectedView)
        bmslot = FakeSlot(data.selection) # not required formally
        tbslot = FakeSlot(data._base)
        super().__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked)
        self._maskchange = BitmapChangeManager(
            bmslot,
            buffer_created=buffer_exposed,
            buffer_updated=False,
            buffer_deleted=buffer_masked,
            buffer_exposed=False,
            buffer_masked=False)
        from .changemanager_table import TableChangeManager
        #import pdb; pdb.set_trace()
        self._tablechange = TableChangeManager(
            tbslot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed=False,
            buffer_masked=False)

    def reset(self, name=None):
        super().reset(name)
        self._maskchange.reset(name)
        self._tablechange.reset(name)

    def update(self, run_number, data, mid):
        assert isinstance(data, TableSelectedView)
        #import pdb; pdb.set_trace()
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        table = data.base
        selection = data.selection
        #super().update(run_number, selection, mid)
        self._maskchange.update(run_number, selection, mid)
        self._tablechange.update(run_number, table, mid)
        # mask
        mask_changes = self._maskchange.row_changes

        self._mask_changes.combine(mask_changes,
                                  self.created.buffer,
                                  self.updated.buffer,
                                  self.deleted.buffer)
        #mask_changes.clear()
        # table
        table_changes = self._tablechange.row_changes

        self._row_changes.combine(table_changes,
                                  self.created.buffer,
                                  self.updated.buffer,
                                  self.deleted.buffer)
        #table_changes.clear()

        #self._last_update = run_number

    @property
    def created(self):
        "Return information of items created"
        return self._tablechange._created

    @property
    def updated(self):
        "Return information of items updated"
        return self._tablechange._updated

    @property
    def deleted(self):
        "Return information of items deleted"
        return self._tablechange._deleted

    @property
    def exposed(self):
        "Return information of items exposed"
        return self._maskchange._created

    @property
    def masked(self):
        "Return information of items masked"
        return self._maskchange._deleted

    @property
    def row_changes(self):
        "Return the IndexUpdate keeping track of the row changes"
        return self._tablechange._row_changes

    @property
    def mask_changes(self):
        "Return the IndexUpdate keeping track of the mask changes"
        return self._maskchange._row_changes

    def has_buffered(self):
        """
        If the change manager has something buffered, then the module is
        ready to run immediately.
        """
        return (self._tablechange.has_buffered() or self._maskchange.has_buffered())

    def last_update(self):
        "Return the date of the last update"
        return max(self._tablechange._last_update, self._maskchange._last_update)

Slot.add_changemanager_type(TableSelectedView, TableSelectedChangeManager)
