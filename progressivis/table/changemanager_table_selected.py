"Change manager for SelectedTable"

from .table_base import TableSelectedView
from ..core.changemanager_base import BaseChangeManager, _base_accessor
from ..core.changemanager_bitmap import BitmapChangeManager
from ..core.slot import Slot
from ..core.bitmap import bitmap


class FakeSlot(object):
    # pylint: disable=too-few-public-methods
    "Fake slot to provide data to inner change manager"
    __fields__ = "data"

    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _double_buffer:
    def __init__(self, first, second):
        self._first = first
        self._second = second
        self.buffer = True

    def clear(self):
        "Clear the buffer"
        self._first.clear()
        self._second.clear()

    def make_result(self, res, as_slice):
        return res.to_slice_maybe() if as_slice else res

    def __len__(self):
        "Return the buffer length"
        return len(self._first.changes | self._second.changes)

    def length(self):
        "Return the buffer length"
        return self.__len__()

    def any(self):  # TODO: improve
        "Return True if there is anything in the buffer"
        return self._first.any() or self._second.any()

    def next(self, length=None, as_slice=True):
        if length is None:
            length = len(self._first.changes | self._second.changes)
        "Return the next items in the buffer"
        acc = bitmap()
        while length and self._first.any():
            # prevents to return second ids twice
            new_ids = self._first.next(length, as_slice=False) - self._second.changes
            length -= len(new_ids)
            acc |= new_ids
        if length and self._second.any():
            acc |= self._second.next(length, as_slice=False)
        return self.make_result(acc, as_slice)


class TableSelectedChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a TableSelectedView between runs.
    """

    def __init__(
        self,
        slot,
        buffer_created=True,
        buffer_updated=False,
        buffer_deleted=False,
        buffer_exposed=False,
        buffer_masked=False,
    ):
        data = slot.data()
        assert isinstance(data, TableSelectedView)
        bmslot = FakeSlot(data.index)  # not required formally
        tbslot = FakeSlot(data._base)
        super().__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        self._mask_cm = BitmapChangeManager(
            bmslot,
            buffer_created=buffer_exposed,
            buffer_updated=False,
            buffer_deleted=buffer_masked,
            buffer_exposed=False,
            buffer_masked=False,
        )
        from .changemanager_table import TableChangeManager

        self._table_cm = TableChangeManager(
            tbslot,
            buffer_created=buffer_created,
            buffer_updated=buffer_updated,
            buffer_deleted=buffer_deleted,
            buffer_exposed=False,
            buffer_masked=False,
        )
        self._selection_changes = self._mask_cm._row_changes

    def reset(self, mid: str):
        super().reset(mid)
        self._mask_cm.reset(mid)
        self._table_cm.reset(mid)

    def update(self, run_number, data, mid):
        assert isinstance(data, TableSelectedView)
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        table = data.base
        selection = data.index
        self._mask_cm.update(run_number, selection, mid)
        self._table_cm.update(run_number, table, mid)
        table_changes = self._table_cm.row_changes
        # Mask table changes with current selection.
        table_changes.created &= selection
        table_changes.updated &= selection
        table_changes.deleted &= selection | self._selection_changes.deleted

        self._row_changes.combine(
            table_changes,
            update_created=self.created.buffer,
            update_updated=self.updated.buffer,
            update_deleted=self.deleted.buffer,
        )
        table_changes.clear()
        self._last_update = run_number

    @property
    def created(self):
        "Return information of items created"
        return _double_buffer(first=self._created, second=self._mask_cm._created)

    @property
    def updated(self):
        "Return information of items updated"
        return self._updated

    @property
    def deleted(self):
        "Return information of items deleted"
        return _double_buffer(first=self._deleted, second=self._mask_cm._deleted)

    @property
    def selection(self):
        if self._selection is None:
            self._selection = _base_accessor(self._mask_cm)
        return self._selection

    @property
    def perm_deleted(self):
        "Return information of items deleted"
        return self._deleted

    """
    @property
    def row_changes(self):
        "Return the IndexUpdate keeping track of the row changes"
        return self._table_cm._row_changes

    @property
    def mask_changes(self):
        "Return the IndexUpdate keeping track of the mask changes"
        return self._mask_cm._row_changes
    """

    def has_buffered(self):
        """
        If the change manager has something buffered, then the module is
        ready to run immediately.
        """
        return self._table_cm.has_buffered() or self._mask_cm.has_buffered()

    def last_update(self):
        "Return the date of the last update"
        return max(self._table_cm._last_update, self._mask_cm._last_update)


Slot.add_changemanager_type(TableSelectedView, TableSelectedChangeManager)
