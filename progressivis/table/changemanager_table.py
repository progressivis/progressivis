"Change manager for tables."

from progressivis.core.changemanager_base import BaseChangeManager

from .table_base import BaseTable
from .tablechanges import TableChanges
from ..core.slot import Slot
from ..core.column_update import ColumnUpdate


class TableChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a Table between runs.
    """
    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=False,
                 buffer_deleted=False):
        super(TableChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted)
        self._columns = set()
        self._column_changes = set()
        data = slot.data()
        if data.changes is None:
            data.changes = TableChanges()

    def update(self, run_number, data, mid):
        if data is None or (run_number != 0
                            and run_number <= self._last_update):
            return
        assert isinstance(data, BaseTable)
        changes = data.compute_updates(self._last_update, run_number, mid)
        self._last_update = run_number
        self._row_changes.combine(changes,
                                  self.created.buffer,
                                  self.updated.buffer,
                                  self.deleted.buffer)
        columns = set(data.columns)
        if self._columns != columns:
            self._column_changes = columns
            self._columns = columns
        else:
            self._column_changes = set()

    @property
    def column_changes(self):
        return ColumnUpdate(created=self._column_changes,
                            updated=set(), deleted=set())


Slot.add_changemanager_type(BaseTable, TableChangeManager)
