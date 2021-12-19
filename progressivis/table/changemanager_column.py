"Change manager for columns"

from progressivis.core.changemanager_base import BaseChangeManager

from .column_base import BaseColumn
from .tablechanges import TableChanges
from ..core.slot import Slot


class ColumnChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a Column between runs.
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
        super(ColumnChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        data = slot.data()
        if data.changes is None:
            data.changes = TableChanges()

    def update(self, run_number, data, mid):
        assert isinstance(data, BaseColumn)
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        changes = data.compute_updates(self._last_update, run_number, mid)
        self._last_update = run_number
        self._row_changes.combine(
            changes, self.created.buffer, self.updated.buffer, self.deleted.buffer
        )


Slot.add_changemanager_type(BaseColumn, ColumnChangeManager)
