"Change manager for columns"

from progressivis.core.changemanager_base import BaseChangeManager

from .column_base import BaseColumn
from .tablechanges import TableChanges
from ..core.slot import Slot
from typing import Any


class ColumnChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a Column between runs.
    """

    def __init__(
        self,
        slot: Slot,
        buffer_created: bool = True,
        buffer_updated: bool = False,
        buffer_deleted: bool = False,
        buffer_exposed: bool = False,
        buffer_masked: bool = False,
    ) -> None:
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

    def update(self, run_number: int, data: Any, mid: str) -> None:
        assert isinstance(data, BaseColumn)
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        changes = data.compute_updates(self._last_update, run_number, mid)
        self._last_update = run_number
        self._row_changes.combine(
            changes, self.created.buffer, self.updated.buffer, self.deleted.buffer
        )


Slot.add_changemanager_type(BaseColumn, ColumnChangeManager)
