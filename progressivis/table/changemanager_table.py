"Change manager for tables."
from __future__ import annotations

from collections import namedtuple

from progressivis.core.changemanager_base import BaseChangeManager
from .table import PTable
from .tablechanges import PTableChanges
from ..core.slot import Slot

from typing import Any, Set


PColumnUpdate = namedtuple("PColumnUpdate", ["created", "updated", "deleted"])


class PTableChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a PTable between runs.
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
        super().__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        self._slot = slot
        self._columns: Set[str] = set()
        self._column_changes: Set[str] = set()
        data = slot.data()
        if data.changes is None:
            data.changes = PTableChanges()

    def reset(self, mid: str) -> None:
        super().reset(mid)
        data = self._slot.data()
        data.reset_updates(mid)

    def update(self, run_number: int, data: Any, mid: str) -> None:
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        assert isinstance(data, PTable)
        changes = data.compute_updates(self._last_update, run_number, mid)
        self._last_update = run_number
        self._row_changes.combine(
            changes, self.created.buffer, self.updated.buffer, self.deleted.buffer
        )
        columns = set(data.columns)
        if self._columns != columns:
            self._column_changes = columns
            self._columns = columns
        else:
            self._column_changes = set()

    @property
    def column_changes(self) -> PColumnUpdate:
        return PColumnUpdate(created=self._column_changes, updated=set(), deleted=set())


Slot.add_changemanager_type(PTable, PTableChangeManager)
