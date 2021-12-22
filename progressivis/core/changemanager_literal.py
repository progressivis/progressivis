"Change Manager for literal values (supporting ==)"
from __future__ import annotations

from .bitmap import bitmap
from .index_update import IndexUpdate
from .changemanager_base import BaseChangeManager

from typing import (
    Any,
    Optional,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .slot import Slot


class LiteralChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a literal value between runs.
    """

    VALUE = bitmap([0])

    def __init__(
        self,
        slot: Slot,
        buffer_created=True,
        buffer_updated=False,
        buffer_deleted=True,
        buffer_exposed=False,
        buffer_masked=False,
    ):
        super(LiteralChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        self._last_value: Any = None

    def reset(self, name: Optional[str] = None) -> None:
        super(LiteralChangeManager, self).reset(name)
        self._last_value = None

    def compute_updates(self, data: Any) -> IndexUpdate:
        last_value = self._last_value
        changes = IndexUpdate()
        if last_value == data:
            return changes
        if last_value is None:
            if self.created.buffer:
                changes.created.update(self.VALUE)
        elif data is None:
            if self.deleted.buffer:
                changes.deleted.update(self.VALUE)
        elif self.updated.buffer:
            changes.updated.update(self.VALUE)
        self._last_value = data
        return changes

    def update(self,
               run_number: int,
               data: Any,
               mid: str) -> None:
        # pylint: disable=unused-argument
        if run_number != 0 and run_number <= self._last_update:
            return

        changes = self.compute_updates(data)
        self._last_update = run_number
        self._row_changes.combine(
            changes, self.created.buffer, self.updated.buffer, self.deleted.buffer
        )
