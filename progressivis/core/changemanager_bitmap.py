"Change Manager for bitmap"
from __future__ import annotations

from .bitmap import bitmap
from .index_update import IndexUpdate
from .changemanager_base import BaseChangeManager
from .slot import Slot

from typing import Any, Optional


class BitmapChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a Bitmap between runs.
    """

    def __init__(
        self,
        slot: Slot,
        buffer_created: bool = True,
        buffer_updated: bool = False,
        buffer_deleted: bool = True,
        buffer_exposed: bool = False,
        buffer_masked: bool = False,
    ):
        super(BitmapChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        self._last_bm: Optional[bitmap] = None

    def reset(self, mid: str) -> None:
        super(BitmapChangeManager, self).reset(mid)
        self._last_bm = None

    def compute_updates(self, data: bitmap) -> IndexUpdate:
        last_bm = self._last_bm
        changes = IndexUpdate()
        if last_bm is None:
            if self.created.buffer:
                changes.created.update(data)
        else:
            if self.created.buffer:
                changes.created.update(data - last_bm)
            if self.deleted.buffer:
                changes.deleted.update(last_bm - data)
        self._last_bm = bitmap(data)
        return changes

    def update(self, run_number: int, data: Any, mid: str) -> None:
        # pylint: disable=unused-argument
        assert isinstance(data, bitmap)
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return

        changes = self.compute_updates(data)
        self._last_update = run_number
        self._row_changes.combine(
            changes, self.created.buffer, self.updated.buffer, self.deleted.buffer
        )


Slot.add_changemanager_type(bitmap, BitmapChangeManager)
