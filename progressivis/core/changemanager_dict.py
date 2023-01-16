from __future__ import annotations

from .changemanager_base import BaseChangeManager
from ..utils.psdict import PDict, EMPTY_PSDICT

# from ..table.tablechanges import PTableChanges
from .slot import Slot
from .index_update import IndexUpdate
import copy

from typing import Any, Optional


class DictChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a DataFrame between runs.
    """

    def __init__(
        self,
        slot: Slot,
        buffer_created: bool = True,
        buffer_updated: bool = True,
        buffer_deleted: bool = True,
        buffer_exposed: bool = False,
        buffer_masked: bool = False,
    ) -> None:
        super(DictChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        self._last_dict: Optional[PDict] = None
        # data = slot.data()
        # if data.changes is None:
        #     data.changes = PTableChanges()

    def reset(self, mid: str) -> None:
        super(DictChangeManager, self).reset(mid)
        self._last_dict = None

    def update(self, run_number: int, data: Any, mid: str) -> None:
        # pylint: disable=unused-argument
        assert isinstance(data, PDict)
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        data.fix_indices()
        last_dict = self._last_dict
        self._last_dict = copy.copy(data)
        if last_dict is None:
            # BUG data.changes.add_created(data.ids)
            changes = IndexUpdate(created=data.created_indices(EMPTY_PSDICT))
        else:
            changes = IndexUpdate(
                created=data.created_indices(last_dict),
                updated=data.updated_indices(last_dict),
                deleted=data.deleted_indices(last_dict),
            )
        # changes = data.compute_updates(self._last_update, run_number, mid)
        self._last_update = run_number
        self._row_changes.combine(
            changes, self.created.buffer, self.updated.buffer, self.deleted.buffer
        )


Slot.add_changemanager_type(PDict, DictChangeManager)
