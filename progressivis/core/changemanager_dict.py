from .changemanager_base import BaseChangeManager
from ..utils.psdict import PsDict

# from ..table.tablechanges import TableChanges
from .slot import Slot
from .index_update import IndexUpdate
import copy


class DictChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a DataFrame between runs.
    """

    def __init__(
        self,
        slot,
        buffer_created=True,
        buffer_updated=True,
        buffer_deleted=True,
        buffer_exposed=False,
        buffer_masked=False,
    ):
        super(DictChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        self._last_dict = None
        # data = slot.data()
        # if data.changes is None:
        #     data.changes = TableChanges()

    def reset(self, name=None):
        super(DictChangeManager, self).reset(name)
        self._last_dict = None

    def update(self, run_number, data, mid):
        # pylint: disable=unused-argument
        assert isinstance(data, PsDict)
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        data.fix_indices()
        last_dict = self._last_dict
        self._last_dict = copy.copy(data)
        if last_dict is None:
            # BUG data.changes.add_created(data.ids)
            changes = IndexUpdate(created=data.created_indices({}))
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


Slot.add_changemanager_type(PsDict, DictChangeManager)
