
from .changemanager_base import BaseChangeManager
from ..utils.psdict import PsDict
from ..table.tablechanges import TableChanges
from .slot import Slot
import copy


class DictChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a DataFrame between runs.
    """
    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=True,
                 buffer_deleted=True):
        super(DictChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted)
        self._last_dict = None
        data = slot.data()
        if data.changes is None:
            data.changes = TableChanges()

    def reset(self, name=None):
        super(DictChangeManager, self).reset(name)
        self._last_dict = None

    def update(self, run_number, data, mid):
        # pylint: disable=unused-argument
        assert isinstance(data, PsDict)
        if data is None or (run_number != 0 and
                            run_number <= self._last_update):
            return
        data.fix_indices()
        last_dict = self._last_dict
        if last_dict is None:
            data.changes.add_created(data.ids)
        else:
            data.changes.add_created(data.new_indices(last_dict))
            data.changes.add_updated(data.updated_indices(last_dict))
            data.changes.add_deleted(data.deleted_indices(last_dict))
        changes = data.compute_updates(self._last_update, run_number, mid)
        self._last_dict = copy.copy(data)
        self._last_update = run_number
        self._row_changes.combine(changes,
                                  self.created.buffer,
                                  self.updated.buffer,
                                  self.deleted.buffer)


Slot.add_changemanager_type(PsDict, DictChangeManager)
