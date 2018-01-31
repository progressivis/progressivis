from __future__ import absolute_import, division, print_function

from ..core.index_update import IndexUpdate
from ..core.tablechanges import BaseChanges
from ..core.bitmap import bitmap

import bisect

import logging
logger = logging.getLogger(__name__)

class TableChanges(BaseChanges):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._saved_time = []
        self._saved_update = []
        self._saved_mid = []
        
    def _last_update(self):
        if len(self._saved_update)==0:
            return None
        return self._saved_update[-1]

    def _saved_index(self, x):
        #print('_saved_index: %d'%x)
        a = self._saved_time
        i = bisect.bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError('Invalid time searched in _saved_index: %d'%x)

    def _save_time(self, time, mid):
        if mid is not None and mid in self._saved_mid:
            # reset
            logger.debug('Reset received for module %s', mid)
            i = self._saved_index(time)
            if self._saved_mid[i] != mid:
                raise ValueError('Invalid module id in compute_updates %s instead of %s'\
                                 %(mid, self._saved_mid[i]))
            del self._saved_time[i]
            del self._saved_mid[i]
            del self._saved_update[i]

        self._saved_time.append(time)
        self._saved_mid.append(mid)
        if len(self._saved_update)==0:
            update = IndexUpdate(created=None, deleted=None, updated=None)
        else:
            update = self._saved_update[-1]
            # if some changes have already been collected, we create a fresh IndexUpdate,
            # otherwise we share the same entry: multiple times will share the same set of changes.
            if (update.created or update.deleted or update.updated):
                update = IndexUpdate()
        self._saved_update.append(update)

    def add_created(self, locs):
        update = self._last_update()
        if update is None:
            return
        update.add_created(locs)

    def add_updated(self, locs):
        update = self._last_update()
        if update is None:
            return
        update.add_updated(locs)

    def add_deleted(self, locs):
        update = self._last_update()
        if update is None:
            return
        update.add_deleted(locs)

    def compute_updates(self, start, mid):
        assert(mid is not None)
        time = self.scheduler.run_number()
        if start == 0:
            self._save_time(time, mid)
            return None
        i = self._saved_index(start)
        if not (mid is None or self._saved_mid[i] == mid):
            raise ValueError('Invalid module id in compute_updates %s instead of %s'\
                             %(mid, self._saved_mid[i]))
        update = self._saved_update[i]
        for j in range(i+1, len(self._saved_update)):
            if self._saved_update[j] is not update:
                update = self._combine_updates(update, j)
                break
        del self._saved_time[i]
        del self._saved_mid[i]
        del self._saved_update[i]
        self._save_time(time, mid)
        return update

    def _combine_updates(self, update, start):
        #TODO reuse cached results if it matches
        nu = IndexUpdate(
            created = bitmap(update.created),
            deleted = bitmap(update.deleted),
            updated = bitmap(update.updated))

        last_u = None
        for i in range(start, len(self._saved_update)):
            u = self._saved_update[i]
            if u is last_u:
                continue
            nu.combine(u)
        #TODO cache results to reuse it if necessary
        return nu

