"""
TableChanges keep track of the changes (creation, updates, deletions) in tables/columns.
"""
from __future__ import absolute_import, division, print_function

import bisect
import logging


from ..core.index_update import IndexUpdate
from ..core.bitmap import bitmap
from .tablechanges_base import BaseChanges

logger = logging.getLogger(__name__)


class Bookmark(object):
    # pylint: disable=too-few-public-methods
    "Bookmark for changes"
    __slots__ = ['time', 'refcount', 'update']
    def __init__(self, time, refcount=1, update=None):
        self.time = time
        self.refcount = refcount
        self.update = update


class TableChanges(BaseChanges):
    "Keep track of changes in tables"
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._times = []     # list of times sorted
        self._bookmarks = [] # list of bookmarks synchronized with times
        self._mid_time = {}  # time associated with last mid update

    def _last_update(self):
        if not self._bookmarks:
            return None
        return self._bookmarks[-1].update

    def _saved_index(self, time):
        times = self._times
        i = bisect.bisect_left(times, time)
        if i != len(times) and times[i] == time:
            return i
        return -1

    def _save_time(self, time, mid):
        if  mid in self._mid_time:
            # reset
            logger.debug('Reset received for module %s', mid)
            last_time = self._mid_time[mid]
            i = self._saved_index(last_time)
            assert i != -1
            bookmark = self._bookmarks[i]
            if bookmark.time != last_time:
                raise ValueError('Invalid module time in compute_updates %s instead of %s',
                                 last_time, bookmark.time)
            bookmark.refcount -= 1
            if bookmark.refcount == 0:
                del self._times[i]
                del self._bookmarks[i]
            else:
                logger.info('refcount is not 0 in reset %s', mid)
            del self._mid_time[mid]

        self._mid_time[mid] = time
        if self._times and self._times[-1] == time:
            bookmark = self._bookmarks[-1]
            bookmark.refcount += 1
            return
        assert  self._saved_index(time) == -1  # double check
        bookmark = Bookmark(time)
        self._times.append(time)
        if not self._bookmarks:
            update = IndexUpdate(created=None, deleted=None, updated=None)
        else:
            update = self._bookmarks[-1].update
        self._bookmarks.append(bookmark)
        # if some changes have already been collected, we create a fresh IndexUpdate,
        # otherwise we share the same entry: multiple times will share the same set of changes.
        if update.created or update.deleted or update.updated:
            update = IndexUpdate()
        bookmark.update = update
        assert len(self._bookmarks) == len(self._times)

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

    def compute_updates(self, last, mid, cleanup=True):
        assert mid is not None
        time = self.scheduler.run_number()
        if last == 0:
            self._save_time(time, mid)
            return None
        assert last == self._mid_time[mid]
        i = self._saved_index(last)
        bookmark = self._bookmarks[i]
        update = bookmark.update
        for j in range(i+1, len(self._bookmarks)):
            if self._bookmarks[j].update is not update:
                update = self._combine_updates(update, j)
                break

        bookmark.refcount -= 1
        if bookmark.refcount == 0:
            del self._times[i]
            del self._bookmarks[i]
        else:
            logger.info('refcount is not 0 in %s', mid)
        del self._mid_time[mid]
        self._save_time(time, mid)
        return update

    def _combine_updates(self, update, start):
        #TODO reuse cached results if it matches
        new_u = IndexUpdate(
            created=bitmap(update.created),
            deleted=bitmap(update.deleted),
            updated=bitmap(update.updated))

        last_u = None
        for i in range(start, len(self._bookmarks)):
            update = self._bookmarks[i].update
            if update is last_u:
                continue
            new_u.combine(update)
            last_u = new_u
        #TODO cache results to reuse it if necessary
        return new_u
