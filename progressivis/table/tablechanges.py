"""
TableChanges keep track of the changes (creation, updates, deletions)
in tables/columns.
"""

import logging


from ..core.index_update import IndexUpdate
from ..core.bitmap import bitmap
from .tablechanges_base import BaseChanges

logger = logging.getLogger(__name__)


# Bookmark = namedtuple('Bookmark',
#                       ['time', 'refcount', 'update'],
#                       defaults=[1, None])
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

    # To keep track of changes in tables, we maintain a list of Deltas
    # (IndexUpdate).
    # Module slots register when they want to be notified of changes.
    # A registration creates an entry in the _mid_time dict that associates
    # the slot id (mid) with the registration time (the run_number).
    # Then, when a change happens in the table, it is recorded in the
    # Bookmark "update" slot at the associated time.
    # Bookmarks are kept in the _bookmarks list sorted by time.
    # The _times list keeps the times in sorted order too.
    # To find the Bookmark associated with a time, a binary search is used in
    # the _times list, giving the index of the time; the Bookmark is at
    # the same index in the _bookmarks list.
    def __init__(self):
        self._times = []      # list of times sorted
        self._bookmarks = []  # list of bookmarks synchronized with times
        self._mid_time = {}   # time associated with last mid update

    def _last_update(self):
        "Return the last delta to update"
        if not self._bookmarks:
            return None
        return self._bookmarks[-1].update

    def _saved_index(self, time):
        "Return the index of the given time, or -1 if not there"
        try:
            return self._times.index(time)
        except ValueError:
            return -1
        # i = bisect.bisect_left(times, time)
        # if i != len(times) and times[i] == time:
        #     return i
        # return -1

    def _unref_bookmark(self, index):
        bookmark = self._bookmarks[index]
        bookmark.refcount -= 1
        if index != 0:
            return
        # If first bookmark and no slot reference it any more,
        # we can remove it from the lists, as well as all the
        # other with bookmarks not referenced
        while self._bookmarks and self._bookmarks[0].refcount == 0:
            self._times.pop(0)
            self._bookmarks.pop(0)

    def _save_time(self, time, mid):
        if mid in self._mid_time:
            # reset
            logger.debug('Reset received for module %s', mid)
            last_time = self._mid_time[mid]
            i = self._saved_index(last_time)
            assert i != -1
            self._unref_bookmark(i)

        self._mid_time[mid] = time
        # We need to create a Bookmark associated with this time,
        # unless there's already one, and then it should be the last
        # TODO there's a bug down here
        if self._times and self._times[-1] == time:
            # We found a bookmark for time
            bookmark = self._bookmarks[-1]
            bookmark.refcount += 1
            return
        assert self._saved_index(time) == -1  # double check
        # We create a new bookmark
        bookmark = Bookmark(time)
        self._times.append(time)
        if not self._bookmarks:
            update = IndexUpdate(created=None, deleted=None, updated=None)
        else:
            update = self._bookmarks[-1].update
        # if some changes have already been collected, we create a fresh
        # IndexUpdate, otherwise we share the same entry: multiple times
        # will share the same set of changes.
        if update.created or update.deleted or update.updated:
            update = IndexUpdate()
        bookmark.update = update
        self._bookmarks.append(bookmark)
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

    def compute_updates(self, last, now, mid, cleanup=True):
        assert mid is not None
        time = now
        if last == 0:
            self._save_time(time, mid)
            return None
        assert last == self._mid_time[mid]
        i = self._saved_index(last)
        # TODO optimize when nothing has changed
        # reuse the bookmark
        bookmark = self._bookmarks[i]
        update = bookmark.update
        update = self._combine_updates(update, i+1)

        self._unref_bookmark(i)
        del self._mid_time[mid]
        self._save_time(time, mid)
        return update

    def _combine_updates(self, update, start):
        # TODO reuse cached results if it matches
        new_u = IndexUpdate(
            created=bitmap(update.created),
            deleted=bitmap(update.deleted),
            updated=bitmap(update.updated))

        last_u = None
        # Since bookmarks can share their update slots,
        # search for a bookmark with a different value
        for i in range(start, len(self._bookmarks)):
            update = self._bookmarks[i].update
            if update is last_u:
                continue
            new_u.combine(update)
            last_u = new_u
        # TODO cache results to reuse it if possible
        return new_u

    def reset(self, mid):
        if mid not in self._mid_time:
            return
        # reset
        logger.debug(f'Reset received for slot {mid}')
        last_time = self._mid_time[mid]
        del self._mid_time[mid]
        i = self._saved_index(last_time)
        if i != -1:
            self._unref_bookmark(i)
