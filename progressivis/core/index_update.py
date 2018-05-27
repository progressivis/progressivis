"""
Return changes inside tables/columns/bitmaps.
"""
from __future__ import absolute_import, division, print_function

from ..core.bitmap import bitmap


class IndexUpdate(object):
    """
    IndexUpdate is used to keep track of chages occuring in linear data structures
    such as tables, columns, or bitmaps.
    """
    def __init__(self, created=None, updated=None, deleted=None):
        created = created if isinstance(created, bitmap) else bitmap(created)
        updated = updated if isinstance(updated, bitmap) else bitmap(updated)
        deleted = deleted if isinstance(deleted, bitmap) else bitmap(deleted)
        self.created = created
        self.updated = updated
        self.deleted = deleted

    def __repr__(self):
        return "IndexUpdate(created=%s,updated=%s,deleted=%s)" % (
            repr(self.created), repr(self.updated), repr(self.deleted))

    def clear(self):
        "Clear the changes"
        self.created.clear()
        self.updated.clear()
        self.deleted.clear()

    def test(self, verbose=False):
        "Test if the IndexUpdate is valid"
        b = bool(self.created & self.updated) \
          or bool(self.created & self.deleted) \
          or bool(self.updated & self.deleted)
        if verbose and b:  # pragma no cover
            print("self.created & self.updated", self.created & self.updated)
            print("self.created & self.deleted", self.created & self.deleted)
            print("self.updated & self.deleted", self.updated & self.deleted)
        return not b

    def add_created(self, bm):
        "Add created items"
        self.created.update(bm)
        self.deleted -= bm
        self.updated -= bm

    def add_updated(self, bm):
        "Add updated items"
        self.updated.update(bm)
        self.updated -= self.created
        self.updated -= self.deleted

    def add_deleted(self, bm):
        "Add deleted items"
        self.deleted.update(bm)
        self.updated -= bm
        self.created -= bm

    def combine(self, other,
                update_created=True, update_updated=True, update_deleted=True):
        "Combine this IndexUpdate with another IndexUpdate"
        if other.deleted:
            # if not created yet, no need to delete
            toignore = other.deleted & self.created
            if update_created:
                self.created -= toignore
            if update_updated:
                self.updated -= other.deleted
            if update_deleted:
                self.deleted |= other.deleted - toignore
        if other.created:
            # if re-created, pretend updated
            toupdate = other.created & self.deleted
            if update_deleted:
                self.deleted -= toupdate
            if update_updated:
                self.updated -= other.updated | toupdate
            if update_created:
                self.created |= other.created - toupdate
        if other.updated and update_updated:
            toignore = self.created
            self.updated |= other.updated
            self.updated -= toignore
        assert self.test()  # test invariant
        return

    def __eq__(self, other):
        return self is other or \
          (self.created == other.created and
           self.updated == other.updated and
           self.deleted == other.deleted)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        "Copy this Indexupdate"
        return IndexUpdate(created=bitmap(self.created),
                           updated=bitmap(self.updated),
                           deleted=bitmap(self.deleted))

NIL_IU = IndexUpdate()
