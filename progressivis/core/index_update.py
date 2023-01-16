"""
Changes (Delta) inside tables/columns/pintsets.
"""
from __future__ import annotations

from typing import Optional, Any

from ..core.pintset import PIntSet


class IndexUpdate:
    """
    IndexUpdate is used to keep track of chages occuring in linear data
    structures such as tables, columns, or pintsets.
    """

    def __init__(
        self,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ):
        created = created if isinstance(created, PIntSet) else PIntSet(created)
        updated = updated if isinstance(updated, PIntSet) else PIntSet(updated)
        deleted = deleted if isinstance(deleted, PIntSet) else PIntSet(deleted)
        self.created: PIntSet = created
        self.updated: PIntSet = updated
        self.deleted: PIntSet = deleted

    def __repr__(self) -> str:
        return "IndexUpdate(created=%s,updated=%s,deleted=%s)" % (
            repr(self.created),
            repr(self.updated),
            repr(self.deleted),
        )

    def clear(self) -> None:
        "Clear the changes"
        self.created.clear()
        self.updated.clear()
        self.deleted.clear()

    def test(self, verbose: bool = False) -> bool:
        "Test if the IndexUpdate is valid"
        b = (
            bool(self.created & self.updated)
            or bool(self.created & self.deleted)
            or bool(self.updated & self.deleted)
        )
        if verbose and b:  # pragma no cover
            print("self.created & self.updated", self.created & self.updated)
            print("self.created & self.deleted", self.created & self.deleted)
            print("self.updated & self.deleted", self.updated & self.deleted)
        return not b

    def add_created(self, bm: PIntSet) -> None:
        "Add created items"
        self.created.update(bm)
        self.deleted -= bm
        self.updated -= bm

    def add_updated(self, bm: PIntSet) -> None:
        "Add updated items"
        self.updated.update(bm)
        self.updated -= self.created
        self.updated -= self.deleted

    def add_deleted(self, bm: PIntSet) -> None:
        "Add deleted items"
        self.deleted.update(bm)
        self.updated -= bm
        self.created -= bm

    def combine(
        self,
        other: Optional[IndexUpdate],
        update_created: bool = True,
        update_updated: bool = True,
        update_deleted: bool = True,
    ) -> None:
        "Combine this IndexUpdate with another IndexUpdate"
        if other is None:
            return
        if other.deleted:
            # No need to expose created when they become deleted
            created_deleted = self.created & other.deleted
            if update_created:
                self.created -= other.deleted
            if update_updated:
                self.updated -= other.deleted
            if update_deleted:
                self.deleted |= other.deleted
                self.deleted -= created_deleted
        if other.created:
            # if re-created, pretend updated
            created_deleted = other.created & self.deleted
            if update_deleted:
                self.deleted -= created_deleted
            if update_created:
                self.created |= other.created
                self.created -= created_deleted
            if update_updated:
                self.updated |= created_deleted
                self.updated -= self.created
            # TODO handle case with created & updated -> created
        if other.updated and update_updated:
            created_updated = self.created & other.updated
            self.updated |= other.updated - created_updated
            self.updated -= other.created
        assert self.test(verbose=True)  # test invariant

    def __eq__(self, other: Any) -> bool:
        return self is other or (
            isinstance(other, IndexUpdate)
            and self.created == other.created
            and self.updated == other.updated
            and self.deleted == other.deleted
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def copy(self) -> IndexUpdate:
        "Copy this Indexupdate"
        return IndexUpdate(
            created=PIntSet(self.created),
            updated=PIntSet(self.updated),
            deleted=PIntSet(self.deleted),
        )


NIL_IU = IndexUpdate()
