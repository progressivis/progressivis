"""
Base class for object keeping track of changes in a PTable/PColumn
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.index_update import IndexUpdate


class BaseChanges(metaclass=ABCMeta):
    "Base class for object keeping track of changes in a PTable"

    def __str__(self) -> str:
        return str(type(self))

    @abstractmethod
    def add_created(self, locs: Any) -> None:
        "Add ids of items created in the PTable"
        pass

    @abstractmethod
    def add_updated(self, locs: Any) -> None:
        "Add ids of items updated in the PTable"
        pass

    @abstractmethod
    def add_deleted(self, locs: Any) -> None:
        "Add ids of items deleted from the PTable"
        pass

    @abstractmethod
    def compute_updates(
        self, last: int, now: int, mid: str, cleanup: bool = True
    ) -> Optional[IndexUpdate]:
        "Compute and return the list of changes as an IndexUpdate or None"
        return None

    @abstractmethod
    def reset(self, mid: str) -> None:
        "Clears all data related to mid"
        pass
