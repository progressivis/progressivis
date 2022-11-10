"""
Base class for object keeping track of changes in a Table/Column
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.delta import Delta


class BaseChanges(metaclass=ABCMeta):
    "Base class for object keeping track of changes in a Table"

    def __str__(self) -> str:
        return str(type(self))

    @abstractmethod
    def add_created(self, locs: Any) -> None:
        "Add ids of items created in the Table"
        pass

    @abstractmethod
    def add_updated(self, locs: Any) -> None:
        "Add ids of items updated in the Table"
        pass

    @abstractmethod
    def add_deleted(self, locs: Any) -> None:
        "Add ids of items deleted from the Table"
        pass

    @abstractmethod
    def compute_updates(
        self, last: int, now: int, mid: str, cleanup: bool = True
    ) -> Optional[Delta]:
        "Compute and return the list of changes as a Delta or None"
        return None

    @abstractmethod
    def reset(self, mid: str) -> None:
        "Clears all data related to mid"
        pass
