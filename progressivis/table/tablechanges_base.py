"""
Base class for object keeping track of changes in a Table/Column
"""
from abc import ABCMeta, abstractmethod


class BaseChanges(metaclass=ABCMeta):
    "Base class for object keeping track of changes in a Table"
    def __str__(self):
        return type(self)

    @abstractmethod
    def add_created(self, locs):
        "Add ids of items created in the Table"
        pass

    @abstractmethod
    def add_updated(self, locs):
        "Add ids of items updated in the Table"
        pass

    @abstractmethod
    def add_deleted(self, locs):
        "Add ids of items deleted from the Table"
        pass

    @abstractmethod
    def compute_updates(self, last, now, mid, cleanup=True):
        "Compute and return the list of changes as an IndexUpdate or None"
        return None

    @abstractmethod
    def reset(self, mid):
        "Clears all data related to mid"
        pass
