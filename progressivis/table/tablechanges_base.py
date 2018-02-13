"""
Base class for object keeping track of changes in a Table/Column
"""
from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
import six


@six.python_2_unicode_compatible
class BaseChanges(six.with_metaclass(ABCMeta, object)):
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
    def compute_updates(self, last, mid, cleanup=True):
        "Compute and return the list of changes as an IndexUpdate or None"
        return None
