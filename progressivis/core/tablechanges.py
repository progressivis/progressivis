from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
import six


@six.python_2_unicode_compatible
class BaseChanges(six.with_metaclass(ABCMeta, object)):
    def __str__(self):
        return type(self)

    @abstractmethod
    def add_created(self, locs):
        pass

    @abstractmethod
    def add_updated(self, locs):
        pass

    @abstractmethod
    def add_deleted(self, locs):
        pass

    @abstractmethod
    def compute_updates(self, start, mid):
        return None
