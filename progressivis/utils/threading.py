"""Multi-Threading."""
from __future__ import absolute_import, division, print_function
import six

multi_threading = False
# multithreading is disabled and it will be removed

class FakeLock(object):
    def acquire(self, blocking=1):
        _ = blocking  # keeps pylint happy
        return False

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class FakeCondition(object):
    def __init__(self, lock=None):
        self._lock = lock  #  still keeps pylint happy

    def wait(self):
        pass

    def notify(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


if multi_threading:
    from threading import Thread, Lock, RLock, Condition

else:
    Lock = FakeLock
    RLock = FakeLock
    Condition = FakeCondition

    class Thread(object):  # fake threads for debug
        def __init__(self, group=None, target=None, name=None,
                     args=(), kwargs=None):
            self._group = group
            self._target = target
            self._name = name
            self._args = args
            self._kwargs = kwargs or {}

        def run(self):
            return self._target(*self._args, **self._kwargs)

        def start(self):
            return self.run()

        def join(self, timeout=None):
            pass
