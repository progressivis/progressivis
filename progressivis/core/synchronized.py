from __future__ import absolute_import, division, print_function


def synchronized(method):
    def sync_method(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return sync_method
