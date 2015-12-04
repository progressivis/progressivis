def synchronized(method):
    def sync_method(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return sync_method

