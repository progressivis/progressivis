
class ProgressiveError(Exception):
    "Errors from ProgressiVis."
    def __init__(self, message=None, details=None):
        self.message = message
        self.details = details


class ProgressiveStopIteration(Exception):
    "Stop Iteration for coroutines"
    def __init__(self, message=None, details=None):
        self.message = message
        self.details = details
