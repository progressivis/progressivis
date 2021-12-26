
from typing import Optional


class ProgressiveError(Exception):
    "Errors from ProgressiVis."
    def __init__(self, message: Optional[str] = None, details: Optional[str] = None):
        self.message = message
        self.details = details


class ProgressiveStopIteration(Exception):
    "Stop Iteration for coroutines"
    def __init__(self, message: Optional[str] = None, details: Optional[str] = None):
        self.message = message
        self.details = details
