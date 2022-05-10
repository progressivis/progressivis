from .csv import RecordBatch
from typing import List, Any

class ArrowInvalid(Exception): ...

class Table:
    @staticmethod
    def from_batches(v: List[RecordBatch]) -> Table: ...
