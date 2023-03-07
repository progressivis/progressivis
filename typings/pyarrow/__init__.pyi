from typing import Optional, List, Any


class ArrowInvalid(Exception):
    ...


class ArrowNotImplementedError(Exception):
    ...


class Schema:

    names: List[str]
    types: List[Any]
    ...


class RecordBatch:

    num_rows: int
    schema: Schema
    columns: Any

    def from_pandas(self, schema: Optional[Schema] = None): ...
    def __len__(self): ...

    @staticmethod
    def from_arrays(v: Any, names: List[str]):
        ...


class Table:
    @staticmethod
    def from_batches(v: List[RecordBatch]) -> "Table": ...


class compute:
    @staticmethod
    def sum(v: Any):
        ...

    @staticmethod
    def invert(v: Any):
        ...

    @staticmethod
    def or_(v1: Any, v2: Any):
        ...

def array(v: Any, type: Any) -> Any:
    ...

class TimestampArray:
    ...
