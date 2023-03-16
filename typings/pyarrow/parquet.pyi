from typing import Any, List, Generator, Dict
from pyarrow import Table
import numpy as np

def read_table(arg: str) -> Table:
    ...


class DataType:
    def __init__(self, *args: Any, **kw: Any) -> None:
        ...

    def to_pandas_dtype(self) -> List[np.dtype[Any]]:
        ...


class ColumnSchema:
    def __init__(self, *args: Any, **kw: Any) -> None:
        ...

class Schema:
    def __init__(self, *args: Any, **kw: Any) -> None:
        ...

    @property
    def names(self) -> List[str]:
        ...

    @property
    def types(self) -> List[DataType]:
        ...



class ParquetSchema:
    def __init__(self, *args: Any, **kw: Any) -> None:
        ...

    def to_arrow_schema(self) -> Schema:
        ...

    def column(self, i: int) -> ColumnSchema:
        ...

class FileMetaData:
    created_by: str
    format_version: str
    metadata: Dict[bytes, bytes]
    num_columns: int
    num_rows: int
    schema: ParquetSchema
    serialized_size: int

class ParquetFile:
    metadata: FileMetaData
    def __init__(self, *args: Any, **kw: Any) -> None:
        ...

    def iter_batches(self, **kw: Any) -> Generator[Any, Any, Any]:
        ...

    @property
    def schema(self) -> ParquetSchema:
        ...
