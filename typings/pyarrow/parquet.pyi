from typing import Any, Generator
from pyarrow import Table


def read_table(arg: str) -> Table:
    ...


class ParquetFile:
    def __init__(self, *args: Any, **kw: Any):
        ...

    def iter_batches(self, **kw: Any) -> Generator:
        ...
