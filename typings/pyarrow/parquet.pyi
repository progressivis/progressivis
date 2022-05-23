from typing import Any, Generator


class ParquetFile:
    def __init__(self, *args: Any, **kw: Any):
        ...

    def iter_batches(self, **kw: Any) -> Generator:
        ...
