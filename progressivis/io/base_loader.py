from __future__ import annotations

import logging
from ..table.module import PTableModule
from .. import SlotDescriptor
from ..table.table import PTable
from ..utils import PDict
from ..core.utils import nn
from typing import Optional, Any, TYPE_CHECKING
import pyarrow as pa
import pyarrow.compute
from ..core.utils import (
    filepath_to_buffer,
    _infer_compression,
)

logger = logging.getLogger(__name__)

# from typing import List, Dict, Any, Callable, Optional, Union, Generator, TYPE_CHECKING

if TYPE_CHECKING:
    import io


class BaseLoader(PTableModule):
    inputs = [SlotDescriptor("filenames", type=PTable, required=False)]
    outputs = [
        SlotDescriptor("anomalies", type=PDict, required=False),
    ]

    def __init__(self, *args, **kw):
        self._rows_read: int = 0
        self._anomalies: Optional[PDict] = None
        super().__init__(*args, **kw)
        self._input_stream: Optional[
            io.IOBase
        ] = None  # stream that returns a position through the 'tell()' method
        self._encoding: Any = None
        self._input_encoding: Optional[str] = None
        self._input_compression: Optional[str] = None
        self._input_size = 0  # length of the file or input stream when available
        self._last_opened: Any = None
        self._compression: Any = "infer"
        self._encoding: None
        self._currow = 0
        self._fs: Any = kw.get("fs")

    def rows_read(self) -> int:
        return self._rows_read

    def is_ready(self) -> bool:
        if self.has_input_slot("filenames"):
            # Can be called before the first update so fn.created can be None
            fn = self.get_input_slot("filenames")
            if fn.created is None or fn.created.any():
                return True
        return super().is_ready()

    def starting(self) -> None:
        super().starting()
        opt_slot = self.get_output_slot("anomalies")
        if opt_slot:
            logger.debug("Maintaining anomalies")
            self.maintain_anomalies(True)
        else:
            logger.debug("Not maintaining anomalies")
            self.maintain_anomalies(False)

    def maintain_anomalies(self, yes: bool = True) -> None:
        if yes and self._anomalies is None:
            self._anomalies = PDict(dict(skipped_cnt=0, invalid_values=set()))
        elif not yes:
            self._anomalies = None

    def anomalies(self) -> Optional[PDict]:
        return self._anomalies

    def get_data(self, name: str) -> Any:
        if name == "anomalies":
            return self.anomalies()
        return super().get_data(name)

    def process_na_values(self, bat) -> pa.RecordBatch:
        null_mask = None
        has_null = False
        for col in bat:
            if not col.null_count:
                continue
            has_null = True
            try:
                null_mask = pa.compute.or_(null_mask, col.is_null())
            except pa.ArrowNotImplementedError:
                assert null_mask is None
                null_mask = col.is_null()
        if not has_null:
            return bat
        if nn(self._anomalies):
            self._anomalies["skipped_cnt"] += pa.compute.sum(null_mask).as_py()  # type: ignore
        return bat.filter(pa.compute.invert(null_mask))

    def is_data_input(self) -> bool:
        # pylint: disable=no-self-use
        "Return True if this module brings new data"
        return True

    def open(self, filepath: Any) -> io.IOBase:
        if nn(self._input_stream):
            self.close()
        compression: Optional[str] = _infer_compression(filepath, self._compression)
        istream: io.IOBase
        encoding: Optional[str]
        size: int
        (istream, encoding, compression, size) = filepath_to_buffer(
            filepath, encoding=self._encoding, compression=compression, fs=self._fs
        )
        self._input_stream = istream
        self._input_encoding = encoding
        self._input_compression = compression
        self._input_size = size
        self._last_opened = filepath
        self._currow = 0
        return istream

    def close(self) -> None:
        if self._input_stream is None:
            return
        try:
            self._input_stream.close()
            # pylint: disable=bare-except
        except Exception:
            pass
        self._input_stream = None
        self._input_encoding = None
        self._input_compression = None
        self._input_size = 0
