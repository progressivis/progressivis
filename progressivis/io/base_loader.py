from __future__ import annotations

import logging
from ..table.module import TableModule
from .. import SlotDescriptor
from ..table.table import Table
from ..utils import PsDict
from ..core.utils import nn
from typing import Optional, Any
import pyarrow as pa

logger = logging.getLogger(__name__)


class BaseLoader(TableModule):
    inputs = [SlotDescriptor("filenames", type=Table, required=False)]
    outputs = [
        SlotDescriptor("anomalies", type=PsDict, required=False),
    ]

    def __init__(self, *args, **kw):
        self._rows_read: int = 0
        self._anomalies: Optional[PsDict] = None
        super().__init__(*args, **kw)

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
            self._anomalies = PsDict(dict(skipped_cnt=0, invalid_values=set()))
        elif not yes:
            self._anomalies = None

    def anomalies(self) -> Optional[PsDict]:
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
