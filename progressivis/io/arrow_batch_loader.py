from __future__ import annotations

import logging
import numpy as np
import pyarrow as pa
from .base_loader import BaseLoader
from ..core.docstrings import RESULT_DOC
from .. import ProgressiveError
from ..utils.errors import ProgressiveStopIteration
from ..core.module import ReturnRunStep, def_output, document
from ..table.table import PTable
from ..table.dshape import dshape_from_pa_batch
from ..core.utils import (
    normalize_columns,
    force_valid_id_columns_pa,
    integer_types,
    nn,
)
from ..utils.psdict import PDict
from typing import Any, Callable

logger = logging.getLogger(__name__)


@document
@def_output("anomalies", PDict, required=False, doc=("Not implemented,"
                                                     "defined only for "
                                                     "compatibility with other leaders"))
@def_output("result", PTable, doc=RESULT_DOC)
class ArrowBatchLoader(BaseLoader):
    """
    This module reads `RecordBatch` data progressively into a {{PTable}}
     using the ``PyArrow`` backend.
    """
    def __init__(
            self,
            reader: pa.lib.RecordBatchReader,
            n_rows: int,
            filter_: Callable[[pa.RecordBatch], pa.RecordBatch] | None = None,
            force_valid_ids: bool = True,
            fillvalues: dict[str, Any] | None = None,
            throttle: bool | int | float = False,
            **kwds: Any,
    ) -> None:
        r"""
         Args:
            reader: pyarrow.lib.RecordBatchReader
            filter\_: filtering function to be applied on input data at loading time
            force_valid_ids: force renaming of columns to make their names valid identifiers according to the `language definition  <https://docs.python.org/3/reference/lexical_analysis.html#identifiers>`_
            fillvalues: the default values of the columns specified as a dictionary (see :class:`PTable <progressivis.PTable>`)
            throttle: limit the number of rows to be loaded in a step
            kwds: extra keyword args to be passed to  :class:`Module <progressivis.core.Module>` superclass
        """
        super().__init__(**kwds)
        if not isinstance(reader, pa.lib.RecordBatchReader):
            raise ValueError("'reader' must be a RecordBatchReader instance")
        self._reader: pa.lib.RecordBatchReader | None = reader
        self._input_size = n_rows
        self.force_valid_ids = force_valid_ids
        if throttle and isinstance(throttle, integer_types + (float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        if nn(filter_) and not callable(filter_):
            raise ProgressiveError("filter parameter should be callable or None")
        self._filter: Callable[[pa.RecordBatch], pa.RecordBatch] | None = filter_
        self._file_mode = False
        self._table_params: dict[str, Any] = dict(name=self.name, fillvalues=fillvalues)
        self._columns: list[str] | None = None

    def get_progress(self) -> tuple[int, int]:
        return (self._rows_read, self._input_size)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if step_size == 0:  # bug
            logger.error("Received a step_size of 0")
            return self._return_run_step(self.state_ready, steps_run=0)
        if self.throttle:
            step_size = np.min([self.throttle, step_size])
        if self._reader is None:
            return self._return_run_step(self.state_zombie, steps_run=0)
        logger.info("loading %d lines", step_size)
        #import pdb; pdb.set_trace()
        pa_batches = []
        cnt = step_size
        creates = 0
        try:
            assert self._reader
            while cnt > 0:
                bat = self._reader.read_next_batch()
                _nrows = bat.num_rows
                cnt -= _nrows
                creates += _nrows
                self._currow += _nrows
                pa_batches.append(bat)
            assert pa_batches
        except StopIteration:
            self._reader = None
            if not pa_batches:
                return self._return_run_step(self.state_ready, steps_run=0)
        bat_list = [self.process_na_values(bat) for bat in pa_batches]
        if creates == 0:  # should not happen
            logger.error("Received 0 elements")
            raise ProgressiveStopIteration
        if self._filter is not None:
            bat_list = [self._filter(bat) for bat in bat_list]
            creates = sum([len(bat) for bat in bat_list])
        if creates == 0:
            logger.info("frame has been filtered out")
        else:
            assert pa_batches
            self._rows_read += creates
            logger.info("Loaded %d lines", self._rows_read)
            if self.force_valid_ids:
                self._columns = normalize_columns(bat_list[0].schema.names)
                for i, bat in enumerate(bat_list):
                    if bat.schema.names == self._columns:
                        continue
                    bat_list[i] = force_valid_id_columns_pa(bat)
            if self.result is None:
                self._table_params["name"] = self.generate_table_name("table")
                self._table_params["dshape"] = dshape_from_pa_batch(bat_list[0])
                self._table_params["data"] = bat_list[0]
                self._table_params["create"] = True
                self.result = PTable(**self._table_params)
                bat_list = bat_list[1:]
                # if self._imputer is not None:
                #    self._imputer.init(bat.dtypes)
            for bat in bat_list:
                self.result.append(bat)
            # if self._imputer is not None:
            #    self._imputer.add_df(bat)
        return self._return_run_step(self.state_ready, steps_run=creates)
