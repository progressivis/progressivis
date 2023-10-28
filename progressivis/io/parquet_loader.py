from __future__ import annotations

import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .. import ProgressiveError
from .base_loader import BaseLoader
from ..core.docstrings import FILENAMES_DOC, RESULT_DOC
from ..utils.errors import ProgressiveStopIteration
from ..utils.inspect import filter_kwds, extract_params_docstring
from ..core.module import ReturnRunStep, def_input, def_output, document
from ..table.table import PTable
from ..table.dshape import dshape_from_pa_batch
from ..core.utils import (
    normalize_columns,
    force_valid_id_columns_pa,
    integer_types,
    is_slice,
    nn,
)
from ..utils import PDict
from typing import List, Dict, Any, Callable, Optional, Union, Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.module import ModuleState
    # from ..stats.utils import SimpleImputer
    import io

logger = logging.getLogger(__name__)


@document
@def_input("filenames", PTable, required=False, doc=FILENAMES_DOC)
@def_output("anomalies", PDict, required=False, doc=("provides: ``anomalies"
                                                     "['skipped_cnt'] ="
                                                     " <skipped-rows-cnt>``"))
@def_output("result", PTable, doc=RESULT_DOC)
class ParquetLoader(BaseLoader):
    def __init__(
        self,
        filepath_or_buffer: Optional[Any] = None,
        filter_: Optional[Callable[[pa.RecordBatch], pa.RecordBatch]] = None,
        force_valid_ids: bool = True,
        fillvalues: Optional[Dict[str, Any]] = None,
        throttle: Union[bool, int, float] = False,
        # imputer: Optional[SimpleImputer] = None,
        # drop_na: Optional[bool] = True,
        columns: Optional[List[str]] = None,
        **kwds: Any,
    ) -> None:
        r"""
         Args:
            filepath_or_buffer: string, path object or file-like object accepted by :func:`pyarrow.csv.open_csv`
            filter\_: filtering function to be applied on input data at loading time
            force_valid_ids: force renaming of columns to make their names valid identifiers according to the `language definition  <https://docs.python.org/3/reference/lexical_analysis.html#identifiers>`_
            fillvalues: the default values of the columns specified as a dictionary (see :class:`PTable <progressivis.table.PTable>`)
            throttle: limit the number of rows to be loaded in a step
            read_options: Options for the CSV reader (see :class:`pyarrow.csv.ReadOptions`)
            parse_options : Options for the CSV parser (see :class:`pyarrow.csv.ParseOptions`)
            convert_options : Options for converting CSV data (see :class:`pyarrow.csv.ConvertOptions`)
            kwds: extra keyword args to be passed to  :class:`pyarrow.parquet.ParquetFile`, :func:`pyarrow.parquet.iter_batches` and :class:`Module <progressivis.core.Module>` superclass
        """
        super().__init__(**kwds)
        self.default_step_size = kwds.get("batch_size", 1000)  # initial guess
        kwds.setdefault("batch_size", self.default_step_size)
        # Filter out the module keywords from the parquet loader keywords
        self.pqfile_kwds: Dict[str, Any] = filter_kwds(kwds, pq.ParquetFile)
        self.iter_batches_kwds: Dict[str, Any] = filter_kwds(
            kwds, pq.ParquetFile.iter_batches
        )
        if columns is not None:
            self.iter_batches_kwds['columns'] = columns
        # When called with a specified chunksize, it returns a parser
        self.filepath_or_buffer = filepath_or_buffer
        self.force_valid_ids = force_valid_ids
        if throttle and isinstance(throttle, integer_types + (float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        self.parser: Optional[Generator[Any, Any, Any]] = None

        if nn(filter_) and not callable(filter_):
            raise ProgressiveError("filter parameter should be callable or None")
        self._filter: Optional[Callable[[pa.RecordBatch], pa.RecordBatch]] = filter_
        self._input_stream: Optional[
            io.IOBase
        ] = None  # stream that returns a position through the 'tell()' method

        self._file_mode = False
        self._table_params: Dict[str, Any] = dict(name=self.name, fillvalues=fillvalues)
        # self._imputer = imputer
        # self._drop_na = drop_na
        self._columns: Optional[List[str]] = None

    def validate_parser(self, run_number: int) -> ModuleState:
        if self.parser is None:
            if nn(self.filepath_or_buffer):
                try:
                    self.parser = pq.ParquetFile(
                        self.open(self.filepath_or_buffer), **self.pqfile_kwds
                    ).iter_batches(**self.iter_batches_kwds)
                except IOError as e:
                    logger.error("Cannot open file %s: %s", self.filepath_or_buffer, e)
                    self.parser = None
                    return self.state_terminated
                self.filepath_or_buffer = None
                self._file_mode = True
            else:
                if not self.has_input_slot("filenames"):
                    return self.state_terminated
                fn_slot = self.get_input_slot("filenames")
                if fn_slot.output_module is None:
                    return self.state_terminated
                fn_slot.update(run_number)
                if fn_slot.deleted.any() or fn_slot.updated.any():
                    raise ProgressiveError("Cannot handle input file changes")
                df = fn_slot.data()
                while self.parser is None:
                    indices = fn_slot.created.next(length=1)
                    assert is_slice(indices)
                    if indices.stop == indices.start:
                        return self.state_blocked
                    filename = df.at[indices.start, "filename"]
                    try:
                        self.parser = pq.ParquetFile(
                            self.open(filename), **self.pqfile_kwds
                        ).iter_batches(**self.iter_batches_kwds)
                    except IOError as e:
                        logger.error("Cannot open file %s: %s", filename, e)
                        self.parser = None
                        # fall through
        return self.state_ready

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if step_size == 0:  # bug
            logger.error("Received a step_size of 0")
            return self._return_run_step(self.state_ready, steps_run=0)
        if self.throttle:
            step_size = np.min([self.throttle, step_size])
        status = self.validate_parser(run_number)
        if status == self.state_terminated:
            raise ProgressiveStopIteration("no more filenames")
        elif status == self.state_blocked:
            return self._return_run_step(status, steps_run=0)
        elif status != self.state_ready:
            logger.error("Invalid state returned by validate_parser: %d", status)
            raise ProgressiveStopIteration("Unexpected situation")
        logger.info("loading %d lines", step_size)
        pa_batches = []
        cnt = step_size
        creates = 0
        try:
            assert self.parser
            while cnt > 0:
                bat = next(self.parser)
                _nrows = bat.num_rows
                cnt -= _nrows
                creates += _nrows
                self._currow += _nrows
                pa_batches.append(bat)
            assert pa_batches
        except StopIteration:
            if self.has_input_slot("filenames"):
                fn_slot = self.get_input_slot("filenames")
                if (
                    fn_slot is None or fn_slot.output_module is None
                ) and not self._file_mode:
                    raise
            self.parser = None
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
                if self._columns is None:
                    self._column = normalize_columns(bat_list[0].schema.names)
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


parquet_docstring = (
    "Parquet_Loader("
    + extract_params_docstring(ParquetLoader.__init__, only_defaults=True)
    + ",force_valid_ids=False,id=None,scheduler=None,tracer=None,predictor=None"
    + ",storage=None,input_descriptors=[],output_descriptors=[])"
)
try:
    if not ParquetLoader.doc_building():
        ParquetLoader.__init__.__func__.__doc__ = parquet_docstring  # type: ignore
except Exception:
    try:
        ParquetLoader.__init__.__doc__ = parquet_docstring
    except Exception:
        pass
