from __future__ import annotations

import logging
import copy
from functools import partial
import numpy as np
import pyarrow as pa
import pyarrow.csv
from .base_loader import BaseLoader
from .. import ProgressiveError
from ..utils.errors import ProgressiveStopIteration
from ..utils.inspect import extract_params_docstring
from ..core.module import ReturnRunStep
from ..table.table import Table
from ..table.dshape import dshape_from_pa_batch
from ..core.utils import (
    filepath_to_buffer,
    _infer_compression,
    normalize_columns,
    force_valid_id_columns_pa,
    integer_types,
    is_str,
    is_slice,
    nn,
)

from typing import List, Dict, Any, Callable, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.module import ModuleState
    from ..stats.utils import SimpleImputer
    import io

logger = logging.getLogger(__name__)


class PACSVLoader(BaseLoader):
    def __init__(
        self,
        filepath_or_buffer: Optional[Any] = None,
        filter_: Optional[Callable[[pa.RecordBatch], pa.RecordBatch]] = None,
        force_valid_ids: bool = True,
        fillvalues: Optional[Dict[str, Any]] = None,
        throttle: Union[bool, int, float] = False,
        imputer: Optional[SimpleImputer] = None,
        read_options: Optional[pa.csv.ReadOptions] = None,
        parse_options: Optional[pa.csv.ParseOptions] = None,
        convert_options: Optional[pa.csv.ConvertOptions] = None,
        drop_na: Optional[bool] = True,
        max_invalid_per_block: int = 100,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.default_step_size = kwds.get("chunksize", 1000)  # initial guess
        kwds.setdefault("chunksize", self.default_step_size)
        # Filter out the module keywords from the csv loader keywords
        self._read_options = read_options
        self._parse_options = parse_options
        self._convert_options = convert_options
        # When called with a specified chunksize, it returns a parser
        self.filepath_or_buffer = filepath_or_buffer
        self.force_valid_ids = force_valid_ids
        if throttle and isinstance(throttle, integer_types + (float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        self._parser: Optional[pa.csv.CSVStreamingReader] = None
        self._parser_func: Optional[Callable] = None
        self._compression: Any = "infer"
        self._encoding: Any = read_options.encoding if read_options else None
        if nn(filter_) and not callable(filter_):
            raise ProgressiveError("filter parameter should be callable or None")
        self._filter: Optional[Callable[[pa.RecordBatch], pa.RecordBatch]] = filter_
        self._input_stream: Optional[
            io.IOBase
        ] = None  # stream that returns a position through the 'tell()' method
        self._input_encoding: Optional[str] = None
        self._input_compression: Optional[str] = None
        self._input_size = 0  # length of the file or input stream when available
        self._file_mode = False
        self._table_params: Dict[str, Any] = dict(name=self.name, fillvalues=fillvalues)
        self._currow = 0
        self._imputer = imputer
        self._last_opened: Any = None
        self._drop_na = drop_na
        self._max_invalid_per_block = max_invalid_per_block
        self._columns: Optional[List[str]] = None
        self._last_schema: Optional[Union[pa.Schema, Dict[Any, Any]]] = None

    @property
    def parser(self) -> pa.csv.CSVStreamingReader:
        """
        When data contains invalid values pyarrow.csv.open_csv can raise an ArrowInvalid
        exception (even though no rows were fetched yet ...)
        so one prefers a late instanciation of the CSVStreamingReader
        """
        if self._parser is None:
            assert self._parser_func is not None
            self._parser = self._parser_func()
        return self._parser

    def open(self, filepath: Any) -> io.IOBase:
        if nn(self._input_stream):
            self.close()
        compression: Optional[str] = _infer_compression(filepath, self._compression)
        istream: io.IOBase
        encoding: Optional[str]
        size: int
        (istream, encoding, compression, size) = filepath_to_buffer(
            filepath, encoding=self._encoding, compression=compression
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

    def get_progress(self) -> Tuple[int, int]:
        if self._input_size == 0:
            return (0, 0)
        if self._input_stream is None:
            return (0, 0)
        pos = self._input_stream.tell()
        return (pos, self._input_size)

    def validate_parser(self, run_number: int) -> ModuleState:
        if self._parser_func is None:
            if nn(self.filepath_or_buffer):
                try:
                    self._parser_func = partial(
                        pa.csv.open_csv,
                        self.open(self.filepath_or_buffer),
                        read_options=self._read_options,
                        parse_options=self._parse_options,
                        convert_options=self._convert_options,
                    )
                    self._parser = None
                except IOError as e:
                    logger.error("Cannot open file %s: %s", self.filepath_or_buffer, e)
                    self._parser = None
                    self._parser_func = None
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
                while self._parser_func is None:
                    indices = fn_slot.created.next(length=1)
                    assert is_slice(indices)
                    if indices.stop == indices.start:
                        return self.state_blocked
                    filename = df.at[indices.start, "filename"]
                    try:
                        self._parser_func = partial(
                            pa.csv.open_csv,
                            self.open(filename),
                            read_options=self._read_options,
                            parse_options=self._parse_options,
                            convert_options=self._convert_options,
                        )
                        self._parser = None
                    except IOError as e:
                        logger.error("Cannot open file %s: %s", filename, e)
                        self._parser = None
                        self._parser_func = None
                        # fall through
        return self.state_ready

    def recovering(self) -> pa.RecordBatch:
        """
        transforms invalid values in NA values
        """
        currow: int = self._currow

        def _reopen_last():
            if self._last_opened is None:
                raise ValueError("Recovery failed")
            if is_str(self._last_opened):
                return self.open(self._last_opened)
            if hasattr(self._last_opened, "seek"):
                self._last_opened.seek(
                    0
                )  # NB seek is defined but not supported on HTTPResponse ...
                return self._last_opened
            raise ValueError("Recovery failed (2)")

        if self._last_schema is None:
            if not (
                self._convert_options is None
                or self._convert_options.column_types is None
            ):
                assert self._convert_options.column_types
                self._last_schema = self._convert_options.column_types
            else:
                raise ValueError(
                    "Cannot guess the schema."
                    " Consider defining 'column_type' in ConvertOptions parameter"
                )
        if self._read_options is not None:
            ropts = copy.copy(self._read_options)
        else:
            ropts = pa.csv.ReadOptions()
        if self._convert_options is not None:
            cvopts = copy.copy(self._convert_options)
        else:
            cvopts = pa.csv.ConvertOptions()
        popts = self._parse_options
        if ropts.skip_rows_after_names:
            assert ropts.skip_rows_after_names <= currow
        ropts.skip_rows_after_names = currow
        cvopts.null_values = [""]
        cvopts.column_types = self._last_schema
        for _ in range(self._max_invalid_per_block):  # avoids infinite loop
            try:
                istream = _reopen_last()
                readr = pa.csv.open_csv(
                    istream,
                    read_options=ropts,  # cannot use read_csv
                    convert_options=cvopts,  # here because
                    parse_options=popts,
                )  # (apparently)
                chunk = readr.read_next_batch()  # read_csv cannot read only one batch
            except pa.ArrowInvalid as ee:
                args = ee.args[0].split("'")
                if len(args) != 3:
                    raise
                if not (
                    "CSV conversion error to" in args[0] or "invalid value " in args[0]
                ):
                    raise
                invalid = args[1]
                cvopts.null_values += [invalid]  # cannot append
                if nn(self._anomalies):
                    self._anomalies["invalid_values"].add(invalid)  # type: ignore
                continue
            break
        else:
            raise ValueError("Internal error: infinite loop in recovering")
        if self._read_options is None:
            self._read_options = pa.csv.ReadOptions()
        self._read_options.skip_rows_after_names = (
            ropts.skip_rows_after_names + chunk.num_rows
        )
        if self._convert_options is None:
            self._convert_options = pa.csv.ConvertOptions()
        self._convert_options.column_types = self._last_schema
        self._convert_options.null_values = cvopts.null_values
        assert self._read_options.skip_rows_after_names is not None
        self._currow = self._read_options.skip_rows_after_names
        istream = _reopen_last()
        self._parser_func = partial(
            pa.csv.open_csv,
            istream,
            read_options=self._read_options,
            convert_options=self._convert_options,
            parse_options=popts,
        )
        self._parser = None

        return chunk

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if step_size == 0:  # bug
            logger.error("Received a step_size of 0")
            return self._return_run_step(self.state_ready, steps_run=0)
        if self.throttle:
            step_size = np.min([self.throttle, step_size])  # type: ignore
        status = self.validate_parser(run_number)
        if status == self.state_terminated:
            raise ProgressiveStopIteration("no more filenames")
        elif status == self.state_blocked:
            return self._return_run_step(status, steps_run=0)
        elif status != self.state_ready:
            logger.error("Invalid state returned by validate_parser: %d", status)
            self.close()
            raise ProgressiveStopIteration("Unexpected situation")
        logger.info("loading %d lines", step_size)
        pa_batches = []
        cnt = step_size
        creates = 0
        try:
            assert self.parser
            while cnt > 0:
                bat = self.parser.read_next_batch()
                _nrows = bat.num_rows
                cnt -= _nrows
                creates += _nrows
                self._currow += _nrows
                pa_batches.append(bat)
                if self._last_schema is None:
                    self._last_schema = copy.copy(bat.schema)
            assert pa_batches
        except StopIteration:
            self.close()
            if self.has_input_slot("filenames"):
                fn_slot = self.get_input_slot("filenames")
                if (
                    fn_slot is None or fn_slot.output_module is None
                ) and not self._file_mode:
                    raise
            self._parser = None
            self._parser_func = None
            if not pa_batches:
                return self._return_run_step(self.state_ready, steps_run=0)
        except pa.ArrowInvalid:
            if self._drop_na:
                chnk = self.recovering()
                nr = chnk.num_rows
                cnt -= nr
                creates += nr
                pa_batches.append(chnk)
            else:
                raise
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
                self.result = Table(**self._table_params)
                bat_list = bat_list[1:]
                # if self._imputer is not None:
                #    self._imputer.init(bat.dtypes)
            for bat in bat_list:
                self.table.append(bat)
            # if self._imputer is not None:
            #    self._imputer.add_df(bat)
        return self._return_run_step(self.state_ready, steps_run=creates)


csv_docstring = (
    "PACSVLoader("
    + extract_params_docstring(PACSVLoader.__init__, only_defaults=True)
    + ",force_valid_ids=False,id=None,scheduler=None,tracer=None,predictor=None"
    + ",storage=None,input_descriptors=[],output_descriptors=[])"
)
try:
    PACSVLoader.__init__.__func__.__doc__ = csv_docstring  # type: ignore
except Exception:
    try:
        PACSVLoader.__init__.__doc__ = csv_docstring
    except Exception:
        pass
