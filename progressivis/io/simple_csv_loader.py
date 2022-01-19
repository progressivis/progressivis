from __future__ import annotations

import logging

import pandas as pd
import numpy as np
from progressivis import ProgressiveError, SlotDescriptor
from progressivis.utils.errors import ProgressiveStopIteration
from progressivis.utils.inspect import filter_kwds, extract_params_docstring
from progressivis.table.module import TableModule
from progressivis.core.module import ReturnRunStep
from progressivis.table.table import Table
from progressivis.table.dshape import dshape_from_dataframe
from progressivis.core.utils import (
    filepath_to_buffer,
    _infer_compression,
    force_valid_id_columns,
)

from typing import Dict, Any, Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import ModuleState
    import io

logger = logging.getLogger(__name__)


class SimpleCSVLoader(TableModule):
    inputs = [SlotDescriptor("filenames", type=Table, required=False)]

    def __init__(
        self,
        filepath_or_buffer: Optional[Any] = None,
        filter_: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        force_valid_ids: bool = True,
        fillvalues: Optional[Dict[str, Any]] = None,
        throttle: bool = False,    
        **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.default_step_size = kwds.get("chunksize", 1000)  # initial guess
        kwds.setdefault("chunksize", self.default_step_size)
        # Filter out the module keywords from the csv loader keywords
        csv_kwds: Dict[str, Any] = filter_kwds(kwds, pd.read_csv)
        # When called with a specified chunksize, it returns a parser
        self.filepath_or_buffer = filepath_or_buffer
        self.force_valid_ids = force_valid_ids
        if throttle and isinstance(throttle, integer_types+(float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        self.parser: Optional[pd.TextReader] = None
        self.csv_kwds = csv_kwds
        self._compression: Any = csv_kwds.get("compression", "infer")
        csv_kwds["compression"] = None
        self._encoding: Any = csv_kwds.get("encoding", None)
        csv_kwds["encoding"] = None
        self._nrows = csv_kwds.get('nrows')
        csv_kwds['nrows'] = None # nrows clashes with chunksize

        self._rows_read = 0
        if filter_ is not None and not callable(filter_):
            raise ProgressiveError("filter parameter should be callable or None")
        self._filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = filter_
        self._input_stream: Optional[
            io.IOBase
        ] = None  # stream that returns a position through the 'tell()' method
        self._input_encoding: Optional[str] = None
        self._input_compression: Optional[str] = None
        self._input_size = 0  # length of the file or input stream when available
        self._file_mode = False
        self._table_params: Dict[str, Any] = dict(name=self.name, fillvalues=fillvalues)

    def rows_read(self) -> int:
        return self._rows_read

    def is_ready(self) -> bool:
        if self.has_input_slot("filenames"):
            # Can be called before the first update so fn.created can be None
            fn = self.get_input_slot("filenames")
            if fn.created is None or fn.created.any():
                return True
        return super().is_ready()

    def is_data_input(self) -> bool:
        # pylint: disable=no-self-use
        "Return True if this module brings new data"
        return True

    def open(self, filepath: Any) -> io.IOBase:
        if self._input_stream is not None:
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
        self.csv_kwds["encoding"] = encoding
        self.csv_kwds["compression"] = compression
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
        if self.parser is None:
            if self.filepath_or_buffer is not None:
                try:
                    self.parser = pd.read_csv(
                        self.open(self.filepath_or_buffer), **self.csv_kwds
                    )
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
                    assert isinstance(indices, slice)
                    if indices.stop == indices.start:
                        return self.state_blocked
                    filename = df.at[indices.start, "filename"]
                    try:
                        self.parser = pd.read_csv(
                            self.open(filename), **self.csv_kwds
                        )
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
            self.close()
            raise ProgressiveStopIteration("Unexpected situation")
        logger.info("loading %d lines", step_size)
        try:
            assert self.parser
            df: pd.DataFrame = self.parser.read(
                step_size
            )  # raises StopIteration at EOF
        except StopIteration:
            self.close()
            if self.has_input_slot("filenames"):
                fn_slot = self.get_input_slot("filenames")
                if (
                    fn_slot is None or fn_slot.output_module is None
                ) and not self._file_mode:
                    raise
            self.parser = None
            return self._return_run_step(self.state_ready, steps_run=0)
        creates = len(df)
        if creates == 0:  # should not happen
            logger.error("Received 0 elements")
            raise ProgressiveStopIteration
        if self._filter is not None:
            df = self._filter(df)
        creates = len(df)
        if creates == 0:
            logger.info("frame has been filtered out")
        else:
            self._rows_read += creates
            logger.info("Loaded %d lines", self._rows_read)
            if self.force_valid_ids:
                force_valid_id_columns(df)
            if self.result is None:
                self._table_params["name"] = self.generate_table_name("table")
                self._table_params["dshape"] = dshape_from_dataframe(df)
                self._table_params["data"] = df
                self._table_params["create"] = True
                self.result = Table(**self._table_params)
            else:
                self.table.append(df)
        return self._return_run_step(self.state_ready, steps_run=creates)


csv_docstring = (
    "SimpleCSVLoader("
    + extract_params_docstring(pd.read_csv)
    + ","
    + extract_params_docstring(SimpleCSVLoader.__init__, only_defaults=True)
    + ",force_valid_ids=False,id=None,scheduler=None,tracer=None,predictor=None"
    + ",storage=None,input_descriptors=[],output_descriptors=[])"
)
try:
    SimpleCSVLoader.__init__.__func__.__doc__ = csv_docstring  # type: ignore
except Exception:
    try:
        SimpleCSVLoader.__init__.__doc__ = csv_docstring
    except Exception:
        pass
