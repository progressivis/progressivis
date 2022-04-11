from __future__ import annotations

import logging

import pandas as pd
import numpy as np
from collections import defaultdict
from .. import ProgressiveError, SlotDescriptor
from ..utils.errors import ProgressiveStopIteration
from ..utils.inspect import filter_kwds, extract_params_docstring
from ..table.module import TableModule
from ..core.module import ReturnRunStep
from ..table.table import Table
from ..table.dshape import dshape_from_dataframe
from ..core.utils import (
    filepath_to_buffer,
    _infer_compression,
    force_valid_id_columns,
    integer_types,
    is_str,
)
from ..utils import PsDict

from typing import Dict, Any, Callable, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.module import ModuleState
    import io

logger = logging.getLogger(__name__)


class SimpleCSVLoader(TableModule):
    inputs = [SlotDescriptor("filenames", type=Table, required=False)]
    outputs = [
        SlotDescriptor("anomalies", type=PsDict, required=False),
    ]

    def __init__(
        self,
        filepath_or_buffer: Optional[Any] = None,
        filter_: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        force_valid_ids: bool = True,
        fillvalues: Optional[Dict[str, Any]] = None,
        throttle: Union[bool, int, float] = False,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.default_step_size = kwds.get("chunksize", 1000)  # initial guess
        kwds.setdefault("chunksize", self.default_step_size)
        # Filter out the module keywords from the csv loader keywords
        csv_kwds: Dict[str, Any] = filter_kwds(kwds, pd.read_csv)
        # When called with a specified chunksize, it returns a parser
        self.filepath_or_buffer = filepath_or_buffer
        self.force_valid_ids = force_valid_ids
        if throttle and isinstance(throttle, integer_types + (float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        self.parser: Optional[pd.TextReader] = None
        self.csv_kwds = csv_kwds
        self._compression: Any = csv_kwds.get("compression", "infer")
        csv_kwds["compression"] = None
        self._encoding: Any = csv_kwds.get("encoding", None)
        csv_kwds["encoding"] = None
        self._nrows = csv_kwds.get("nrows")
        csv_kwds["nrows"] = None  # nrows clashes with chunksize

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
        self._last_opened: Any = None
        self._anomalies: Optional[PsDict] = None

    def rows_read(self) -> int:
        return self._rows_read

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
            self._anomalies = PsDict()
        elif not yes:
            self._anomalies = None

    def anomalies(self) -> Optional[PsDict]:
        return self._anomalies

    def get_data(self, name: str) -> Any:
        if name == "anomalies":
            return self.anomalies()
        return super().get_data(name)

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
        self._last_opened = filepath
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
                        self.parser = pd.read_csv(self.open(filename), **self.csv_kwds)
                    except IOError as e:
                        logger.error("Cannot open file %s: %s", filename, e)
                        self.parser = None
                        # fall through
        return self.state_ready

    def recovering(self, step_size: int) -> pd.DataFrame:
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

        assert self.csv_kwds.get("skiprows", 0) <= self.parser._currow  # type: ignore
        kw = self.csv_kwds.copy()
        skip = set(range(1, self.parser._currow + 1))  # type: ignore
        kw["skiprows"] = skip
        usecols = self.parser.orig_options.get("usecols")  # type: ignore
        istream = _reopen_last()
        na_filter = kw.get("na_filter")
        # reading the same slice with no type constraint
        df = pd.read_csv(
            istream,
            usecols=usecols,
            skiprows=skip,
            na_filter=na_filter,
            nrows=step_size,
        )  # type: ignore
        anomalies = defaultdict(dict)  # type: ignore
        last_id = self.table.last_id
        for col, dt in zip(df.columns, df.dtypes):
            dtt = self.table._column(col).dtype
            if dtt == dt:
                continue
            try:
                df[col] = df[col].astype(dtt)
                continue
            except ValueError:
                pass
            if np.issubdtype(dtt, np.integer):
                na_int = np.iinfo(dtt).max
                arr = np.empty(len(df), dtype=dtt)
                for i, elt in df[col].items():
                    try:
                        arr[i] = int(elt)
                    except Exception:
                        arr[i] = na_int
                        anomalies[last_id + i + 1][col] = elt
                df[col] = arr
            elif np.issubdtype(dtt, np.floating):
                arr = np.empty(len(df), dtype=dtt)
                for i, elt in df[col].items():
                    try:
                        arr[i] = float(elt)
                    except Exception:
                        arr[i] = np.nan
                        if na_filter and elt == "":  # in this case
                            continue  # do not report empty strings as anomalies
                        anomalies[last_id + i + 1][col] = elt
                df[col] = arr

            else:
                raise ValueError(f"Cannot recover dtype {dtt}")
        kw["skiprows"].update(
            range(self.parser._currow + 1, self.parser._currow + 1 + step_size)  # type: ignore
        )
        istream = _reopen_last()
        self.parser = pd.read_csv(istream, **kw)
        assert self._anomalies is not None
        self._anomalies.update(anomalies)
        return df

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
        except ValueError:
            if self.table is not None and self.anomalies() is not None:
                df = self.recovering(step_size)
            else:
                raise

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
