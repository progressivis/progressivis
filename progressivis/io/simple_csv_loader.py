from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import io
import fsspec  # type: ignore
from .. import ProgressiveError
from ..core.docstrings import FILENAMES_DOC, RESULT_DOC
from ..utils.errors import ProgressiveStopIteration
from ..utils.inspect import filter_kwds, extract_params_docstring
from ..core.module import Module
from ..core.module import ReturnRunStep, def_input, def_output, document
from ..table.table import PTable
from ..table.dshape import dshape_from_dataframe
from ..core.utils import (
    filepath_to_buffer,
    _infer_compression,
    force_valid_id_columns,
    integer_types,
    is_str,
    is_slice,
    nn,
)
from ..utils.psdict import PDict
from ..core.pintset import PIntSet
from typing import (
    Dict,
    Any,
    Type,
    Callable,
    Optional,
    Tuple,
    Union,
    Sequence,
    TYPE_CHECKING,
    cast,
)

from pandas._typing import ReadCsvBuffer

if TYPE_CHECKING:
    from ..core.module import ModuleState
    from ..stats.utils import SimpleImputer

logger = logging.getLogger(__name__)

FSSPEC_HTTPS = fsspec.filesystem("https")


@document
@def_input("filenames", PTable, required=False, doc=FILENAMES_DOC)
@def_output("result", PTable, doc=RESULT_DOC)
@def_output(
    "anomalies",
    PDict,
    required=False,
    doc=(
        "provides invalid values" " as: ``anomalies[id][column]" " = <invalid-value>``"
    ),
)
@def_output(
    "missing",
    PDict,
    required=False,
    doc=("provides missing values as:" " ``missing[column] = <set-of-ids>``"),
)
class SimpleCSVLoader(Module):
    """
    This module reads comma-separated values (csv) files progressively into a {{PTable}}.
    Optionally, it provides information about missing values and anomalies via
    `anomalies` and `missing` output slots.
    Internally it uses :func:`pandas.read_csv`.
    """

    def __init__(
        self,
        filepath_or_buffer: Optional[Any] = None,
        filter_: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        force_valid_ids: bool = True,
        fillvalues: Optional[Dict[str, Any]] = None,
        throttle: Union[bool, int, float] = False,
        imputer: Optional[SimpleImputer] = None,
        **kwds: Any,
    ) -> None:
        r"""
        Args:
            filepath_or_buffer: str, path object or file-like object accepted by :func:`pandas.read_csv`
            filter\_: filtering function to be applied on input data at loading time

                Example:
                    >>> def filter_(df):
                    ...     lon = df['dropoff_longitude']
                    ...     lat = df['dropoff_latitude']
                    ...     return df[(lon>-74.10)&(lon<-73.7)&(lat>40.60)&(lat<41)]
            force_valid_ids: force renaming of columns to make their names valid identifiers according to the `language definition  <https://docs.python.org/3/reference/lexical_analysis.html#identifiers>`_
            fillvalues: the default values of the columns specified as a dictionary (see :class:`PTable <progressivis.PTable>`)
            throttle: limit the number of rows to be loaded in a step
            imputer: a ``SimpleImputer`` provides basic strategies for imputing missing values
            kwds: extra keyword args to be passed to :func:`pandas.read_csv` and :class:`Module <progressivis.core.Module>` superclass
        """
        super().__init__(**kwds)
        self.default_step_size = 1000
        chunksize_ = kwds.get("chunksize")
        if isinstance(chunksize_, int):  # initial guess
            self.default_step_size = chunksize_
        if chunksize_ is None:
            kwds["chunksize"] = self.default_step_size
        else:
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
        self.parser: Optional[pd.io.parsers.readers.TextFileReader] = None
        self.csv_kwds = csv_kwds
        self._compression: Any = csv_kwds.get("compression", "infer")
        csv_kwds["compression"] = None
        self._encoding: Any = csv_kwds.get("encoding", None)
        csv_kwds["encoding"] = None
        self._nrows = csv_kwds.get("nrows")
        csv_kwds["nrows"] = None  # nrows clashes with chunksize

        self._rows_read = 0
        if nn(filter_) and not callable(filter_):
            raise ProgressiveError("filter parameter should be callable or None")
        self._filter: Callable[[pd.DataFrame], pd.DataFrame] | None = filter_
        self._input_stream: io.IOBase | None = None
        self._input_encoding: Optional[str] = None
        self._input_compression: Optional[str] = None
        self._input_size = 0  # length of the file or input stream when available
        self._total_input_size = 0
        self._total_size = 0
        self._last_progress = (0, 0)
        self._n_files = 0
        self._file_mode = False
        self._table_params: Dict[str, Any] = dict(name=self.name, fillvalues=fillvalues)
        self._imputer = imputer
        self._last_opened: Any = None

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
        opt_slot = self.get_output_slot("missing")
        if opt_slot:
            logger.debug("Maintaining missing")
            self.maintain_missing(True)
        else:
            logger.debug("Not maintaining missing")
            self.maintain_missing(False)

    def maintain_anomalies(self, yes: bool = True) -> None:
        if yes and self.anomalies is None:
            self.anomalies = PDict()
        elif not yes:
            self.anomalies = None

    def maintain_missing(self, yes: bool = True) -> None:
        if yes and self.missing is None:
            self.missing = PDict()
        elif not yes:
            self.missing = None

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
        if not self._total_size:
            self._total_size = size
        self.csv_kwds["encoding"] = encoding
        self.csv_kwds["compression"] = compression
        self._last_opened = filepath
        return self._input_stream

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
        self._total_input_size += self._input_size
        self._input_size = 0

    def get_progress(self) -> Tuple[int, int]:
        if (
            self._total_size == 0
            or self._total_input_size == 0
            or self.result is None
            or self._input_stream is None
        ):
            return self._last_progress
        pos = self._input_stream.tell()
        length = len(self.result)
        if length <= 0:
            return self._last_progress
        estimated_row_size = (self._total_input_size + pos) / length
        estimated_size = int(self._total_size / estimated_row_size)
        self._last_progress = length, estimated_size
        return self._last_progress

    def validate_parser(self, run_number: int) -> ModuleState:
        if self.parser is None:
            if nn(self.filepath_or_buffer) and self.has_input_slot("filenames"):
                raise ProgressiveError(
                    "'filepath_or_buffer' parameter and"
                    " 'filenames' slot cannot both be defined "
                )
            if nn(self.filepath_or_buffer):
                try:
                    self.parser = pd.read_csv(
                        cast(ReadCsvBuffer[str], self.open(self.filepath_or_buffer)),
                        **self.csv_kwds,
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
                    assert is_slice(indices)
                    if indices.stop == indices.start:
                        return self.state_blocked
                    filename = df.at[indices.start, "filename"]
                    assert "chunksize" in self.csv_kwds
                    assert isinstance(self.csv_kwds["chunksize"], int)
                    try:
                        self.parser = pd.read_csv(
                            cast(ReadCsvBuffer[str], self.open(filename)),
                            **self.csv_kwds,
                        )
                    except IOError as e:
                        logger.error("Cannot open file %s: %s", filename, e)
                        self.parser = None
                        # fall through
        return self.state_ready

    def refresh_total_size(self) -> None:
        if self._file_mode:
            return
        fn_slot = self.get_input_slot("filenames")
        df = fn_slot.data()
        if df is None or len(df) == self._n_files:
            return
        self._n_files = len(df)
        total_size = 0
        for fname in df["filename"].loc[:]:
            if fname.startswith("https://"):
                total_size += FSSPEC_HTTPS.size(fname)
            elif fname.startswith("buffer://"):
                continue  # TODO: decide if buffer:// is still useful
            else:
                file_stats = os.stat(fname)
                total_size += file_stats.st_size
        self._total_size = total_size

    def recovering(self, step_size: int) -> pd.DataFrame:
        def _reopen_last() -> Any:
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
        skip = PIntSet(range(1, self.parser._currow + 1))  # type: ignore
        kw["skiprows"] = skip
        usecols = self.parser.orig_options.get("usecols")  # type: ignore
        istream = _reopen_last()
        na_filter = bool(kw.get("na_filter", True))
        # reading the same slice with no type constraint
        df = pd.read_csv(
            cast(ReadCsvBuffer[Any], istream),
            usecols=usecols,
            skiprows=cast(Sequence[int], skip),
            na_filter=na_filter,
            nrows=step_size,
        )
        anomalies = defaultdict(dict)  # type: ignore
        missing: Dict[str, PIntSet] = defaultdict(PIntSet)
        assert self.result is not None
        last_id = self.result.last_id
        for col, dt in zip(df.columns, df.dtypes):
            dtt = self.result._column(col).dtype
            if dtt == dt:
                continue
            try:
                df[col] = df[col].astype(dtt)
                continue
            except ValueError:
                pass
            na_: Union[int, float]
            conv_: Union[Type[int], Type[float]]
            imp = self._imputer
            if np.issubdtype(dtt, np.integer):
                na_ = imp.getvalue(col) if imp is not None else np.iinfo(dtt).max
                conv_ = int
            elif np.issubdtype(dtt, np.floating):
                na_ = imp.getvalue(col) if imp is not None else np.nan
                conv_ = float
            else:
                raise ValueError(f"Cannot recover dtype {dtt}")
            arr = np.empty(len(df), dtype=dtt)
            for i, elt in df[col].items():
                try:
                    arr[i] = conv_(elt)
                except Exception:
                    arr[i] = na_
                    if na_ == np.nan and na_filter and elt == "":  # in this case
                        continue  # do not report empty strings as anomalies
                    if TYPE_CHECKING:
                        assert isinstance(i, int)
                    id_ = last_id + i + 1
                    if nn(self.anomalies):
                        anomalies[id_][col] = elt
                    if nn(self.missing):
                        missing[col].add(id_)
            df[col] = arr
        kw["skiprows"].update(
            range(self.parser._currow + 1, self.parser._currow + 1 + step_size)  # type: ignore
        )
        istream = _reopen_last()
        self.parser = pd.read_csv(istream, **kw)
        if self.anomalies is not None:
            self.anomalies.update(anomalies)
        if self.missing is not None:
            for k, v in missing.items():
                if k in self.missing:
                    self.missing[k] |= v
                else:
                    self.missing[k] = v
        return df

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
            assert self.parser is not None
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
            if nn(self.result) and (
                nn(self.anomalies) or nn(self.missing) or nn(self._imputer)
            ):
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
                self.result = PTable(**self._table_params)
                if self._imputer is not None:
                    self._imputer.init(df.dtypes)
            else:
                self.result.append(df)
            # if not self._file_mode:
            #    self.refresh_total_size()
            if self._imputer is not None:
                self._imputer.add_df(df)
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
    if not SimpleCSVLoader.doc_building():
        SimpleCSVLoader.__init__.__func__.__doc__ = csv_docstring  # type: ignore
except Exception:
    try:
        SimpleCSVLoader.__init__.__doc__ = csv_docstring
    except Exception:
        pass
