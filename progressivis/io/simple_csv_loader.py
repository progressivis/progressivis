from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from datasketches import (
    kll_floats_sketch,
    kll_ints_sketch,
    frequent_strings_sketch,
    frequent_items_error_type,
)
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
    is_dict,
    is_slice,
    nn,
)
from ..stats.utils import OnlineMean
from ..utils import PsDict
from ..core.bitmap import bitmap

from typing import Dict, Any, Type, Callable, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.module import ModuleState
    import io

logger = logging.getLogger(__name__)


class SimpleImputer:
    def __init__(
        self,
        strategy: Optional[Union[str, Dict[str, str]]] = None,
        default_strategy: Optional[str] = None,
        fill_values: Optional[
            Union[str, np.number, Dict[str, Union[str, np.number]]]
        ] = None,
    ):
        if nn(default_strategy):
            assert is_str(default_strategy)
            if not is_dict(strategy):
                raise ValueError(
                    "'default_strategy' is allowed" " only when strategy is a dict"
                )
        _strategy: dict = (
            {} if (strategy is None or is_str(strategy)) else strategy
        )  # type: ignore
        assert is_dict(_strategy)
        _default_strategy = default_strategy or "mean"
        self._impute_all: bool = False
        if isinstance(strategy, str):
            _default_strategy = strategy
        self._strategy: Dict = defaultdict(lambda: _default_strategy, **_strategy)
        if is_str(strategy) or strategy is None:
            self._impute_all = True
        if _default_strategy == "constant" or "constant" in _strategy:
            assert nn(fill_values)
            self._fill_values = (
                fill_values if is_dict(fill_values) else defaultdict(lambda: fill_values)  # type: ignore
            )
        self._means: Dict[str, OnlineMean] = {}
        self._medians: Dict[str, Union[kll_ints_sketch, kll_floats_sketch]] = {}
        self._frequent: Dict[str, frequent_strings_sketch] = {}
        self._dtypes: Dict[str, Union[str, np.dtype]] = {}
        self._k = 200  # for sketching

    def init(self, dtypes):
        if self._impute_all:
            self._dtypes = dtypes
        else:
            self._dtypes = {k: v for (k, v) in dtypes.items() if k in self.strategy}
        for col, ty in self._dtypes.items():
            strategy = self.get_strategy(col)
            if strategy == "mean":
                if not np.issubdtype(ty, np.number):
                    raise ValueError(f"{strategy = } not compatible with {ty}")
                self._means[col] = OnlineMean()
            elif strategy == "median":
                if np.issubdtype(ty, np.floating):
                    self._medians[col] = kll_floats_sketch(self._k)
                elif np.issubdtype(ty, np.integer):
                    self._medians[col] = kll_ints_sketch(self._k)
                else:
                    raise ValueError(f"{strategy = } not compatible with {ty}")
            elif strategy == "most_frequent":
                self._frequent[col] = frequent_strings_sketch(self._k)
            elif strategy != "constant":
                raise ValueError(f"Unknown imputation {strategy = }")

    def get_strategy(self, col):
        return self._strategy[col]

    def add_df(self, df):
        for col, dt in self._dtypes.items():
            strategy = self.get_strategy(col)
            if strategy == "constant":
                continue
            add_strategy = getattr(self, f"add_{strategy}")
            add_strategy(col, df[col], dt)

    def add_mean(self, col, val, dt):
        self._means[col].add(val)

    def add_median(self, col, val, dt):
        if np.issubdtype(dt, np.integer):
            sk = kll_ints_sketch(self._k)
        else:
            assert np.issubdtype(dt, np.floating)
            sk = kll_floats_sketch(self._k)
        sk.update(val)
        self._medians[col].merge(sk)

    def add_most_frequent(self, col, val, dt):
        fi = frequent_strings_sketch(self._k)
        for s in val.astype(str):
            fi.update(s)
        self._frequent[col].merge(fi)

    def add_constant(self, col, val, dt):
        pass

    def getvalue(self, col):
        strategy = self.get_strategy(col)
        get_val_strategy = getattr(self, f"get_val_{strategy}")
        return get_val_strategy(col)

    def get_val_mean(self, col):
        return self._means[col].mean

    def get_val_median(self, col):
        return self._medians[col].get_quantile(0.5)

    def get_val_most_frequent(self, col):
        return self._frequent[col].get_frequent_items(
            frequent_items_error_type.NO_FALSE_POSITIVES
        )[0][0]

    def get_val_constant(self, col):
        return self._fill_values[col]


class SimpleCSVLoader(TableModule):
    inputs = [SlotDescriptor("filenames", type=Table, required=False)]
    outputs = [
        SlotDescriptor("anomalies", type=PsDict, required=False),
        SlotDescriptor("missing", type=PsDict, required=False),
    ]

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
        if nn(filter_) and not callable(filter_):
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
        self._imputer = imputer
        self._last_opened: Any = None
        self._anomalies: Optional[PsDict] = None
        self._missing: Optional[PsDict] = None

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
        if yes and self._anomalies is None:
            self._anomalies = PsDict()
        elif not yes:
            self._anomalies = None

    def anomalies(self) -> Optional[PsDict]:
        return self._anomalies

    def maintain_missing(self, yes: bool = True) -> None:
        if yes and self._missing is None:
            self._missing = PsDict()
        elif not yes:
            self._missing = None

    def missing(self) -> Optional[PsDict]:
        return self._missing

    def get_data(self, name: str) -> Any:
        if name == "anomalies":
            return self.anomalies()
        if name == "missing":
            return self.missing()
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
            if nn(self.filepath_or_buffer):
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
                    assert is_slice(indices)
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
        missing: Dict[str, bitmap] = defaultdict(bitmap)
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
                    id_ = last_id + i + 1
                    if nn(self._anomalies):
                        anomalies[id_][col] = elt
                    if nn(self._missing):
                        missing[col].add(id_)
            df[col] = arr
        kw["skiprows"].update(
            range(self.parser._currow + 1, self.parser._currow + 1 + step_size)  # type: ignore
        )
        istream = _reopen_last()
        self.parser = pd.read_csv(istream, **kw)
        if self._anomalies is not None:
            self._anomalies.update(anomalies)
        if self._missing is not None:
            for k, v in missing.items():
                if k in self._missing:
                    self._missing[k] |= v
                else:
                    self._missing[k] = v
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
            if nn(self.table) and (
                nn(self.anomalies()) or nn(self.missing()) or nn(self._imputer)
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
                self.result = Table(**self._table_params)
                if self._imputer is not None:
                    self._imputer.init(df.dtypes)
            else:
                self.table.append(df)
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
    SimpleCSVLoader.__init__.__func__.__doc__ = csv_docstring  # type: ignore
except Exception:
    try:
        SimpleCSVLoader.__init__.__doc__ = csv_docstring
    except Exception:
        pass
